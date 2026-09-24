//! Frame loop: frames in flight, chunk uploads, GPU culling for the camera and
//! each shadow cascade, shadow passes, then sky and chunks.
const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const world = @import("world");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");
const Swapchain = @import("Swapchain.zig");
const pipeline = @import("pipeline.zig");
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const ChunkBuffers = @import("ChunkBuffers.zig");
const Shadows = @import("Shadows.zig");
const gpu = @import("gpu.zig");
const Camera = @import("../Camera.zig");
const Sun = @import("../Sun.zig");

const Renderer = @This();

pub const frames_in_flight = 2;
const depth_format: vk.Format = .d32_sfloat;
/// Views culled each frame: the camera, then one per shadow cascade.
const views = 1 + Shadows.cascades;
/// Indirect range filled with the camera's water groups (after the culled views).
const water_view = views;
comptime {
    if (water_view != 4) @compileError("cull.comp hard-codes water_view = 4");
}
/// Indirect ranges and counters: one per culled view, plus water.
const regions = views + 1;
const max_draws = ChunkBuffers.max_chunks * 6;
const chunk_stages: vk.ShaderStageFlags = .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true };

const Frame = struct {
    pool: vk.CommandPool,
    cmd: vk.CommandBuffer,
    image_acquired: vk.Semaphore,
    fence: vk.Fence,
    /// Written by the CPU while the frame is recorded, read by the GPU.
    frame_data: Buffer,
    staging: Buffer,
};

pub const Options = struct {
    shadow_resolution: u32 = 2048,
};

pub const FrameInput = struct {
    camera: Camera,
    light: Sun.Lighting,
    /// Shadows are skipped at night (the moon casts none).
    shadows: bool,
    /// Block to outline (the one the player aims at).
    target: ?world.BlockPos,
    /// The camera is inside a water block: blue fog and tint.
    underwater: bool,
    fog_start: f32,
    fog_end: f32,
};

gpa: Allocator,
ctx: *const Context,
swapchain: Swapchain,
/// Framebuffer size the swapchain was last built for (not the clamped one).
requested_extent: vk.Extent2D,
depth: Image,
frames: [frames_in_flight]Frame,
frame_index: usize = 0,
sky_layout: vk.PipelineLayout,
sky_pipeline: vk.Pipeline,
chunks: ChunkBuffers,
shadows: Shadows,
draws: Buffer,
draw_count: Buffer,
set_layout: vk.DescriptorSetLayout,
chunk_layout: vk.PipelineLayout,
cull_pipeline: vk.Pipeline,
chunk_pipeline: vk.Pipeline,
shadow_pipeline: vk.Pipeline,
water_pipeline: vk.Pipeline,
outline_layout: vk.PipelineLayout,
outline_pipeline: vk.Pipeline,

pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D, options: Options) !Renderer {
    const d = ctx.device;
    var swapchain: Swapchain = try .init(ctx, gpa, extent, .null_handle);
    errdefer swapchain.deinit(ctx, gpa);
    var depth: Image = try createDepth(ctx, swapchain.extent);
    errdefer depth.deinit(ctx);

    var frames: [frames_in_flight]Frame = undefined;
    var frames_done: usize = 0;
    errdefer for (frames[0..frames_done]) |*f| destroyFrame(ctx, f);
    for (&frames) |*f| {
        f.* = try createFrame(ctx);
        frames_done += 1;
    }

    var chunks: ChunkBuffers = try .init(ctx, gpa);
    errdefer chunks.deinit(ctx);
    var shadows: Shadows = try .init(ctx, options.shadow_resolution);
    errdefer shadows.deinit(ctx);
    var draws: Buffer = try .init(ctx, regions * max_draws * @sizeOf(gpu.DrawCmd), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true }, false);
    errdefer draws.deinit(ctx);
    var draw_count: Buffer = try .init(ctx, regions * @sizeOf(u32), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true, .transfer_dst_bit = true }, false);
    errdefer draw_count.deinit(ctx);

    // The shadow map is bound with a push descriptor: no pool, no sets.
    // Binding 1: the scene depth, read in place by the water pass (input attachment).
    const bindings = [_]vk.DescriptorSetLayoutBinding{
        .{ .binding = 0, .descriptor_type = .combined_image_sampler, .descriptor_count = 1, .stage_flags = .{ .fragment_bit = true } },
        .{ .binding = 1, .descriptor_type = .input_attachment, .descriptor_count = 1, .stage_flags = .{ .fragment_bit = true } },
    };
    const set_layout = try d.createDescriptorSetLayout(&.{ .flags = .{ .push_descriptor_bit = true }, .binding_count = bindings.len, .p_bindings = &bindings }, null);
    errdefer d.destroyDescriptorSetLayout(set_layout, null);

    const sky_layout = try pipeline.createLayout(ctx, @sizeOf(SkyPush), .{ .fragment_bit = true }, &.{});
    errdefer d.destroyPipelineLayout(sky_layout, null);
    const chunk_layout = try pipeline.createLayout(ctx, @sizeOf(gpu.Push), chunk_stages, &.{set_layout});
    errdefer d.destroyPipelineLayout(chunk_layout, null);

    const sky_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = sky_layout,
        .vertex = pipeline.spirv("fullscreen.vert"),
        .fragment = pipeline.spirv("sky.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
        .depth_test = false,
        .depth_write = false,
        .cull_back = false,
    });
    errdefer d.destroyPipeline(sky_pipeline, null);
    const cull_pipeline = try pipeline.createCompute(ctx, chunk_layout, pipeline.spirv("cull.comp"));
    errdefer d.destroyPipeline(cull_pipeline, null);
    const chunk_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = chunk_layout,
        .vertex = pipeline.spirv("chunk.vert"),
        .fragment = pipeline.spirv("chunk.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
    });
    errdefer d.destroyPipeline(chunk_pipeline, null);
    // Casters beyond the cascade's near plane are clamped instead of clipped.
    const shadow_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = chunk_layout,
        .vertex = pipeline.spirv("shadow.vert"),
        .fragment = null,
        .color_format = null,
        .depth_format = Shadows.format,
        .depth_clamp = true,
        .depth_bias = true,
    });
    errdefer d.destroyPipeline(shadow_pipeline, null);
    // Transparent water: blended over the opaque scene, depth-tested but not
    // written, both sides visible (from under water too).
    const water_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = chunk_layout,
        .vertex = pipeline.spirv("chunk.vert"),
        .fragment = pipeline.spirv("water.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
        .depth_write = false,
        .cull_back = false,
        .blend = true,
        .reads_depth = true,
    });
    errdefer d.destroyPipeline(water_pipeline, null);
    const outline_layout = try pipeline.createLayout(ctx, @sizeOf(OutlinePush), .{ .vertex_bit = true }, &.{});
    errdefer d.destroyPipelineLayout(outline_layout, null);
    const outline_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = outline_layout,
        .vertex = pipeline.spirv("outline.vert"),
        .fragment = pipeline.spirv("outline.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
        .depth_write = false,
        .cull_back = false,
        .topology = .line_list,
    });

    return .{
        .gpa = gpa,
        .ctx = ctx,
        .swapchain = swapchain,
        .requested_extent = extent,
        .depth = depth,
        .frames = frames,
        .sky_layout = sky_layout,
        .sky_pipeline = sky_pipeline,
        .chunks = chunks,
        .shadows = shadows,
        .draws = draws,
        .draw_count = draw_count,
        .set_layout = set_layout,
        .chunk_layout = chunk_layout,
        .cull_pipeline = cull_pipeline,
        .chunk_pipeline = chunk_pipeline,
        .shadow_pipeline = shadow_pipeline,
        .water_pipeline = water_pipeline,
        .outline_layout = outline_layout,
        .outline_pipeline = outline_pipeline,
    };
}

pub fn deinit(self: *Renderer) void {
    const d = self.ctx.device;
    d.deviceWaitIdle() catch {};
    d.destroyPipeline(self.outline_pipeline, null);
    d.destroyPipelineLayout(self.outline_layout, null);
    d.destroyPipeline(self.water_pipeline, null);
    d.destroyPipeline(self.shadow_pipeline, null);
    d.destroyPipeline(self.chunk_pipeline, null);
    d.destroyPipeline(self.cull_pipeline, null);
    d.destroyPipeline(self.sky_pipeline, null);
    d.destroyPipelineLayout(self.chunk_layout, null);
    d.destroyPipelineLayout(self.sky_layout, null);
    d.destroyDescriptorSetLayout(self.set_layout, null);
    self.draw_count.deinit(self.ctx);
    self.draws.deinit(self.ctx);
    self.shadows.deinit(self.ctx);
    self.chunks.deinit(self.ctx);
    for (&self.frames) |*f| destroyFrame(self.ctx, f);
    self.depth.deinit(self.ctx);
    self.swapchain.deinit(self.ctx, self.gpa);
}

fn createFrame(ctx: *const Context) !Frame {
    const d = ctx.device;
    const pool = try d.createCommandPool(&.{ .flags = .{ .reset_command_buffer_bit = true }, .queue_family_index = ctx.queue_family }, null);
    errdefer d.destroyCommandPool(pool, null);
    var cmd: vk.CommandBuffer = undefined;
    try d.allocateCommandBuffers(&.{ .command_pool = pool, .level = .primary, .command_buffer_count = 1 }, @ptrCast(&cmd));
    const image_acquired = try d.createSemaphore(&.{}, null);
    errdefer d.destroySemaphore(image_acquired, null);
    const fence = try d.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
    errdefer d.destroyFence(fence, null);
    var frame_data: Buffer = try .init(ctx, @sizeOf(gpu.FrameData), .{ .storage_buffer_bit = true }, true);
    errdefer frame_data.deinit(ctx);
    const staging: Buffer = try .init(ctx, ChunkBuffers.staging_size, .{ .transfer_src_bit = true }, true);
    return .{ .pool = pool, .cmd = cmd, .image_acquired = image_acquired, .fence = fence, .frame_data = frame_data, .staging = staging };
}

fn destroyFrame(ctx: *const Context, f: *Frame) void {
    f.staging.deinit(ctx);
    f.frame_data.deinit(ctx);
    ctx.device.destroyFence(f.fence, null);
    ctx.device.destroySemaphore(f.image_acquired, null);
    ctx.device.destroyCommandPool(f.pool, null);
}

const SkyPush = extern struct {
    inv_view_proj: zm.Mat,
    sun_dir: [4]f32,
};

const OutlinePush = extern struct {
    view_proj: zm.Mat,
    block: [3]f32,
};

/// Renders one frame. `extent` is the current framebuffer size (for resizes).
pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
    const d = self.ctx.device;
    const frame = &self.frames[self.frame_index];
    _ = try d.waitForFences(&.{frame.fence}, .true, std.math.maxInt(u64));

    const acquired = d.acquireNextImageKHR(self.swapchain.handle, std.math.maxInt(u64), frame.image_acquired, .null_handle) catch |err| switch (err) {
        error.OutOfDateKHR => return self.recreate(extent),
        else => return err,
    };
    // Reset only once an image is acquired: an early return must leave the fence signalled.
    try d.resetFences(&.{frame.fence});
    const image_index = acquired.image_index;

    const cmd: vk.CommandBufferProxy = .init(frame.cmd, self.ctx.vkd);
    try cmd.resetCommandBuffer(.{});
    try cmd.beginCommandBuffer(&.{ .flags = .{ .one_time_submit_bit = true } });

    const ext = self.swapchain.extent;
    const aspect = @as(f32, @floatFromInt(ext.width)) / @as(f32, @floatFromInt(ext.height));
    const view_proj = in.camera.viewProj(aspect);
    // Uploads first: slots allocated this frame are then culled this frame.
    try self.chunks.record(cmd, &frame.staging);
    self.writeFrameData(frame, in, view_proj, aspect);

    // 1. Chunk uploads, then GPU culling for every view into the indirect buffer.
    var push: gpu.Push = .{
        .frame = frame.frame_data.address,
        .metas = self.chunks.metas.address,
        .quads = self.chunks.quads.address,
        .draws = self.draws.address,
        .count = self.draw_count.address,
        .view = 0,
    };
    cmd.fillBuffer(self.draw_count.handle, 0, regions * @sizeOf(u32), 0);
    ChunkBuffers.barrier(cmd, .{ .copy_bit = true, .clear_bit = true }, .{ .transfer_write_bit = true }, .{ .compute_shader_bit = true, .vertex_shader_bit = true }, .{ .shader_storage_read_bit = true, .shader_storage_write_bit = true });
    cmd.bindPipeline(.compute, self.cull_pipeline);
    const groups = std.math.divCeil(u32, self.chunks.slot_high, 64) catch unreachable;
    const culled_views: u32 = if (in.shadows) views else 1;
    for (0..culled_views) |v| {
        push.view = @intCast(v);
        cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
        cmd.dispatch(groups, 1, 1);
    }
    ChunkBuffers.barrier(cmd, .{ .compute_shader_bit = true }, .{ .shader_storage_write_bit = true }, .{ .draw_indirect_bit = true }, .{ .indirect_command_read_bit = true });

    // 2. Shadow cascades, depth only. Cleared even when skipped: the lighting pass samples them.
    const shadow_image = self.shadows.image.image;
    imageBarrier(cmd, shadow_image, .{ .depth_bit = true }, .undefined, .depth_attachment_optimal, .{ .fragment_shader_bit = true }, .{}, .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true });
    const res = self.shadows.resolution;
    for (self.shadows.layer_views, 1..) |layer_view, v| {
        const att: vk.RenderingAttachmentInfo = .{
            .image_view = layer_view,
            .image_layout = .depth_attachment_optimal,
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .depth_stencil = .{ .depth = 0, .stencil = 0 } },
        };
        cmd.beginRendering(&.{
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = res, .height = res } },
            .layer_count = 1,
            .view_mask = 0,
            .color_attachment_count = 0,
            .p_depth_attachment = &att,
        });
        if (in.shadows) {
            setViewport(cmd, .{ .width = res, .height = res });
            // Reverse-Z: a negative bias pushes casters away from the light.
            cmd.setDepthBias(-1.5, 0, -2.0);
            cmd.bindPipeline(.graphics, self.shadow_pipeline);
            push.view = @intCast(v);
            cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
            cmd.drawIndirectCount(self.draws.handle, v * max_draws * @sizeOf(gpu.DrawCmd), self.draw_count.handle, v * @sizeOf(u32), max_draws, @sizeOf(gpu.DrawCmd));
        }
        cmd.endRendering();
    }
    imageBarrier(cmd, shadow_image, .{ .depth_bit = true }, .depth_attachment_optimal, .depth_read_only_optimal, .{ .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .fragment_shader_bit = true }, .{ .shader_sampled_read_bit = true });

    // 3. Main pass: sky, then chunks.
    const image = self.swapchain.images[image_index];
    imageBarrier(cmd, image, .{ .color_bit = true }, .undefined, .color_attachment_optimal, .{ .color_attachment_output_bit = true }, .{}, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true });
    // The depth stays in RENDERING_LOCAL_READ for the whole pass: the water reads it in place.
    imageBarrier(cmd, self.depth.image, .{ .depth_bit = true }, .undefined, .rendering_local_read, .{ .late_fragment_tests_bit = true, .fragment_shader_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true, .fragment_shader_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true, .input_attachment_read_bit = true });
    const color_att: vk.RenderingAttachmentInfo = .{
        .image_view = self.swapchain.views[image_index],
        .image_layout = .color_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .dont_care,
        .store_op = .store,
        .clear_value = .{ .color = .{ .float_32 = .{ 0, 0, 0, 1 } } },
    };
    const depth_att: vk.RenderingAttachmentInfo = .{
        .image_view = self.depth.view,
        .image_layout = .rendering_local_read,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .clear,
        .store_op = .dont_care,
        .clear_value = .{ .depth_stencil = .{ .depth = 0, .stencil = 0 } },
    };
    cmd.beginRendering(&.{
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = ext },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_att),
        .p_depth_attachment = &depth_att,
    });
    setViewport(cmd, ext);

    const d_sun = in.light.dir;
    const sky: SkyPush = .{ .inv_view_proj = zm.inverse(view_proj), .sun_dir = .{ d_sun[0], d_sun[1], d_sun[2], if (in.underwater) 1 else 0 } };
    cmd.bindPipeline(.graphics, self.sky_pipeline);
    cmd.pushConstants(self.sky_layout, .{ .fragment_bit = true }, 0, @sizeOf(SkyPush), &sky);
    cmd.draw(3, 1, 0, 0);

    const shadow_info: vk.DescriptorImageInfo = .{ .sampler = self.shadows.sampler, .image_view = self.shadows.image.view, .image_layout = .depth_read_only_optimal };
    const depth_info: vk.DescriptorImageInfo = .{ .sampler = .null_handle, .image_view = self.depth.view, .image_layout = .rendering_local_read };
    cmd.pushDescriptorSet(.graphics, self.chunk_layout, 0, &.{ .{
        .dst_set = .null_handle,
        .dst_binding = 0,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = .combined_image_sampler,
        .p_image_info = @ptrCast(&shadow_info),
        .p_buffer_info = undefined,
        .p_texel_buffer_view = undefined,
    }, .{
        .dst_set = .null_handle,
        .dst_binding = 1,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = .input_attachment,
        .p_image_info = @ptrCast(&depth_info),
        .p_buffer_info = undefined,
        .p_texel_buffer_view = undefined,
    } });
    cmd.bindPipeline(.graphics, self.chunk_pipeline);
    push.view = 0;
    cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
    cmd.drawIndirectCount(self.draws.handle, 0, self.draw_count.handle, 0, max_draws, @sizeOf(gpu.DrawCmd));

    // Water, inside the same pass: make this pass's depth writes visible to the
    // water's in-place depth reads (by region: each pixel only reads itself).
    const local_read: vk.ImageMemoryBarrier2 = .{
        .src_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
        .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
        .dst_stage_mask = .{ .fragment_shader_bit = true },
        .dst_access_mask = .{ .input_attachment_read_bit = true },
        .old_layout = .rendering_local_read,
        .new_layout = .rendering_local_read,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = self.depth.image,
        .subresource_range = .{ .aspect_mask = .{ .depth_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
    };
    cmd.pipelineBarrier2(&.{ .dependency_flags = .{ .by_region_bit = true }, .image_memory_barrier_count = 1, .p_image_memory_barriers = @ptrCast(&local_read) });
    cmd.setRenderingInputAttachmentIndices(&pipeline.depth_input_mapping);
    cmd.bindPipeline(.graphics, self.water_pipeline);
    push.view = water_view;
    cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
    cmd.drawIndirectCount(self.draws.handle, water_view * max_draws * @sizeOf(gpu.DrawCmd), self.draw_count.handle, water_view * @sizeOf(u32), max_draws, @sizeOf(gpu.DrawCmd));
    cmd.setRenderingInputAttachmentIndices(&pipeline.default_input_mapping);

    if (in.target) |t| {
        const outline: OutlinePush = .{ .view_proj = view_proj, .block = .{ @floatFromInt(t.x), @floatFromInt(t.y), @floatFromInt(t.z) } };
        cmd.bindPipeline(.graphics, self.outline_pipeline);
        cmd.pushConstants(self.outline_layout, .{ .vertex_bit = true }, 0, @sizeOf(OutlinePush), &outline);
        cmd.draw(24, 1, 0, 0);
    }

    cmd.endRendering();
    imageBarrier(cmd, image, .{ .color_bit = true }, .color_attachment_optimal, .present_src_khr, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true }, .{}, .{});
    try cmd.endCommandBuffer();

    const render_done = self.swapchain.render_done[image_index];
    try self.ctx.queue.submit2(&.{.{
        .wait_semaphore_info_count = 1,
        .p_wait_semaphore_infos = &.{.{ .semaphore = frame.image_acquired, .value = 0, .stage_mask = .{ .color_attachment_output_bit = true }, .device_index = 0 }},
        .command_buffer_info_count = 1,
        .p_command_buffer_infos = &.{.{ .command_buffer = frame.cmd, .device_mask = 0 }},
        .signal_semaphore_info_count = 1,
        .p_signal_semaphore_infos = &.{.{ .semaphore = render_done, .value = 0, .stage_mask = .{ .all_commands_bit = true }, .device_index = 0 }},
    }}, frame.fence);

    self.frame_index = (self.frame_index + 1) % frames_in_flight;
    const present = self.ctx.queue.presentKHR(&.{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&render_done),
        .swapchain_count = 1,
        .p_swapchains = @ptrCast(&self.swapchain.handle),
        .p_image_indices = @ptrCast(&image_index),
    }) catch |err| switch (err) {
        error.OutOfDateKHR => return self.recreate(extent),
        else => return err,
    };
    if (present == .suboptimal_khr or extent.width != self.requested_extent.width or extent.height != self.requested_extent.height)
        try self.recreate(extent);
}

fn writeFrameData(self: *Renderer, frame: *Frame, in: FrameInput, view_proj: zm.Mat, aspect: f32) void {
    var palette: [8][4]f32 = undefined;
    for (&palette, 0..) |*c, i| {
        const rgb = @as(world.Block, @enumFromInt(i)).color();
        c.* = .{ rgb[0], rgb[1], rgb[2], 1 };
    }
    const cam = in.camera;
    const l = in.light;
    const cascades = Shadows.fitCascades(self.shadows.resolution, cam.pos, cam.forward(), cam.fov_y, aspect, cam.near, l.dir);
    var cascade_vp: [Shadows.cascades]zm.Mat = undefined;
    var cascade_planes: [Shadows.cascades * 6][4]f32 = undefined;
    var cascade_texel: [4]f32 = .{ 0, 0, 0, 0 };
    for (cascades, 0..) |c, i| {
        cascade_vp[i] = c.view_proj;
        @memcpy(cascade_planes[i * 6 ..][0..6], &gpu.frustumPlanes(c.view_proj));
        cascade_texel[i] = c.texel;
    }

    const data: *gpu.FrameData = @ptrCast(@alignCast(frame.frame_data.mapped.?));
    data.* = .{
        .view_proj = view_proj,
        .planes = gpu.frustumPlanes(view_proj),
        .camera_pos = .{ cam.pos[0], cam.pos[1], cam.pos[2], 1 },
        .sun_dir = .{ l.dir[0], l.dir[1], l.dir[2], 0 },
        .sun_color = .{ l.color[0], l.color[1], l.color[2], 0 },
        .ambient = .{ l.ambient[0], l.ambient[1], l.ambient[2], 0 },
        .fog = .{ in.fog_start, in.fog_end, cam.near, if (in.underwater) 1 else 0 },
        .palette = palette,
        .cascade_vp = cascade_vp,
        .cascade_planes = cascade_planes,
        .cascade_splits = .{ Shadows.splits[0], Shadows.splits[1], Shadows.splits[2], 0 },
        .cascade_texel = cascade_texel,
        .shadow = .{ if (in.shadows) 1 else 0, 0, 0, 0 },
        .chunk_capacity = self.chunks.slot_high,
        .max_draws = max_draws,
    };
}

/// Rebuilds swapchain and depth buffer for `extent`. Atomic: on failure the
/// current ones stay valid.
fn recreate(self: *Renderer, extent: vk.Extent2D) !void {
    if (extent.width == 0 or extent.height == 0) return; // minimized
    // deviceWaitIdle only waits on queue work, not the presentation engine: a
    // per-image render_done semaphore the old swapchain owns could still be
    // consumed by an in-flight present when it is destroyed below.
    // VK_EXT_swapchain_maintenance1 (vkWaitForPresentKHR / per-present fences)
    // would close that gap; not required here.
    try self.ctx.device.deviceWaitIdle();
    var swapchain = Swapchain.init(self.ctx, self.gpa, extent, self.swapchain.handle) catch |err| switch (err) {
        error.ZeroExtent => return, // minimized between the size query and now
        else => return err,
    };
    errdefer swapchain.deinit(self.ctx, self.gpa);
    const depth: Image = try createDepth(self.ctx, swapchain.extent);
    self.swapchain.deinit(self.ctx, self.gpa); // the old handle is retired, destroying it is valid
    self.depth.deinit(self.ctx);
    self.swapchain = swapchain;
    self.depth = depth;
    self.requested_extent = extent;
}

/// Scene depth: a depth attachment the water pass also reads as input attachment.
fn createDepth(ctx: *const Context, extent: vk.Extent2D) !Image {
    return .init(ctx, extent, depth_format, .{ .depth_stencil_attachment_bit = true, .input_attachment_bit = true }, .{ .depth_bit = true }, 1);
}

fn setViewport(cmd: vk.CommandBufferProxy, ext: vk.Extent2D) void {
    cmd.setViewport(0, &.{.{ .x = 0, .y = 0, .width = @floatFromInt(ext.width), .height = @floatFromInt(ext.height), .min_depth = 0, .max_depth = 1 }});
    cmd.setScissor(0, &.{.{ .offset = .{ .x = 0, .y = 0 }, .extent = ext }});
}

pub fn imageBarrier(
    cmd: vk.CommandBufferProxy,
    image: vk.Image,
    aspect: vk.ImageAspectFlags,
    old: vk.ImageLayout,
    new: vk.ImageLayout,
    src_stage: vk.PipelineStageFlags2,
    src_access: vk.AccessFlags2,
    dst_stage: vk.PipelineStageFlags2,
    dst_access: vk.AccessFlags2,
) void {
    const b: vk.ImageMemoryBarrier2 = .{
        .src_stage_mask = src_stage,
        .src_access_mask = src_access,
        .dst_stage_mask = dst_stage,
        .dst_access_mask = dst_access,
        .old_layout = old,
        .new_layout = new,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{ .aspect_mask = aspect, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = vk.REMAINING_ARRAY_LAYERS },
    };
    cmd.pipelineBarrier2(&.{ .image_memory_barrier_count = 1, .p_image_memory_barriers = @ptrCast(&b) });
}
