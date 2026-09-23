//! Frame loop: frames in flight, depth buffer, passes.
const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");
const Swapchain = @import("Swapchain.zig");
const pipeline = @import("pipeline.zig");
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const ChunkBuffers = @import("ChunkBuffers.zig");
const gpu = @import("gpu.zig");
const world = @import("world");

const Renderer = @This();

pub const frames_in_flight = 2;
const depth_format: vk.Format = .d32_sfloat;

const Frame = struct {
    pool: vk.CommandPool,
    cmd: vk.CommandBuffer,
    image_acquired: vk.Semaphore,
    fence: vk.Fence,
    /// Written by the CPU while the frame is recorded, read by the GPU.
    frame_data: Buffer,
    staging: Buffer,
};

pub const FrameInput = struct {
    view_proj: zm.Mat,
    camera_pos: [3]f32,
    sun_dir: [3]f32,
    sun_color: [3]f32,
    ambient: [3]f32,
    fog_start: f32,
    fog_end: f32,
};

gpa: Allocator,
ctx: *const Context,
swapchain: Swapchain,
depth: Image,
frames: [frames_in_flight]Frame,
frame_index: usize = 0,
sky_layout: vk.PipelineLayout,
sky_pipeline: vk.Pipeline,
chunks: ChunkBuffers,
draws: Buffer,
draw_count: Buffer,
chunk_layout: vk.PipelineLayout,
cull_pipeline: vk.Pipeline,
chunk_pipeline: vk.Pipeline,

pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D) !Renderer {
    var swapchain: Swapchain = try .init(ctx, gpa, extent, .null_handle);
    errdefer swapchain.deinit(ctx, gpa);
    var depth: Image = try .initDepth(ctx, swapchain.extent, depth_format);
    errdefer depth.deinit(ctx);

    var frames: [frames_in_flight]Frame = undefined;
    for (&frames) |*f| {
        f.pool = try ctx.device.createCommandPool(&.{ .flags = .{ .reset_command_buffer_bit = true }, .queue_family_index = ctx.queue_family }, null);
        try ctx.device.allocateCommandBuffers(&.{ .command_pool = f.pool, .level = .primary, .command_buffer_count = 1 }, @ptrCast(&f.cmd));
        f.image_acquired = try ctx.device.createSemaphore(&.{}, null);
        f.fence = try ctx.device.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
        f.frame_data = try .init(ctx, @sizeOf(gpu.FrameData), .{ .storage_buffer_bit = true }, true);
        f.staging = try .init(ctx, ChunkBuffers.staging_size, .{ .transfer_src_bit = true }, true);
    }

    var chunks: ChunkBuffers = try .init(ctx, gpa);
    errdefer chunks.deinit(ctx);
    const max_draws = ChunkBuffers.max_chunks * 6;
    var draws: Buffer = try .init(ctx, max_draws * @sizeOf(gpu.DrawCmd), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true }, false);
    errdefer draws.deinit(ctx);
    var draw_count: Buffer = try .init(ctx, 4, .{ .storage_buffer_bit = true, .indirect_buffer_bit = true, .transfer_dst_bit = true }, false);
    errdefer draw_count.deinit(ctx);
    const chunk_layout = try pipeline.createLayout(ctx, @sizeOf(gpu.Push), .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true }, &.{});
    const cull_pipeline = try pipeline.createCompute(ctx, chunk_layout, pipeline.spirv("cull.comp"));

    const sky_layout = try pipeline.createLayout(ctx, @sizeOf(SkyPush), .{ .fragment_bit = true }, &.{});
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
    const chunk_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = chunk_layout,
        .vertex = pipeline.spirv("chunk.vert"),
        .fragment = pipeline.spirv("chunk.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
    });
    return .{
        .gpa = gpa,
        .ctx = ctx,
        .swapchain = swapchain,
        .depth = depth,
        .frames = frames,
        .sky_layout = sky_layout,
        .sky_pipeline = sky_pipeline,
        .chunks = chunks,
        .draws = draws,
        .draw_count = draw_count,
        .chunk_layout = chunk_layout,
        .cull_pipeline = cull_pipeline,
        .chunk_pipeline = chunk_pipeline,
    };
}

pub fn deinit(self: *Renderer) void {
    const d = self.ctx.device;
    d.deviceWaitIdle() catch {};
    d.destroyPipeline(self.chunk_pipeline, null);
    d.destroyPipeline(self.cull_pipeline, null);
    d.destroyPipelineLayout(self.chunk_layout, null);
    self.draw_count.deinit(self.ctx);
    self.draws.deinit(self.ctx);
    self.chunks.deinit(self.ctx);
    d.destroyPipeline(self.sky_pipeline, null);
    d.destroyPipelineLayout(self.sky_layout, null);
    for (&self.frames) |*f| {
        f.staging.deinit(self.ctx);
        f.frame_data.deinit(self.ctx);
        d.destroyFence(f.fence, null);
        d.destroySemaphore(f.image_acquired, null);
        d.destroyCommandPool(f.pool, null);
    }
    self.depth.deinit(self.ctx);
    self.swapchain.deinit(self.ctx, self.gpa);
}

const SkyPush = extern struct {
    inv_view_proj: zm.Mat,
    sun_dir: [4]f32,
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
    try d.resetFences(&.{frame.fence});
    const image_index = acquired.image_index;

    const cmd: vk.CommandBufferProxy = .init(frame.cmd, self.ctx.vkd);
    try cmd.resetCommandBuffer(.{});
    try cmd.beginCommandBuffer(&.{ .flags = .{ .one_time_submit_bit = true } });

    // Chunk uploads, then GPU culling into the indirect draw buffer.
    self.writeFrameData(frame, in);
    const push: gpu.Push = .{
        .frame = frame.frame_data.address,
        .metas = self.chunks.metas.address,
        .quads = self.chunks.quads.address,
        .draws = self.draws.address,
        .count = self.draw_count.address,
    };
    try self.chunks.record(cmd, &frame.staging);
    cmd.fillBuffer(self.draw_count.handle, 0, 4, 0);
    ChunkBuffers.barrier(cmd, .{ .copy_bit = true, .clear_bit = true }, .{ .transfer_write_bit = true }, .{ .compute_shader_bit = true, .vertex_shader_bit = true }, .{ .shader_storage_read_bit = true, .shader_storage_write_bit = true });
    cmd.bindPipeline(.compute, self.cull_pipeline);
    cmd.pushConstants(self.chunk_layout, .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(gpu.Push), &push);
    cmd.dispatch(std.math.divCeil(u32, self.chunks.slot_high, 64) catch unreachable, 1, 1);
    ChunkBuffers.barrier(cmd, .{ .compute_shader_bit = true }, .{ .shader_storage_write_bit = true }, .{ .draw_indirect_bit = true }, .{ .indirect_command_read_bit = true });

    const image = self.swapchain.images[image_index];
    imageBarrier(cmd, image, .{ .color_bit = true }, .undefined, .color_attachment_optimal, .{ .color_attachment_output_bit = true }, .{}, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true });
    imageBarrier(cmd, self.depth.image, .{ .depth_bit = true }, .undefined, .depth_attachment_optimal, .{ .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .early_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true });

    const ext = self.swapchain.extent;
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
        .image_layout = .depth_attachment_optimal,
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

    const sky: SkyPush = .{ .inv_view_proj = zm.inverse(in.view_proj), .sun_dir = .{ in.sun_dir[0], in.sun_dir[1], in.sun_dir[2], 0 } };
    cmd.bindPipeline(.graphics, self.sky_pipeline);
    cmd.pushConstants(self.sky_layout, .{ .fragment_bit = true }, 0, @sizeOf(SkyPush), &sky);
    cmd.draw(3, 1, 0, 0);

    cmd.bindPipeline(.graphics, self.chunk_pipeline);
    cmd.pushConstants(self.chunk_layout, .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(gpu.Push), &push);
    cmd.drawIndirectCount(self.draws.handle, 0, self.draw_count.handle, 0, ChunkBuffers.max_chunks * 6, @sizeOf(gpu.DrawCmd));

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
    if (present == .suboptimal_khr or extent.width != self.swapchain.extent.width or extent.height != self.swapchain.extent.height)
        try self.recreate(extent);
}

fn writeFrameData(self: *Renderer, frame: *Frame, in: FrameInput) void {
    var palette: [8][4]f32 = undefined;
    for (&palette, 0..) |*c, i| {
        const rgb = @as(world.Block, @enumFromInt(i)).color();
        c.* = .{ rgb[0], rgb[1], rgb[2], 1 };
    }
    const data: *gpu.FrameData = @ptrCast(@alignCast(frame.frame_data.mapped.?));
    data.* = .{
        .view_proj = in.view_proj,
        .planes = gpu.frustumPlanes(in.view_proj),
        .camera_pos = .{ in.camera_pos[0], in.camera_pos[1], in.camera_pos[2], 1 },
        .sun_dir = .{ in.sun_dir[0], in.sun_dir[1], in.sun_dir[2], 0 },
        .sun_color = .{ in.sun_color[0], in.sun_color[1], in.sun_color[2], 0 },
        .ambient = .{ in.ambient[0], in.ambient[1], in.ambient[2], 0 },
        .fog = .{ in.fog_start, in.fog_end, 0, 0 },
        .palette = palette,
        .chunk_capacity = self.chunks.slot_high,
    };
}

fn recreate(self: *Renderer, extent: vk.Extent2D) !void {
    if (extent.width == 0 or extent.height == 0) return; // minimized
    try self.ctx.device.deviceWaitIdle();
    self.swapchain.deinitKeepHandle(self.ctx, self.gpa);
    self.swapchain = try .init(self.ctx, self.gpa, extent, self.swapchain.handle);
    self.depth.deinit(self.ctx);
    self.depth = try .initDepth(self.ctx, self.swapchain.extent, depth_format);
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
