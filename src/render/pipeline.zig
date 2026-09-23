//! Pipeline helpers. SPIR-V goes straight into the stage via maintenance5
//! (VkShaderModuleCreateInfo chained on the stage): no VkShaderModule objects.
const std = @import("std");
const vk = @import("vulkan");
const Context = @import("Context.zig");

/// Embedded SPIR-V of a shader built by build.zig (see `shaders` there).
pub fn spirv(comptime name: []const u8) []const u32 {
    const bytes align(@alignOf(u32)) = @embedFile(name).*;
    return std.mem.bytesAsSlice(u32, &bytes);
}

pub const GraphicsDesc = struct {
    layout: vk.PipelineLayout,
    vertex: []const u32,
    fragment: ?[]const u32,
    color_format: ?vk.Format,
    depth_format: vk.Format,
    depth_test: bool = true,
    depth_write: bool = true,
    depth_clamp: bool = false,
    cull_back: bool = true,
    topology: vk.PrimitiveTopology = .triangle_list,
    depth_bias: bool = false,
};

pub fn createGraphics(ctx: *const Context, d: GraphicsDesc) !vk.Pipeline {
    var modules: [2]vk.ShaderModuleCreateInfo = undefined;
    var stages: [2]vk.PipelineShaderStageCreateInfo = undefined;
    var n: u32 = 0;
    for ([_]?[]const u32{ d.vertex, d.fragment }, [_]vk.ShaderStageFlags{ .{ .vertex_bit = true }, .{ .fragment_bit = true } }) |code, stage| {
        const c = code orelse continue;
        modules[n] = .{ .code_size = c.len * 4, .p_code = c.ptr };
        stages[n] = .{ .p_next = &modules[n], .stage = stage, .module = .null_handle, .p_name = "main" };
        n += 1;
    }
    const color_formats: []const vk.Format = if (d.color_format) |*f| f[0..1] else &.{};
    const rendering: vk.PipelineRenderingCreateInfo = .{
        .view_mask = 0,
        .color_attachment_count = @intCast(color_formats.len),
        .p_color_attachment_formats = color_formats.ptr,
        .depth_attachment_format = d.depth_format,
        .stencil_attachment_format = .undefined,
    };
    const dynamic = [_]vk.DynamicState{ .viewport, .scissor, .depth_bias };
    const blend = vk.PipelineColorBlendAttachmentState{
        .blend_enable = .false,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };
    var pipeline: vk.Pipeline = undefined;
    _ = try ctx.device.createGraphicsPipelines(.null_handle, &.{.{
        .p_next = &rendering,
        .stage_count = n,
        .p_stages = &stages,
        .p_vertex_input_state = &.{},
        .p_input_assembly_state = &.{ .topology = d.topology, .primitive_restart_enable = .false },
        .p_viewport_state = &.{ .viewport_count = 1, .scissor_count = 1 },
        .p_rasterization_state = &.{
            .depth_clamp_enable = if (d.depth_clamp) .true else .false,
            .rasterizer_discard_enable = .false,
            .polygon_mode = .fill,
            .cull_mode = if (d.cull_back) .{ .back_bit = true } else .{},
            .front_face = .counter_clockwise,
            .depth_bias_enable = if (d.depth_bias) .true else .false,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1,
        },
        .p_multisample_state = &.{ .rasterization_samples = .{ .@"1_bit" = true }, .sample_shading_enable = .false, .min_sample_shading = 1, .alpha_to_coverage_enable = .false, .alpha_to_one_enable = .false },
        .p_depth_stencil_state = &.{
            .depth_test_enable = if (d.depth_test) .true else .false,
            .depth_write_enable = if (d.depth_write) .true else .false,
            .depth_compare_op = .greater_or_equal, // reverse-Z
            .depth_bounds_test_enable = .false,
            .stencil_test_enable = .false,
            .front = std.mem.zeroes(vk.StencilOpState),
            .back = std.mem.zeroes(vk.StencilOpState),
            .min_depth_bounds = 0,
            .max_depth_bounds = 1,
        },
        .p_color_blend_state = &.{ .logic_op_enable = .false, .logic_op = .copy, .attachment_count = @intCast(color_formats.len), .p_attachments = @ptrCast(&blend), .blend_constants = .{ 0, 0, 0, 0 } },
        .p_dynamic_state = &.{ .dynamic_state_count = dynamic.len, .p_dynamic_states = &dynamic },
        .layout = d.layout,
        .render_pass = .null_handle,
        .subpass = 0,
        .base_pipeline_index = -1,
    }}, null, (&pipeline)[0..1]);
    return pipeline;
}

pub fn createCompute(ctx: *const Context, layout: vk.PipelineLayout, code: []const u32) !vk.Pipeline {
    const module: vk.ShaderModuleCreateInfo = .{ .code_size = code.len * 4, .p_code = code.ptr };
    var pipeline: vk.Pipeline = undefined;
    _ = try ctx.device.createComputePipelines(.null_handle, &.{.{
        .stage = .{ .p_next = &module, .stage = .{ .compute_bit = true }, .module = .null_handle, .p_name = "main" },
        .layout = layout,
        .base_pipeline_index = -1,
    }}, null, (&pipeline)[0..1]);
    return pipeline;
}

pub fn createLayout(ctx: *const Context, push_size: u32, stages: vk.ShaderStageFlags, set_layouts: []const vk.DescriptorSetLayout) !vk.PipelineLayout {
    const range: vk.PushConstantRange = .{ .stage_flags = stages, .offset = 0, .size = push_size };
    return ctx.device.createPipelineLayout(&.{
        .set_layout_count = @intCast(set_layouts.len),
        .p_set_layouts = set_layouts.ptr,
        .push_constant_range_count = if (push_size > 0) 1 else 0,
        .p_push_constant_ranges = @ptrCast(&range),
    }, null);
}
