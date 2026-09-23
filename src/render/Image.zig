//! A device-local image with its own memory allocation and one view.
const std = @import("std");
const vk = @import("vulkan");
const Context = @import("Context.zig");

const Image = @This();

image: vk.Image,
memory: vk.DeviceMemory,
view: vk.ImageView,

pub fn init(ctx: *const Context, extent: vk.Extent2D, format: vk.Format, usage: vk.ImageUsageFlags, aspect: vk.ImageAspectFlags, layers: u32) !Image {
    const image = try ctx.device.createImage(&.{
        .image_type = .@"2d",
        .format = format,
        .extent = .{ .width = extent.width, .height = extent.height, .depth = 1 },
        .mip_levels = 1,
        .array_layers = layers,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = usage,
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, null);
    errdefer ctx.device.destroyImage(image, null);
    const req = ctx.device.getImageMemoryRequirements(image);
    const memory = try ctx.device.allocateMemory(&.{
        .allocation_size = req.size,
        .memory_type_index = try ctx.findMemoryType(req.memory_type_bits, .{ .device_local_bit = true }),
    }, null);
    errdefer ctx.device.freeMemory(memory, null);
    try ctx.device.bindImageMemory(image, memory, 0);
    const view = try ctx.device.createImageView(&.{
        .image = image,
        .view_type = if (layers > 1) .@"2d_array" else .@"2d",
        .format = format,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = .{ .aspect_mask = aspect, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = layers },
    }, null);
    return .{ .image = image, .memory = memory, .view = view };
}

pub fn initDepth(ctx: *const Context, extent: vk.Extent2D, format: vk.Format) !Image {
    return init(ctx, extent, format, .{ .depth_stencil_attachment_bit = true }, .{ .depth_bit = true }, 1);
}

pub fn deinit(self: *Image, ctx: *const Context) void {
    ctx.device.destroyImageView(self.view, null);
    ctx.device.destroyImage(self.image, null);
    ctx.device.freeMemory(self.memory, null);
    self.* = undefined;
}
