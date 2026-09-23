//! Swapchain + image views. Recreated on resize or when presentation reports out of date.
const std = @import("std");
const vk = @import("vulkan");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");

const Swapchain = @This();

handle: vk.SwapchainKHR,
format: vk.Format,
extent: vk.Extent2D,
images: []vk.Image,
views: []vk.ImageView,
/// One per image: signalled when rendering to that image is done, waited on by present.
/// Per image rather than per frame, since present gives no signal of when it releases it.
render_done: []vk.Semaphore,

pub fn init(ctx: *const Context, gpa: Allocator, extent: vk.Extent2D, old: vk.SwapchainKHR) !Swapchain {
    const caps = try ctx.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(ctx.pdev, ctx.surface);
    const format = try pickFormat(ctx, gpa);
    const present_mode = try pickPresentMode(ctx, gpa);
    const actual: vk.Extent2D = if (caps.current_extent.width != std.math.maxInt(u32)) caps.current_extent else .{
        .width = std.math.clamp(extent.width, caps.min_image_extent.width, caps.max_image_extent.width),
        .height = std.math.clamp(extent.height, caps.min_image_extent.height, caps.max_image_extent.height),
    };
    var image_count = caps.min_image_count + 1;
    if (caps.max_image_count > 0) image_count = @min(image_count, caps.max_image_count);

    const handle = try ctx.device.createSwapchainKHR(&.{
        .surface = ctx.surface,
        .min_image_count = image_count,
        .image_format = format.format,
        .image_color_space = format.color_space,
        .image_extent = actual,
        .image_array_layers = 1,
        .image_usage = .{ .color_attachment_bit = true },
        .image_sharing_mode = .exclusive,
        .pre_transform = caps.current_transform,
        .composite_alpha = .{ .opaque_bit_khr = true },
        .present_mode = present_mode,
        .clipped = .true,
        .old_swapchain = old,
    }, null);
    errdefer ctx.device.destroySwapchainKHR(handle, null);
    if (old != .null_handle) ctx.device.destroySwapchainKHR(old, null);

    const images = try ctx.device.getSwapchainImagesAllocKHR(handle, gpa);
    errdefer gpa.free(images);
    const views = try gpa.alloc(vk.ImageView, images.len);
    errdefer gpa.free(views);
    const render_done = try gpa.alloc(vk.Semaphore, images.len);
    errdefer gpa.free(render_done);
    for (images, views, render_done) |img, *view, *sem| {
        view.* = try ctx.device.createImageView(&.{
            .image = img,
            .view_type = .@"2d",
            .format = format.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{ .aspect_mask = .{ .color_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
        }, null);
        sem.* = try ctx.device.createSemaphore(&.{}, null);
    }
    return .{ .handle = handle, .format = format.format, .extent = actual, .images = images, .views = views, .render_done = render_done };
}

/// Destroys views and semaphores; keeps `handle` alive when it is handed to `init` as `old`.
pub fn deinitKeepHandle(self: *Swapchain, ctx: *const Context, gpa: Allocator) void {
    for (self.views, self.render_done) |v, s| {
        ctx.device.destroyImageView(v, null);
        ctx.device.destroySemaphore(s, null);
    }
    gpa.free(self.views);
    gpa.free(self.render_done);
    gpa.free(self.images);
}

pub fn deinit(self: *Swapchain, ctx: *const Context, gpa: Allocator) void {
    self.deinitKeepHandle(ctx, gpa);
    ctx.device.destroySwapchainKHR(self.handle, null);
    self.* = undefined;
}

fn pickFormat(ctx: *const Context, gpa: Allocator) !vk.SurfaceFormatKHR {
    const formats = try ctx.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(ctx.pdev, ctx.surface, gpa);
    defer gpa.free(formats);
    for (formats) |f| {
        if (f.format == .b8g8r8a8_srgb and f.color_space == .srgb_nonlinear_khr) return f;
    }
    return formats[0];
}

/// Mailbox (low latency, no tearing) when available, otherwise FIFO (always supported).
fn pickPresentMode(ctx: *const Context, gpa: Allocator) !vk.PresentModeKHR {
    const modes = try ctx.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(ctx.pdev, ctx.surface, gpa);
    defer gpa.free(modes);
    for (modes) |m| if (m == .mailbox_khr) return m;
    return .fifo_khr;
}
