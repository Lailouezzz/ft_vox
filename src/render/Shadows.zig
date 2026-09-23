//! Cascaded shadow maps: a depth array (one layer per cascade), per-layer
//! views for rendering, an array view + comparison sampler for lighting, and
//! stable cascade fitting (bounding spheres, texel-snapped).
const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const Context = @import("Context.zig");
const Image = @import("Image.zig");

const Shadows = @This();

pub const cascades = 3;
pub const format: vk.Format = .d32_sfloat;
/// Far distance (blocks from the camera) covered by each cascade.
pub const splits = [cascades]f32{ 20, 64, 180 };
/// Extra depth range towards the sun so casters outside a cascade still cast.
const caster_margin = 256;

resolution: u32,
image: Image,
layer_views: [cascades]vk.ImageView,
sampler: vk.Sampler,

pub fn init(ctx: *const Context, resolution: u32) !Shadows {
    var image: Image = try .init(ctx, .{ .width = resolution, .height = resolution }, format, .{ .depth_stencil_attachment_bit = true, .sampled_bit = true }, .{ .depth_bit = true }, cascades);
    errdefer image.deinit(ctx);
    var layer_views: [cascades]vk.ImageView = undefined;
    for (&layer_views, 0..) |*v, i| {
        v.* = try ctx.device.createImageView(&.{
            .image = image.image,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{ .aspect_mask = .{ .depth_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = @intCast(i), .layer_count = 1 },
        }, null);
    }
    // Reverse-Z: a fragment is lit when its depth >= the stored occluder depth.
    // Outside the map the border (depth 0) leaves everything lit.
    const sampler = try ctx.device.createSampler(&.{
        .mag_filter = .linear,
        .min_filter = .linear,
        .mipmap_mode = .nearest,
        .address_mode_u = .clamp_to_border,
        .address_mode_v = .clamp_to_border,
        .address_mode_w = .clamp_to_border,
        .mip_lod_bias = 0,
        .anisotropy_enable = .false,
        .max_anisotropy = 1,
        .compare_enable = .true,
        .compare_op = .greater_or_equal,
        .min_lod = 0,
        .max_lod = 0,
        .border_color = .float_opaque_black,
        .unnormalized_coordinates = .false,
    }, null);
    return .{ .resolution = resolution, .image = image, .layer_views = layer_views, .sampler = sampler };
}

pub fn deinit(self: *Shadows, ctx: *const Context) void {
    ctx.device.destroySampler(self.sampler, null);
    for (self.layer_views) |v| ctx.device.destroyImageView(v, null);
    self.image.deinit(ctx);
    self.* = undefined;
}

pub const Cascade = struct {
    /// Light view-projection (row-vector zmath convention, reverse-Z orthographic, Vulkan clip Y down).
    view_proj: zm.Mat,
    /// World size of one shadow texel.
    texel: f32,
};

pub fn fitCascades(resolution: u32, camera_pos: [3]f32, camera_forward: [3]f32, fov_y: f32, aspect: f32, near: f32, sun_dir: [3]f32) [cascades]Cascade {
    const up = if (@abs(sun_dir[1]) > 0.99) zm.f32x4(0, 0, 1, 0) else zm.f32x4(0, 1, 0, 0);
    // Rotation only: the light "camera" sits at the origin looking down -sun_dir.
    const light_view = zm.lookToRh(zm.f32x4(0, 0, 0, 1), zm.f32x4(-sun_dir[0], -sun_dir[1], -sun_dir[2], 0), up);
    const tan_y = @tan(fov_y / 2);
    const tan_x = tan_y * aspect;
    var out: [cascades]Cascade = undefined;
    var d0: f32 = near;
    for (splits, 0..) |d1, i| {
        // Bounding sphere of the frustum slice [d0, d1]: its center lies on the
        // view axis; the radius depends only on the slice, so it is stable
        // under camera rotation (no shimmering).
        const k = tan_x * tan_x + tan_y * tan_y;
        const mid = @min(d1, 0.5 * (d0 + d1) * (1 + k));
        const r_far = @sqrt((d1 - mid) * (d1 - mid) + d1 * d1 * k);
        const r_near = @sqrt((mid - d0) * (mid - d0) + d0 * d0 * k);
        const radius = @ceil(@max(r_far, r_near));
        const center_ws = zm.f32x4(camera_pos[0] + camera_forward[0] * mid, camera_pos[1] + camera_forward[1] * mid, camera_pos[2] + camera_forward[2] * mid, 1);
        var c = zm.mul(center_ws, light_view);
        // Snap the center to the shadow texel grid.
        const texel = 2 * radius / @as(f32, @floatFromInt(resolution));
        c[0] = @floor(c[0] / texel) * texel;
        c[1] = @floor(c[1] / texel) * texel;
        const n = -c[2] - radius - caster_margin; // distances along -Z
        const f = -c[2] + radius;
        const s = 1 / radius;
        const proj: zm.Mat = .{
            zm.f32x4(s, 0, 0, 0),
            zm.f32x4(0, -s, 0, 0),
            zm.f32x4(0, 0, 1 / (f - n), 0),
            zm.f32x4(-c[0] * s, c[1] * s, f / (f - n), 1),
        };
        out[i] = .{ .view_proj = zm.mul(light_view, proj), .texel = texel };
        d0 = d1;
    }
    return out;
}

const testing = std.testing;

test "cascade covers the camera and maps it inside the depth range" {
    const cs = fitCascades(2048, .{ 10, 80, -5 }, .{ 0, 0, -1 }, std.math.degreesToRadians(70), 16.0 / 9.0, 0.1, .{ 0.3, 0.8, -0.2 });
    for (cs) |c| {
        const p = zm.mul(zm.f32x4(10, 80, -12, 1), c.view_proj); // a point just in front of the camera
        try testing.expect(@abs(p[0]) <= 1 and @abs(p[1]) <= 1);
        try testing.expect(p[2] >= 0 and p[2] <= 1);
    }
}

test "closer to the sun means larger depth (reverse-Z)" {
    const sun = [3]f32{ 0, 1, 0 };
    const cs = fitCascades(2048, .{ 0, 80, 0 }, .{ 0, 0, -1 }, 1.2, 1.5, 0.1, sun);
    const low = zm.mul(zm.f32x4(0, 70, -10, 1), cs[0].view_proj);
    const high = zm.mul(zm.f32x4(0, 90, -10, 1), cs[0].view_proj);
    try testing.expect(high[2] > low[2]);
}
