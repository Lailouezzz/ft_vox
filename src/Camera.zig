//! Free-fly camera. Matrices use zmath's row-vector convention: their memory
//! layout is exactly what GLSL's column-major `mat4 * vec4` expects.
const std = @import("std");
const zm = @import("zmath");

const Camera = @This();

pos: [3]f32 = .{ 0, 110, 0 },
yaw: f32 = 0, // radians, 0 = looking towards -Z
pitch: f32 = -0.3,
fov_y: f32 = std.math.degreesToRadians(70),
near: f32 = 0.1,

pub fn forward(c: Camera) [3]f32 {
    return .{ -@sin(c.yaw) * @cos(c.pitch), @sin(c.pitch), -@cos(c.yaw) * @cos(c.pitch) };
}

pub fn right(c: Camera) [3]f32 {
    return .{ @cos(c.yaw), 0, -@sin(c.yaw) };
}

pub fn view(c: Camera) zm.Mat {
    const f = c.forward();
    return zm.lookToRh(zm.f32x4(c.pos[0], c.pos[1], c.pos[2], 1), zm.f32x4(f[0], f[1], f[2], 0), zm.f32x4(0, 1, 0, 0));
}

/// Infinite reverse-Z perspective for Vulkan (clip Y down, depth 1 at `near`, 0 at infinity).
/// Depth test GREATER, clear depth 0.
pub fn projection(c: Camera, aspect: f32) zm.Mat {
    const f = 1 / @tan(c.fov_y / 2);
    return .{
        zm.f32x4(f / aspect, 0, 0, 0),
        zm.f32x4(0, -f, 0, 0),
        zm.f32x4(0, 0, 0, -1),
        zm.f32x4(0, 0, c.near, 0),
    };
}

pub fn viewProj(c: Camera, aspect: f32) zm.Mat {
    return zm.mul(c.view(), c.projection(aspect));
}

const testing = std.testing;

test "reverse-Z maps near to 1 and far to ~0" {
    const c: Camera = .{ .pos = .{ 0, 0, 0 }, .yaw = 0, .pitch = 0 };
    const vp = c.viewProj(1);
    const at_near = zm.mul(zm.f32x4(0, 0, -c.near, 1), vp);
    const at_far = zm.mul(zm.f32x4(0, 0, -1e6, 1), vp);
    try testing.expectApproxEqAbs(@as(f32, 1), at_near[2] / at_near[3], 1e-5);
    try testing.expect(at_far[2] / at_far[3] < 1e-6);
}

test "up in the world is up on screen (Vulkan clip Y points down)" {
    const c: Camera = .{ .pos = .{ 0, 0, 0 }, .yaw = 0, .pitch = 0 };
    const p = zm.mul(zm.f32x4(0, 1, -5, 1), c.viewProj(1));
    try testing.expect(p[1] / p[3] < 0);
}
