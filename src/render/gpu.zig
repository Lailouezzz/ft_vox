//! CPU mirrors of the GPU structures declared in shaders/gpu.glsl (scalar layout).
const zm = @import("zmath");
const world = @import("world");
// No import cycle: Shadows.zig only reaches Context.zig and Image.zig, neither
// of which imports gpu.zig.
const Shadows = @import("Shadows.zig");

pub const FrameData = extern struct {
    view_proj: zm.Mat,
    planes: [6][4]f32,
    camera_pos: [4]f32,
    sun_dir: [4]f32,
    sun_color: [4]f32,
    ambient: [4]f32,
    /// x: fog start, y: fog end (blocks), z: camera near plane, w: 1 when the camera is under water.
    fog: [4]f32,
    palette: [8][4]f32,
    cascade_vp: [3]zm.Mat,
    cascade_planes: [18][4]f32,
    cascade_splits: [4]f32,
    cascade_texel: [4]f32,
    shadow: [4]f32,
    chunk_capacity: u32,
    max_draws: u32,
};

pub const ChunkMeta = extern struct {
    origin: [3]i32,
    first_quad: u32,
    /// Quads per group: 6 opaque faces, then 6 water faces (`world.mesher.groups`).
    counts: [world.mesher.groups]u32,
    enabled: u32,
};

pub const DrawCmd = extern struct {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

/// Push constants of every chunk pipeline: device addresses only.
pub const Push = extern struct {
    frame: u64,
    metas: u64,
    quads: u64,
    draws: u64,
    count: u64,
    /// Culling/drawing view: 0 camera (opaque), 1..3 shadow cascades, 4 camera (water).
    view: u32,
};

// Comptime layout checks: fail the build if these CPU mirrors drift from
// src/render/shaders/gpu.glsl.
comptime {
    if (@sizeOf(ChunkMeta) != 68) @compileError("ChunkMeta size drifted from gpu.glsl's ChunkMeta");
    if (@offsetOf(Push, "view") != 40) @compileError("Push.view offset drifted from gpu.glsl's Push");
    if (@offsetOf(FrameData, "chunk_capacity") != 896) @compileError("FrameData.chunk_capacity offset drifted from gpu.glsl's FrameData");
    if (@offsetOf(FrameData, "palette") != 240) @compileError("FrameData.palette offset drifted from gpu.glsl's FrameData");
    const palette_len = @typeInfo(@FieldType(FrameData, "palette")).array.len;
    const block_count = @typeInfo(world.Block).@"enum".fields.len;
    if (palette_len != block_count) @compileError("FrameData.palette length must equal world.Block's field count");
    const cascade_vp_len = @typeInfo(@FieldType(FrameData, "cascade_vp")).array.len;
    if (cascade_vp_len != Shadows.cascades) @compileError("FrameData.cascade_vp length must equal Shadows.cascades");
    const cascade_planes_len = @typeInfo(@FieldType(FrameData, "cascade_planes")).array.len;
    if (cascade_planes_len != Shadows.cascades * 6) @compileError("FrameData.cascade_planes length must equal Shadows.cascades * 6 planes per cascade");
    if (world.mesher.groups > 16) @compileError("firstInstance packs slot * 16 + group (pull.glsl >> 4, cull.comp)");
}

/// Frustum planes (normal pointing inside, xyz·p + w >= 0 inside) of a
/// row-vector view-projection matrix with reverse-Z depth in [0, 1].
pub fn frustumPlanes(vp: zm.Mat) [6][4]f32 {
    // Columns of the row-vector matrix are the rows of the column-vector one.
    const m = zm.transpose(vp);
    const raw = [6]zm.F32x4{
        m[3] + m[0], // left
        m[3] - m[0], // right
        m[3] + m[1], // top/bottom (Y flipped, both kept)
        m[3] - m[1],
        m[3] - m[2], // near (reverse-Z: depth <= w)
        m[2], // far at infinity: depth >= 0
    };
    var out: [6][4]f32 = undefined;
    for (raw, &out) |p, *o| {
        const len = @sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        o.* = if (len > 0) .{ p[0] / len, p[1] / len, p[2] / len, p[3] / len } else .{ 0, 0, 0, 1 };
    }
    return out;
}

const std = @import("std");
const Camera = @import("../Camera.zig");

test "frustum planes keep what is in front and reject what is behind" {
    const c: Camera = .{ .pos = .{ 0, 0, 0 }, .yaw = 0, .pitch = 0 };
    const planes = frustumPlanes(c.viewProj(16.0 / 9.0));
    const inside = struct {
        fn f(ps: [6][4]f32, p: [3]f32) bool {
            for (ps) |pl| if (pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3] < 0) return false;
            return true;
        }
    }.f;
    try std.testing.expect(inside(planes, .{ 0, 0, -10 })); // straight ahead
    try std.testing.expect(inside(planes, .{ 0, 0, -1e5 })); // far away: infinite far plane
    try std.testing.expect(!inside(planes, .{ 0, 0, 10 })); // behind
    try std.testing.expect(!inside(planes, .{ 100, 0, -10 })); // far to the right
    try std.testing.expect(!inside(planes, .{ 0, 100, -10 })); // far above
}
