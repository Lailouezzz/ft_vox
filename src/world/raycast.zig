const std = @import("std");
const Block = @import("block.zig").Block;
const coords = @import("coords.zig");
const BlockPos = coords.BlockPos;
const Face = coords.Face;

pub const Hit = struct {
    pos: BlockPos,
    /// Face of the hit block the ray entered through.
    face: Face,
};

/// Amanatides & Woo voxel traversal. Returns the first solid block within
/// `max_dist` of `origin` along `dir` (need not be normalized, must be non-zero).
/// `lookup.blockAt(BlockPos) Block` gives the world's blocks.
pub fn raycast(origin: [3]f32, dir: [3]f32, max_dist: f32, lookup: anytype) ?Hit {
    const len = @sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    var cell: [3]i32 = undefined;
    var step: [3]i32 = undefined;
    var t_max: [3]f32 = undefined;
    var t_delta: [3]f32 = undefined;
    for (0..3) |i| {
        const d = dir[i] / len;
        cell[i] = @intFromFloat(@floor(origin[i]));
        if (d > 0) {
            step[i] = 1;
            t_delta[i] = 1 / d;
            t_max[i] = (@as(f32, @floatFromInt(cell[i] + 1)) - origin[i]) / d;
        } else if (d < 0) {
            step[i] = -1;
            t_delta[i] = -1 / d;
            t_max[i] = (origin[i] - @as(f32, @floatFromInt(cell[i]))) / -d;
        } else {
            step[i] = 0;
            t_delta[i] = std.math.inf(f32);
            t_max[i] = std.math.inf(f32);
        }
    }

    var face: Face = undefined;
    var t: f32 = 0;
    // The starting cell is skipped: the camera sits inside air.
    while (true) {
        const axis: usize = if (t_max[0] < t_max[1])
            (if (t_max[0] < t_max[2]) 0 else 2)
        else
            (if (t_max[1] < t_max[2]) 1 else 2);
        t = t_max[axis];
        if (t > max_dist) return null;
        cell[axis] += step[axis];
        t_max[axis] += t_delta[axis];
        face = switch (axis) {
            0 => if (step[0] > 0) .neg_x else .pos_x,
            1 => if (step[1] > 0) .neg_y else .pos_y,
            else => if (step[2] > 0) .neg_z else .pos_z,
        };
        const pos: BlockPos = .{ .x = cell[0], .y = cell[1], .z = cell[2] };
        if (lookup.blockAt(pos).isSolid()) return .{ .pos = pos, .face = face };
    }
}

// ---
// Tests
// ---

const testing = std.testing;

/// A world containing a single stone block.
const OneBlock = struct {
    at: BlockPos,
    pub fn blockAt(w: OneBlock, p: BlockPos) Block {
        return if (std.meta.eql(p, w.at)) .stone else .air;
    }
};

test "hits along each axis with the right face" {
    const cases = [_]struct { dir: [3]f32, at: BlockPos, face: Face }{
        .{ .dir = .{ 1, 0, 0 }, .at = .{ .x = 3, .y = 0, .z = 0 }, .face = .neg_x },
        .{ .dir = .{ -1, 0, 0 }, .at = .{ .x = -3, .y = 0, .z = 0 }, .face = .pos_x },
        .{ .dir = .{ 0, 1, 0 }, .at = .{ .x = 0, .y = 3, .z = 0 }, .face = .neg_y },
        .{ .dir = .{ 0, -1, 0 }, .at = .{ .x = 0, .y = -3, .z = 0 }, .face = .pos_y },
        .{ .dir = .{ 0, 0, 1 }, .at = .{ .x = 0, .y = 0, .z = 3 }, .face = .neg_z },
        .{ .dir = .{ 0, 0, -1 }, .at = .{ .x = 0, .y = 0, .z = -3 }, .face = .pos_z },
    };
    for (cases) |c| {
        const hit = raycast(.{ 0.5, 0.5, 0.5 }, c.dir, 8, OneBlock{ .at = c.at }) orelse return error.NoHit;
        try testing.expectEqual(c.at, hit.pos);
        try testing.expectEqual(c.face, hit.face);
    }
}

test "diagonal hit in negative coordinates" {
    const hit = raycast(.{ -0.5, 0.5, -0.5 }, .{ -1, 0, -1.01 }, 8, OneBlock{ .at = .{ .x = -3, .y = 0, .z = -3 } }) orelse return error.NoHit;
    try testing.expectEqual(BlockPos{ .x = -3, .y = 0, .z = -3 }, hit.pos);
}

test "respects max distance" {
    const w: OneBlock = .{ .at = .{ .x = 9, .y = 0, .z = 0 } };
    try testing.expectEqual(null, raycast(.{ 0.5, 0.5, 0.5 }, .{ 1, 0, 0 }, 8, w));
    try testing.expect(raycast(.{ 0.5, 0.5, 0.5 }, .{ 1, 0, 0 }, 9, w) != null);
}

test "water is not a hit" {
    const Water = struct {
        pub fn blockAt(_: @This(), p: BlockPos) Block {
            return if (p.x == 2) .water else if (p.x == 4) .stone else .air;
        }
    };
    const hit = raycast(.{ 0.5, 0.5, 0.5 }, .{ 1, 0, 0 }, 8, Water{}) orelse return error.NoHit;
    try testing.expectEqual(@as(i32, 4), hit.pos.x);
}
