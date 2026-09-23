const std = @import("std");

pub const chunk_bits = 5;
pub const chunk_size = 1 << chunk_bits; // 32
/// Vertical world extent in chunks: cy ∈ [0, height_chunks).
pub const height_chunks = 8;

pub const ChunkPos = struct {
    x: i32,
    y: i32,
    z: i32,

    pub fn inWorld(p: ChunkPos) bool {
        return p.y >= 0 and p.y < height_chunks;
    }

    /// World coordinate of the chunk's minimum corner.
    pub fn origin(p: ChunkPos) BlockPos {
        return .{ .x = p.x << chunk_bits, .y = p.y << chunk_bits, .z = p.z << chunk_bits };
    }
};

pub const LocalPos = struct {
    x: u5,
    y: u5,
    z: u5,
};

pub const BlockPos = struct {
    x: i32,
    y: i32,
    z: i32,

    /// Arithmetic shift floors, so -1 lands in chunk -1.
    pub fn chunk(p: BlockPos) ChunkPos {
        return .{ .x = p.x >> chunk_bits, .y = p.y >> chunk_bits, .z = p.z >> chunk_bits };
    }

    pub fn local(p: BlockPos) LocalPos {
        return .{ .x = @intCast(p.x & (chunk_size - 1)), .y = @intCast(p.y & (chunk_size - 1)), .z = @intCast(p.z & (chunk_size - 1)) };
    }
};

pub const Face = enum(u3) {
    pos_x,
    neg_x,
    pos_y,
    neg_y,
    pos_z,
    neg_z,

    pub fn normal(f: Face) [3]i32 {
        return switch (f) {
            .pos_x => .{ 1, 0, 0 },
            .neg_x => .{ -1, 0, 0 },
            .pos_y => .{ 0, 1, 0 },
            .neg_y => .{ 0, -1, 0 },
            .pos_z => .{ 0, 0, 1 },
            .neg_z => .{ 0, 0, -1 },
        };
    }
};

// ---
// Tests
// ---

test "block to chunk coordinates" {
    const cases = [_]struct { w: i32, c: i32, l: u5 }{
        .{ .w = 0, .c = 0, .l = 0 },
        .{ .w = 31, .c = 0, .l = 31 },
        .{ .w = 32, .c = 1, .l = 0 },
        .{ .w = -1, .c = -1, .l = 31 },
        .{ .w = -32, .c = -1, .l = 0 },
        .{ .w = -33, .c = -2, .l = 31 },
    };
    for (cases) |c| {
        const p: BlockPos = .{ .x = c.w, .y = c.w, .z = c.w };
        try std.testing.expectEqual(ChunkPos{ .x = c.c, .y = c.c, .z = c.c }, p.chunk());
        try std.testing.expectEqual(LocalPos{ .x = c.l, .y = c.l, .z = c.l }, p.local());
    }
}

test "chunk origin round-trips" {
    const c: ChunkPos = .{ .x = -3, .y = 2, .z = 7 };
    try std.testing.expectEqual(c, c.origin().chunk());
    try std.testing.expectEqual(LocalPos{ .x = 0, .y = 0, .z = 0 }, c.origin().local());
}
