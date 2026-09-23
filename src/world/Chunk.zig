const std = @import("std");
const Block = @import("block.zig").Block;
const coords = @import("coords.zig");
const LocalPos = coords.LocalPos;

const Chunk = @This();

pub const size = coords.chunk_size;
pub const volume = size * size * size;

/// Index layout: x + 32·z + 1024·y.
blocks: [volume]Block,

pub const air: Chunk = .{ .blocks = @splat(.air) };

pub fn index(p: LocalPos) usize {
    return @as(usize, p.x) + size * @as(usize, p.z) + size * size * @as(usize, p.y);
}

pub fn get(c: *const Chunk, p: LocalPos) Block {
    return c.blocks[index(p)];
}

pub fn set(c: *Chunk, p: LocalPos, b: Block) void {
    c.blocks[index(p)] = b;
}

/// All air: nothing to mesh.
pub fn isEmpty(c: *const Chunk) bool {
    return std.mem.allEqual(Block, &c.blocks, .air);
}

// ---
// Tests
// ---

test "get set index" {
    var c: Chunk = .air;
    try std.testing.expect(c.isEmpty());
    c.set(.{ .x = 1, .y = 2, .z = 3 }, .stone);
    try std.testing.expectEqual(Block.stone, c.get(.{ .x = 1, .y = 2, .z = 3 }));
    try std.testing.expectEqual(1 + 32 * 3 + 1024 * 2, index(.{ .x = 1, .y = 2, .z = 3 }));
    try std.testing.expect(!c.isEmpty());
}
