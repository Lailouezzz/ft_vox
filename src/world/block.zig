const std = @import("std");

pub const Block = enum(u8) {
    air,
    grass,
    dirt,
    stone,
    sand,
    water,
    log,
    leaves,

    /// Solid = neither air nor water. Solid blocks hide the faces of their neighbors.
    pub fn isSolid(b: Block) bool {
        return b != .air and b != .water;
    }

    /// Whether the face of `self` touching `neighbor` must be drawn.
    /// Solid faces show against air and water; water faces only against air.
    pub fn faceVisible(self: Block, neighbor: Block) bool {
        return switch (self) {
            .air => false,
            .water => neighbor == .air,
            else => !neighbor.isSolid(),
        };
    }

    /// Linear RGB albedo, uploaded to the GPU as the block palette.
    pub fn color(b: Block) [3]f32 {
        return switch (b) {
            .air => .{ 0, 0, 0 },
            .grass => .{ 0.30, 0.60, 0.20 },
            .dirt => .{ 0.45, 0.30, 0.18 },
            .stone => .{ 0.50, 0.50, 0.52 },
            .sand => .{ 0.86, 0.80, 0.55 },
            .water => .{ 0.15, 0.35, 0.70 },
            .log => .{ 0.40, 0.27, 0.13 },
            .leaves => .{ 0.18, 0.45, 0.15 },
        };
    }
};

pub const count = @typeInfo(Block).@"enum".fields.len;

// ---
// Tests
// ---

test "face visibility" {
    try std.testing.expect(Block.stone.faceVisible(.air));
    try std.testing.expect(Block.stone.faceVisible(.water));
    try std.testing.expect(!Block.stone.faceVisible(.dirt));
    try std.testing.expect(Block.water.faceVisible(.air));
    try std.testing.expect(!Block.water.faceVisible(.water));
    try std.testing.expect(!Block.water.faceVisible(.stone));
    try std.testing.expect(!Block.air.faceVisible(.air));
}
