const std = @import("std");
const Block = @import("block.zig").Block;
const Chunk = @import("Chunk.zig");
const coords = @import("coords.zig");
const ChunkPos = coords.ChunkPos;
const BlockPos = coords.BlockPos;
const terrain = @import("terrain.zig");
const Generator = terrain.Generator;

/// Trees per 1000 eligible columns.
const density = 8;
/// Horizontal reach of leaves from the trunk.
pub const reach = 2;

pub const Tree = struct {
    base: BlockPos, // first trunk block, just above the surface
    trunk_height: i32, // 4..6
};

fn hash(seed: u64, wx: i32, wz: i32) u64 {
    // splitmix64 finalizer over the packed column coordinates.
    var h = seed ^ (@as(u64, @as(u32, @bitCast(wx))) << 32 | @as(u32, @bitCast(wz)));
    h = (h ^ (h >> 30)) *% 0xbf58476d1ce4e5b9;
    h = (h ^ (h >> 27)) *% 0x94d049bb133111eb;
    return h ^ (h >> 31);
}

/// The tree rooted at column (wx, wz), if any. Pure.
pub fn treeAt(g: *const Generator, wx: i32, wz: i32) ?Tree {
    const h = hash(g.seed, wx, wz);
    if (h % 1000 >= density) return null;
    const surface = g.heightAt(wx, wz);
    if (surface <= terrain.beach_level) return null; // not grass
    if (g.isCave(wx, surface, wz)) return null; // surface carved away
    return .{
        .base = .{ .x = wx, .y = surface + 1, .z = wz },
        .trunk_height = 4 + @as(i32, @intCast((h >> 32) % 3)),
    };
}

/// Calls `emit(ctx, pos, block)` for every block of the tree.
pub fn forEachBlock(t: Tree, ctx: anytype, comptime emit: fn (@TypeOf(ctx), BlockPos, Block) void) void {
    const top = t.base.y + t.trunk_height - 1;
    var dy: i32 = -1;
    while (dy <= 2) : (dy += 1) {
        const r: i32 = if (dy <= 0) reach else 1;
        var dz: i32 = -r;
        while (dz <= r) : (dz += 1) {
            var dx: i32 = -r;
            while (dx <= r) : (dx += 1) {
                if (@abs(dx) == reach and @abs(dz) == reach) continue; // round the corners
                emit(ctx, .{ .x = t.base.x + dx, .y = top + dy, .z = t.base.z + dz }, .leaves);
            }
        }
    }
    var y = t.base.y;
    while (y <= top) : (y += 1) emit(ctx, .{ .x = t.base.x, .y = y, .z = t.base.z }, .log);
}

const PlaceCtx = struct { chunk: *Chunk, pos: ChunkPos };

fn placeBlock(ctx: PlaceCtx, p: BlockPos, b: Block) void {
    if (!std.meta.eql(p.chunk(), ctx.pos)) return;
    const l = p.local();
    const cur = ctx.chunk.get(l);
    // Order independent: logs override leaves, leaves only fill air.
    const write = switch (b) {
        .log => cur == .air or cur == .leaves,
        else => cur == .air,
    };
    if (write) ctx.chunk.set(l, b);
}

/// Writes into `out` every tree block falling inside chunk `pos`, including
/// trees rooted up to `reach` blocks outside it.
pub fn place(g: *const Generator, pos: ChunkPos, out: *Chunk) void {
    const o = pos.origin();
    var wz = o.z - reach;
    while (wz < o.z + Chunk.size + reach) : (wz += 1) {
        var wx = o.x - reach;
        while (wx < o.x + Chunk.size + reach) : (wx += 1) {
            const t = treeAt(g, wx, wz) orelse continue;
            // Tree spans [base.y, base.y + trunk_height + 1]; skip if outside this chunk's Y range.
            if (t.base.y + t.trunk_height + 1 < o.y or t.base.y >= o.y + Chunk.size) continue;
            forEachBlock(t, PlaceCtx{ .chunk = out, .pos = pos }, placeBlock);
        }
    }
}

// ---
// Tests
// ---

const testing = std.testing;

fn findTreeNearBorder(g: *const Generator) Tree {
    // A tree whose trunk sits on local x == 31, so leaves spill into chunk x + 1.
    var cz: i32 = 0;
    while (true) : (cz += 1) {
        var wz = cz * 32;
        while (wz < cz * 32 + 32) : (wz += 1) {
            var cx: i32 = 0;
            while (cx < 8) : (cx += 1) {
                if (treeAt(g, cx * 32 + 31, wz)) |t| return t;
            }
        }
    }
}

const CheckCtx = struct { g: *const Generator, failed: *bool };

fn checkBlock(ctx: CheckCtx, p: BlockPos, b: Block) void {
    var c: Chunk = undefined;
    terrain.generate(ctx.g, p.chunk(), &c);
    // Tree blocks only yield to terrain or another tree, never to air:
    // an air cell means the chunk on this side missed the tree.
    _ = b;
    if (c.get(p.local()) == .air) ctx.failed.* = true;
}

test "tree crossing a chunk border is whole on both sides" {
    const g: Generator = .init(42);
    const t = findTreeNearBorder(&g);
    var failed = false;
    forEachBlock(t, CheckCtx{ .g = &g, .failed = &failed }, checkBlock);
    try testing.expect(!failed);
}

test "trees stand on grass above the beach" {
    const g: Generator = .init(42);
    var n: usize = 0;
    var wx: i32 = 0;
    while (wx < 512) : (wx += 1) {
        const t = treeAt(&g, wx, 0) orelse continue;
        n += 1;
        try testing.expect(t.base.y - 1 > terrain.beach_level);
        try testing.expect(t.trunk_height >= 4 and t.trunk_height <= 6);
    }
    try testing.expect(n > 0);
}
