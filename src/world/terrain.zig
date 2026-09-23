const std = @import("std");
const znoise = @import("znoise");
const Block = @import("block.zig").Block;
const Chunk = @import("Chunk.zig");
const coords = @import("coords.zig");
const ChunkPos = coords.ChunkPos;
const trees = @import("trees.zig");

pub const sea_level = 62;
/// Surfaces at or below this height become sand (beaches, sea floor).
pub const beach_level = sea_level + 2;
/// Columns whose surface is at or below this height are never carved:
/// no fluid simulation, so a cave there would leave hanging water walls.
pub const cave_min_surface = sea_level + 1;
const dirt_depth = 3;
const cave_threshold = 0.1;
/// Cave noise is sampled every `cave_step` blocks and trilinearly interpolated.
const cave_step = 4;
const cave_grid = Chunk.size / cave_step + 1;

/// Noise state derived from the seed. Cheap to build, immutable, shared by all queries.
pub const Generator = struct {
    seed: u64,
    height: znoise.FnlGenerator,
    cave_a: znoise.FnlGenerator,
    cave_b: znoise.FnlGenerator,

    pub fn init(seed: u64) Generator {
        const s: i32 = @truncate(@as(i64, @bitCast(seed ^ (seed >> 32))));
        return .{
            .seed = seed,
            .height = .{ .seed = s, .frequency = 0.004, .fractal_type = .fbm, .octaves = 5 },
            .cave_a = .{ .seed = s +% 1, .frequency = 0.02 },
            .cave_b = .{ .seed = s +% 2, .frequency = 0.02 },
        };
    }

    /// Surface height of column (wx, wz), in [40, 140].
    pub fn heightAt(g: *const Generator, wx: i32, wz: i32) i32 {
        const n = g.height.noise2(@floatFromInt(wx), @floatFromInt(wz));
        return 90 + @as(i32, @intFromFloat(@round(std.math.clamp(n, -1, 1) * 50)));
    }

    fn caveNoise(g: *const Generator, x: f32, y: f32, z: f32) [2]f32 {
        return .{ g.cave_a.noise3(x, y, z), g.cave_b.noise3(x, y, z) };
    }

    /// Spaghetti caves: tunnels where two 3D noises are both near zero.
    /// Exact version, used by tree placement; `generate` uses an interpolated grid.
    pub fn isCave(g: *const Generator, wx: i32, wy: i32, wz: i32) bool {
        if (wy < 1 or g.heightAt(wx, wz) <= cave_min_surface) return false;
        const n = g.caveNoise(@floatFromInt(wx), @floatFromInt(wy), @floatFromInt(wz));
        return @abs(n[0]) < cave_threshold and @abs(n[1]) < cave_threshold;
    }
};

/// Fills `out` with the chunk at `pos`. Pure: depends only on `g.seed` and `pos`.
pub fn generate(g: *const Generator, pos: ChunkPos, out: *Chunk) void {
    out.* = .air;
    if (!pos.inWorld()) return;
    const origin = pos.origin();

    var heights: [Chunk.size][Chunk.size]i32 = undefined;
    var max_height: i32 = sea_level;
    for (0..Chunk.size) |z| for (0..Chunk.size) |x| {
        const h = g.heightAt(origin.x + @as(i32, @intCast(x)), origin.z + @as(i32, @intCast(z)));
        heights[z][x] = h;
        max_height = @max(max_height, h);
    };

    if (origin.y <= max_height) {
        fillColumns(origin, &heights, out);
        carveCaves(g, origin, &heights, out);
    }
    trees.place(g, pos, out);
}

fn fillColumns(origin: coords.BlockPos, heights: *const [Chunk.size][Chunk.size]i32, out: *Chunk) void {
    for (0..Chunk.size) |z| for (0..Chunk.size) |x| {
        const h = heights[z][x];
        for (0..Chunk.size) |y| {
            const wy = origin.y + @as(i32, @intCast(y));
            const b: Block = if (wy > h)
                (if (wy <= sea_level) .water else .air)
            else if (wy == h)
                (if (h <= beach_level) .sand else .grass)
            else if (wy > h - 1 - dirt_depth)
                .dirt
            else
                .stone;
            out.set(.{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) }, b);
        }
    };
}

fn carveCaves(g: *const Generator, origin: coords.BlockPos, heights: *const [Chunk.size][Chunk.size]i32, out: *Chunk) void {
    var grid: [cave_grid][cave_grid][cave_grid][2]f32 = undefined;
    for (0..cave_grid) |gy| for (0..cave_grid) |gz| for (0..cave_grid) |gx| {
        grid[gy][gz][gx] = g.caveNoise(
            @floatFromInt(origin.x + @as(i32, @intCast(gx * cave_step))),
            @floatFromInt(origin.y + @as(i32, @intCast(gy * cave_step))),
            @floatFromInt(origin.z + @as(i32, @intCast(gz * cave_step))),
        );
    };

    for (0..Chunk.size) |z| for (0..Chunk.size) |x| {
        if (heights[z][x] <= cave_min_surface) continue;
        for (0..Chunk.size) |y| {
            const wy = origin.y + @as(i32, @intCast(y));
            if (wy < 1 or wy > heights[z][x]) continue;
            const n = trilinear(&grid, x, y, z);
            if (@abs(n[0]) < cave_threshold and @abs(n[1]) < cave_threshold)
                out.set(.{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) }, .air);
        }
    };
}

fn trilinear(grid: *const [cave_grid][cave_grid][cave_grid][2]f32, x: usize, y: usize, z: usize) [2]f32 {
    const x0 = x / cave_step;
    const y0 = y / cave_step;
    const z0 = z / cave_step;
    const fx: f32 = @as(f32, @floatFromInt(x % cave_step)) / cave_step;
    const fy: f32 = @as(f32, @floatFromInt(y % cave_step)) / cave_step;
    const fz: f32 = @as(f32, @floatFromInt(z % cave_step)) / cave_step;
    var r: [2]f32 = undefined;
    for (0..2) |i| {
        const c00 = std.math.lerp(grid[y0][z0][x0][i], grid[y0][z0][x0 + 1][i], fx);
        const c01 = std.math.lerp(grid[y0][z0 + 1][x0][i], grid[y0][z0 + 1][x0 + 1][i], fx);
        const c10 = std.math.lerp(grid[y0 + 1][z0][x0][i], grid[y0 + 1][z0][x0 + 1][i], fx);
        const c11 = std.math.lerp(grid[y0 + 1][z0 + 1][x0][i], grid[y0 + 1][z0 + 1][x0 + 1][i], fx);
        r[i] = std.math.lerp(std.math.lerp(c00, c01, fz), std.math.lerp(c10, c11, fz), fy);
    }
    return r;
}

// ---
// Tests
// ---

const testing = std.testing;

fn worldY(pos: ChunkPos, y: usize) i32 {
    return pos.origin().y + @as(i32, @intCast(y));
}

test "generate is deterministic" {
    const g: Generator = .init(42);
    var a: Chunk = undefined;
    var b: Chunk = undefined;
    const pos: ChunkPos = .{ .x = -3, .y = 2, .z = 5 };
    generate(&g, pos, &a);
    generate(&g, pos, &b);
    try testing.expectEqualSlices(Block, &a.blocks, &b.blocks);
}

test "different seeds give different terrain" {
    const g1: Generator = .init(1);
    const g2: Generator = .init(2);
    var diff: usize = 0;
    var x: i32 = 0;
    while (x < 64) : (x += 1) {
        if (g1.heightAt(x, 0) != g2.heightAt(x, 0)) diff += 1;
    }
    try testing.expect(diff > 0);
}

test "height stays in range" {
    const g: Generator = .init(7);
    var x: i32 = -2000;
    while (x < 2000) : (x += 37) {
        const h = g.heightAt(x, x * 3);
        try testing.expect(h >= 40 and h <= 140);
    }
}

test "outside world is air" {
    const g: Generator = .init(42);
    var c: Chunk = undefined;
    generate(&g, .{ .x = 0, .y = -1, .z = 0 }, &c);
    try testing.expect(c.isEmpty());
    generate(&g, .{ .x = 0, .y = coords.height_chunks, .z = 0 }, &c);
    try testing.expect(c.isEmpty());
}

test "layers, water and caves follow the rules" {
    const g: Generator = .init(42);
    var c: Chunk = undefined;
    var pos: ChunkPos = .{ .x = -2, .y = 0, .z = -2 };
    while (pos.x < 2) : (pos.x += 1) {
        pos.z = -2;
        while (pos.z < 2) : (pos.z += 1) {
            pos.y = 0;
            while (pos.y < 5) : (pos.y += 1) {
                generate(&g, pos, &c);
                const o = pos.origin();
                for (0..Chunk.size) |z| for (0..Chunk.size) |x| {
                    const h = g.heightAt(o.x + @as(i32, @intCast(x)), o.z + @as(i32, @intCast(z)));
                    for (0..Chunk.size) |y| {
                        const wy = worldY(pos, y);
                        const b = c.get(.{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) });
                        // Water only at or below sea level, and only above the surface.
                        if (b == .water) try testing.expect(wy <= sea_level and wy > h);
                        // Above the surface: air, water, or tree blocks only.
                        if (wy > h) try testing.expect(b == .air or b == .water or b == .log or b == .leaves);
                        // Stone is never above the dirt layer.
                        if (b == .stone) try testing.expect(wy <= h - 1 - dirt_depth);
                        // Submerged or near-sea columns are never carved.
                        if (h <= cave_min_surface and wy <= h and wy >= 0) try testing.expect(b != .air);
                    }
                };
            }
        }
    }
}

test "surface block matches beach rule" {
    const g: Generator = .init(42);
    var c: Chunk = undefined;
    var cx: i32 = -4;
    while (cx < 4) : (cx += 1) {
        const x: i32 = cx * 32;
        const h = g.heightAt(x, 0);
        const pos = (coords.BlockPos{ .x = x, .y = h, .z = 0 }).chunk();
        generate(&g, pos, &c);
        const b = c.get((coords.BlockPos{ .x = x, .y = h, .z = 0 }).local());
        if (b == .air) continue; // carved by a cave
        if (h <= beach_level) {
            try testing.expectEqual(Block.sand, b);
        } else {
            try testing.expect(b == .grass or b == .log);
        }
    }
}
