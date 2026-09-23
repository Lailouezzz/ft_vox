//! Binary greedy meshing. Input: the chunk plus a one-block border from its
//! six face neighbors (34³). Output: quads packed in u64, grouped by face.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Block = @import("block.zig").Block;
const block_count = @import("block.zig").count;
const Chunk = @import("Chunk.zig");
const coords = @import("coords.zig");
const Face = coords.Face;

const cs = Chunk.size; // 32
pub const padded = cs + 2; // 34

/// Index layout: x + 34·z + 34²·y, coordinates shifted by +1 (0 and 33 are the border).
pub const Volume = [padded * padded * padded]Block;

pub fn volumeIndex(x: usize, y: usize, z: usize) usize {
    return x + padded * z + padded * padded * y;
}

/// One greedy rectangle. (x, y, z) is its minimum block, in chunk-local coordinates.
/// Width/height axes per face: ±X → (z, y), ±Y → (x, z), ±Z → (x, y).
pub const Quad = packed struct(u64) {
    x: u6,
    y: u6,
    z: u6,
    w: u6, // 1..32
    h: u6, // 1..32
    face: Face,
    block: Block,
    _pad: u23 = 0,
};

pub const Mesh = struct {
    /// Quads sorted by face, in `Face` order.
    quads: []Quad,
    /// Number of quads per face; face f starts at sum(counts[0..f]).
    counts: [6]u32,

    pub fn deinit(m: *Mesh, gpa: Allocator) void {
        gpa.free(m.quads);
        m.* = undefined;
    }
};

/// Builds the padded volume of `center`. Missing neighbors (null) count as air.
/// Neighbor order follows `Face`: +X, -X, +Y, -Y, +Z, -Z.
/// Runs on the main thread for every mesh job, so rows of 32 blocks are copied
/// with memcpy wherever both layouts keep x contiguous.
pub fn buildVolume(center: *const Chunk, neighbors: [6]?*const Chunk, out: *Volume) void {
    out.* = @splat(.air);
    for (0..cs) |y| for (0..cs) |z| {
        @memcpy(out[volumeIndex(1, y + 1, z + 1)..][0..cs], center.blocks[cs * z + cs * cs * y ..][0..cs]);
    };
    // ±Y and ±Z faces: rows along x.
    for (0..cs) |a| {
        if (neighbors[@intFromEnum(Face.pos_y)]) |n| @memcpy(out[volumeIndex(1, padded - 1, a + 1)..][0..cs], n.blocks[cs * a ..][0..cs]);
        if (neighbors[@intFromEnum(Face.neg_y)]) |n| @memcpy(out[volumeIndex(1, 0, a + 1)..][0..cs], n.blocks[cs * a + cs * cs * (cs - 1) ..][0..cs]);
        if (neighbors[@intFromEnum(Face.pos_z)]) |n| @memcpy(out[volumeIndex(1, a + 1, padded - 1)..][0..cs], n.blocks[cs * cs * a ..][0..cs]);
        if (neighbors[@intFromEnum(Face.neg_z)]) |n| @memcpy(out[volumeIndex(1, a + 1, 0)..][0..cs], n.blocks[cs * (cs - 1) + cs * cs * a ..][0..cs]);
    }
    // ±X faces: one block per row.
    for (0..cs) |y| for (0..cs) |z| {
        const i = cs * z + cs * cs * y;
        if (neighbors[@intFromEnum(Face.pos_x)]) |n| out[volumeIndex(padded - 1, y + 1, z + 1)] = n.blocks[i];
        if (neighbors[@intFromEnum(Face.neg_x)]) |n| out[volumeIndex(0, y + 1, z + 1)] = n.blocks[i + cs - 1];
    };
}

/// Maps (axis-aligned depth d, width coord u, height coord v) of `face`'s axis to (x, y, z).
fn toXyz(face: Face, d: usize, u: usize, v: usize) [3]usize {
    return switch (face) {
        .pos_x, .neg_x => .{ d, v, u },
        .pos_y, .neg_y => .{ u, d, v },
        .pos_z, .neg_z => .{ u, v, d },
    };
}

/// Scratch memory: per block type, per axis, a 34×34 grid of 34-bit columns.
const Scratch = struct {
    /// cols[type][axis][v * 34 + u], bit i = block at depth i along the axis.
    cols: [block_count][3][padded * padded]u64,
    solid: [3][padded * padded]u64,
    non_air: [3][padded * padded]u64,
    present: [block_count]bool,
};

threadlocal var scratch: Scratch = undefined;

/// Concatenates the six per-face lists into one slice sorted by face, and takes
/// ownership of `lists`: they are freed before this returns, on both the success
/// and the allocation-failure path. Callers must not deinit/free `lists` themselves.
fn fromLists(gpa: Allocator, lists: *[6]std.ArrayList(Quad)) Allocator.Error!Mesh {
    var counts: [6]u32 = undefined;
    var total: usize = 0;
    for (lists, 0..) |l, f| {
        counts[f] = @intCast(l.items.len);
        total += l.items.len;
    }
    const quads = gpa.alloc(Quad, total) catch |err| {
        for (lists) |*l| l.deinit(gpa);
        return err;
    };
    var at: usize = 0;
    for (lists) |*l| {
        @memcpy(quads[at..][0..l.items.len], l.items);
        at += l.items.len;
        l.deinit(gpa);
    }
    return .{ .quads = quads, .counts = counts };
}

/// Binary greedy mesher.
pub fn mesh(gpa: Allocator, vol: *const Volume) Allocator.Error!Mesh {
    const s = &scratch;
    s.present = @splat(false);

    // 1. One pass over the volume: set the block's bit in its column on each axis.
    //    A type's columns are zeroed lazily, the first time the type shows up.
    for (0..padded) |y| for (0..padded) |z| for (0..padded) |x| {
        const b = vol[volumeIndex(x, y, z)];
        if (b == .air) continue;
        const t = @intFromEnum(b);
        if (!s.present[t]) {
            s.present[t] = true;
            @memset(std.mem.asBytes(&s.cols[t]), 0);
        }
        // axis X: u = z, v = y, depth = x ; axis Y: u = x, v = z, depth = y ; axis Z: u = x, v = y, depth = z
        s.cols[t][0][y * padded + z] |= @as(u64, 1) << @intCast(x);
        s.cols[t][1][z * padded + x] |= @as(u64, 1) << @intCast(y);
        s.cols[t][2][y * padded + x] |= @as(u64, 1) << @intCast(z);
    };

    // 2. Occluder masks: OR of the present types' columns.
    @memset(std.mem.asBytes(&s.solid), 0);
    @memset(std.mem.asBytes(&s.non_air), 0);
    for (0..block_count) |t| {
        if (!s.present[t]) continue;
        const solid = (@as(Block, @enumFromInt(t))).isSolid();
        for (0..3) |axis| for (0..padded * padded) |i| {
            s.non_air[axis][i] |= s.cols[t][axis][i];
            if (solid) s.solid[axis][i] |= s.cols[t][axis][i];
        };
    }

    var lists: [6]std.ArrayList(Quad) = @splat(.empty);
    {
        errdefer for (&lists) |*l| l.deinit(gpa);

        const inner: u64 = ((@as(u64, 1) << cs) - 1) << 1; // bits 1..32
        for (0..block_count) |t| {
            if (!s.present[t] or t == @intFromEnum(Block.air)) continue;
            const block: Block = @enumFromInt(t);
            for (0..6) |f| {
                const face: Face = @enumFromInt(f);
                const axis = f / 2;
                const occ = if (block == .water) &s.non_air[axis] else &s.solid[axis];
                // planes[depth][v]: bit u set = visible face at (depth, u, v). Inner coords 0..31.
                var planes: [cs][cs]u32 = @splat(@splat(0));
                var any = false;
                for (0..cs) |v| for (0..cs) |u| {
                    const i = (v + 1) * padded + (u + 1);
                    const c = s.cols[t][axis][i];
                    if (c == 0) continue;
                    // Positive face: neighbor at depth + 1 must not occlude.
                    const visible = inner & c & ~(if (f % 2 == 0) occ[i] >> 1 else occ[i] << 1);
                    var bits = visible;
                    while (bits != 0) : (bits &= bits - 1) {
                        const d = @ctz(bits) - 1;
                        planes[d][v] |= @as(u32, 1) << @intCast(u);
                        any = true;
                    }
                };
                if (!any) continue;
                for (&planes, 0..) |*plane, d| try greedyPlane(gpa, &lists[f], plane, face, block, d);
            }
        }
    }

    return fromLists(gpa, &lists);
}

/// Greedy-merges one 32×32 binary plane (rows = v, bits = u) into rectangles.
fn greedyPlane(gpa: Allocator, out: *std.ArrayList(Quad), plane: *[cs]u32, face: Face, block: Block, d: usize) Allocator.Error!void {
    for (0..cs) |v| {
        while (plane[v] != 0) {
            const row: u64 = plane[v];
            const u_start = @ctz(row);
            const w = @ctz(~(row >> @intCast(u_start))); // run length, ≤ 32 thanks to u64
            const mask: u32 = @truncate(((@as(u64, 1) << @intCast(w)) - 1) << @intCast(u_start));
            var h: usize = 1;
            while (v + h < cs and plane[v + h] & mask == mask) : (h += 1) plane[v + h] &= ~mask;
            plane[v] &= ~mask;
            const p = toXyz(face, d, u_start, v);
            try out.append(gpa, .{
                .x = @intCast(p[0]),
                .y = @intCast(p[1]),
                .z = @intCast(p[2]),
                .w = @intCast(w),
                .h = @intCast(h),
                .face = face,
                .block = block,
            });
        }
    }
}

/// Reference mesher: one quad per visible face. Slow, obviously correct; used by tests.
pub fn meshNaive(gpa: Allocator, vol: *const Volume) Allocator.Error!Mesh {
    var lists: [6]std.ArrayList(Quad) = @splat(.empty);
    {
        errdefer for (&lists) |*l| l.deinit(gpa);
        for (0..cs) |y| for (0..cs) |z| for (0..cs) |x| {
            const b = vol[volumeIndex(x + 1, y + 1, z + 1)];
            for (0..6) |f| {
                const face: Face = @enumFromInt(f);
                const n = face.normal();
                const nb = vol[
                    volumeIndex(
                        @intCast(@as(i32, @intCast(x + 1)) + n[0]),
                        @intCast(@as(i32, @intCast(y + 1)) + n[1]),
                        @intCast(@as(i32, @intCast(z + 1)) + n[2]),
                    )
                ];
                if (!b.faceVisible(nb)) continue;
                try lists[f].append(gpa, .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z), .w = 1, .h = 1, .face = face, .block = b });
            }
        };
    }

    return fromLists(gpa, &lists);
}

// ---
// Tests
// ---

const testing = std.testing;

fn setInner(vol: *Volume, x: usize, y: usize, z: usize, b: Block) void {
    vol[volumeIndex(x + 1, y + 1, z + 1)] = b;
}

/// Expands quads to unit faces: covered[face][block index] = block type, and
/// fails if any unit face is covered twice.
fn coverage(gpa: Allocator, m: Mesh) ![]Block {
    const covered = try gpa.alloc(Block, 6 * Chunk.volume);
    @memset(covered, .air);
    for (m.quads) |q| {
        for (0..q.h) |dv| for (0..q.w) |du| {
            const base = toXyz(q.face, 0, du, dv);
            const x = q.x + base[0];
            const y = q.y + base[1];
            const z = q.z + base[2];
            const i = @as(usize, @intFromEnum(q.face)) * Chunk.volume + x + cs * z + cs * cs * y;
            if (covered[i] != .air) return error.Overlap;
            covered[i] = q.block;
        };
    }
    return covered;
}

test "quad packs into u64 and round-trips" {
    const q: Quad = .{ .x = 31, .y = 0, .z = 17, .w = 32, .h = 1, .face = .neg_z, .block = .leaves };
    const bits: u64 = @bitCast(q);
    try testing.expectEqual(q, @as(Quad, @bitCast(bits)));
}

test "single block gives 6 quads" {
    var vol: Volume = @splat(.air);
    setInner(&vol, 5, 5, 5, .stone);
    var m = try mesh(testing.allocator, &vol);
    defer m.deinit(testing.allocator);
    try testing.expectEqual(6, m.quads.len);
    try testing.expectEqual([6]u32{ 1, 1, 1, 1, 1, 1 }, m.counts);
}

test "full chunk gives 6 quads of 32x32" {
    var vol: Volume = @splat(.air);
    for (0..cs) |y| for (0..cs) |z| for (0..cs) |x| setInner(&vol, x, y, z, .stone);
    var m = try mesh(testing.allocator, &vol);
    defer m.deinit(testing.allocator);
    try testing.expectEqual(6, m.quads.len);
    for (m.quads) |q| {
        try testing.expectEqual(32, q.w);
        try testing.expectEqual(32, q.h);
    }
}

test "different block types are not merged" {
    var vol: Volume = @splat(.air);
    setInner(&vol, 0, 0, 0, .stone);
    setInner(&vol, 1, 0, 0, .dirt);
    var m = try mesh(testing.allocator, &vol);
    defer m.deinit(testing.allocator);
    try testing.expectEqual(10, m.quads.len); // 2 × 6 minus the 2 shared faces
}

test "solid border neighbor hides faces, missing neighbor does not" {
    var vol: Volume = @splat(.air);
    setInner(&vol, 0, 0, 0, .stone);
    vol[volumeIndex(0, 1, 1)] = .stone; // -X neighbor chunk
    var m = try mesh(testing.allocator, &vol);
    defer m.deinit(testing.allocator);
    try testing.expectEqual(5, m.quads.len);
    try testing.expectEqual(0, m.counts[@intFromEnum(Face.neg_x)]);
}

test "buildVolume copies neighbor borders" {
    var center: Chunk = .air;
    var east: Chunk = .air;
    east.set(.{ .x = 0, .y = 3, .z = 4 }, .stone);
    var vol: Volume = undefined;
    buildVolume(&center, .{ &east, null, null, null, null, null }, &vol);
    try testing.expectEqual(Block.stone, vol[volumeIndex(padded - 1, 4, 5)]);
    try testing.expectEqual(Block.air, vol[volumeIndex(0, 4, 5)]);
}

test "greedy covers exactly the naive faces on random volumes" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const rand = prng.random();
    const vol = try gpa.create(Volume);
    defer gpa.destroy(vol);
    for (0..20) |round| {
        // Vary density so both sparse and dense volumes are covered.
        const density = 0.1 + 0.8 * @as(f32, @floatFromInt(round)) / 20;
        for (vol) |*b| {
            b.* = if (rand.float(f32) >= density) .air else switch (rand.uintLessThan(u8, 4)) {
                0 => .stone,
                1 => .dirt,
                2 => .water,
                else => .leaves,
            };
        }
        var greedy = try mesh(gpa, vol);
        defer greedy.deinit(gpa);
        var naive = try meshNaive(gpa, vol);
        defer naive.deinit(gpa);
        const a = try coverage(gpa, greedy);
        defer gpa.free(a);
        const b = try coverage(gpa, naive);
        defer gpa.free(b);
        try testing.expectEqualSlices(Block, b, a);
        try testing.expect(greedy.quads.len <= naive.quads.len);
    }
}

/// Quads for `face`, as sorted into `m.quads` (grouped by `Face` order).
fn quadsForFace(m: Mesh, face: Face) []Quad {
    const f = @intFromEnum(face);
    var start: usize = 0;
    for (0..f) |i| start += m.counts[i];
    return m.quads[start..][0..m.counts[f]];
}

test "quad width/height axis convention: strip along z pins ±X" {
    var vol: Volume = @splat(.air);
    setInner(&vol, 5, 5, 5, .stone);
    setInner(&vol, 5, 5, 6, .stone);
    setInner(&vol, 5, 5, 7, .stone);
    var m = try mesh(testing.allocator, &vol);
    defer m.deinit(testing.allocator);

    for ([_]Face{ .pos_x, .neg_x }) |face| {
        const qs = quadsForFace(m, face);
        try testing.expectEqual(1, qs.len);
        try testing.expectEqual(3, qs[0].w);
        try testing.expectEqual(1, qs[0].h);
        try testing.expectEqual(5, qs[0].x);
        try testing.expectEqual(5, qs[0].y);
        try testing.expectEqual(5, qs[0].z);
    }
}

test "quad width/height axis convention: strip along x pins ±Y and ±Z" {
    var vol: Volume = @splat(.air);
    setInner(&vol, 5, 5, 5, .stone);
    setInner(&vol, 6, 5, 5, .stone);
    setInner(&vol, 7, 5, 5, .stone);
    var m = try mesh(testing.allocator, &vol);
    defer m.deinit(testing.allocator);

    for ([_]Face{ .pos_y, .neg_y, .pos_z, .neg_z }) |face| {
        const qs = quadsForFace(m, face);
        try testing.expectEqual(1, qs.len);
        try testing.expectEqual(3, qs[0].w);
        try testing.expectEqual(1, qs[0].h);
        try testing.expectEqual(5, qs[0].x);
        try testing.expectEqual(5, qs[0].y);
        try testing.expectEqual(5, qs[0].z);
    }
}

test "quad width/height axis convention: strip along y pins ±X" {
    var vol: Volume = @splat(.air);
    setInner(&vol, 5, 5, 5, .stone);
    setInner(&vol, 5, 6, 5, .stone);
    setInner(&vol, 5, 7, 5, .stone);
    var m = try mesh(testing.allocator, &vol);
    defer m.deinit(testing.allocator);

    for ([_]Face{ .pos_x, .neg_x }) |face| {
        const qs = quadsForFace(m, face);
        try testing.expectEqual(1, qs.len);
        try testing.expectEqual(1, qs[0].w);
        try testing.expectEqual(3, qs[0].h);
        try testing.expectEqual(5, qs[0].x);
        try testing.expectEqual(5, qs[0].y);
        try testing.expectEqual(5, qs[0].z);
    }
}

test "greedy merges on real terrain" {
    const terrain = @import("terrain.zig");
    const gpa = testing.allocator;
    const g: terrain.Generator = .init(42);
    var c: Chunk = undefined;
    terrain.generate(&g, .{ .x = 0, .y = 2, .z = 0 }, &c);
    const vol = try gpa.create(Volume);
    defer gpa.destroy(vol);
    buildVolume(&c, @splat(null), vol);
    var greedy = try mesh(gpa, vol);
    defer greedy.deinit(gpa);
    var naive = try meshNaive(gpa, vol);
    defer naive.deinit(gpa);
    const a = try coverage(gpa, greedy);
    defer gpa.free(a);
    const b = try coverage(gpa, naive);
    defer gpa.free(b);
    try testing.expectEqualSlices(Block, b, a);
    try testing.expect(greedy.quads.len * 2 < naive.quads.len);
}
