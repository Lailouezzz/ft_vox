//! `zig build bench`: time generation and binary greedy meshing on real terrain.
const std = @import("std");
const world = @import("world");
const mesher = world.mesher;

const radius = 4; // chunks around the origin, all heights

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.smp_allocator;
    const io = init.io;
    const g: world.Generator = .init(42);

    const side = 2 * radius + 1;
    const chunks = try gpa.alloc(world.Chunk, side * side * world.height_chunks);
    defer gpa.free(chunks);

    const gen_start = std.Io.Clock.awake.now(io);
    for (chunks, 0..) |*c, i| world.generate(&g, posOf(i), c);
    const gen_ns = gen_start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();

    const vol = try gpa.create(mesher.Volume);
    defer gpa.destroy(vol);
    var meshed: usize = 0;
    var quads: usize = 0;
    var mesh_ns: i96 = 0;
    for (chunks, 0..) |*c, i| {
        if (c.isEmpty()) continue;
        mesher.buildVolume(c, neighbors(chunks, i), vol);
        const t = std.Io.Clock.awake.now(io);
        var m = try mesher.mesh(gpa, vol);
        mesh_ns += t.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        quads += m.quads.len;
        meshed += 1;
        m.deinit(gpa);
    }

    std.debug.print(
        \\generate: {d} chunks, {d:.1} us/chunk
        \\mesh:     {d} non-empty chunks, {d:.1} us/chunk, {d:.0} quads/chunk
        \\
    , .{
        chunks.len,
        @as(f64, @floatFromInt(gen_ns)) / @as(f64, @floatFromInt(chunks.len)) / 1000,
        meshed,
        @as(f64, @floatFromInt(mesh_ns)) / @as(f64, @floatFromInt(meshed)) / 1000,
        @as(f64, @floatFromInt(quads)) / @as(f64, @floatFromInt(meshed)),
    });
}

fn posOf(i: usize) world.ChunkPos {
    const side = 2 * radius + 1;
    return .{
        .x = @as(i32, @intCast(i % side)) - radius,
        .z = @as(i32, @intCast(i / side % side)) - radius,
        .y = @intCast(i / (side * side)),
    };
}

fn neighbors(chunks: []const world.Chunk, i: usize) [6]?*const world.Chunk {
    const p = posOf(i);
    var out: [6]?*const world.Chunk = @splat(null);
    for (0..6) |f| {
        const n = (world.Face.normal(@enumFromInt(f)));
        const q: world.ChunkPos = .{ .x = p.x + n[0], .y = p.y + n[1], .z = p.z + n[2] };
        if (@abs(q.x) > radius or @abs(q.z) > radius or !q.inWorld()) continue;
        const side = 2 * radius + 1;
        const j: usize = @intCast((q.x + radius) + (q.z + radius) * side + q.y * side * side);
        out[f] = &chunks[j];
    }
    return out;
}
