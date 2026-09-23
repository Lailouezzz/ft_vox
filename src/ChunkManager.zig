//! Chunk streaming and editing. Lives on the main thread between two worker
//! pools: gen_in → GenPool → gen_out → ChunkManager → mesh_in → MeshPool → mesh_out.
//! No GPU code: the renderer consumes `takeUploads` / `takeUnloads` each frame
//! and keys its allocations by `ChunkPos`.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const threading = @import("threading");
const world = @import("world");
const Chunk = world.Chunk;
const ChunkPos = world.ChunkPos;
const BlockPos = world.BlockPos;
const Face = world.Face;
const Mesh = world.mesher.Mesh;
const Volume = world.mesher.Volume;

const ChunkManager = @This();

pub const Config = struct {
    seed: u64,
    /// Horizontal meshing radius in chunks, Euclidean on (x, z). Chunks are
    /// generated up to `radius + 1` so every meshed chunk has its 6 neighbours.
    radius: u16 = 16,
    /// Chunks are unloaded beyond `radius + unload_margin` (hysteresis).
    unload_margin: u16 = 2,
    gen_workers: usize,
    mesh_workers: usize,
};

pub const Upload = struct { pos: ChunkPos, mesh: Mesh };

pub const Stats = struct {
    loaded: usize = 0,
    /// Generation jobs sent and not yet received back.
    gen_queued: usize = 0,
    /// Mesh jobs sent and not yet received back.
    mesh_in_flight: usize = 0,
    mesh_dispatched: u64 = 0,
    uploads: u64 = 0,
};

/// The sender allocates `chunk`; ownership travels with the job and comes back in the result.
const GenJob = struct { id: u64, pos: ChunkPos, seed: u64, chunk: *Chunk };
const GenResult = struct { id: u64, pos: ChunkPos, chunk: *Chunk };
/// The mesh task owns and frees `volume`: nobody needs the snapshot afterwards.
const MeshJob = struct { id: u64, pos: ChunkPos, version: u32, volume: *Volume };
/// `mesh == null`: meshing failed (OOM), the chunk goes back to dirty.
const MeshResult = struct { id: u64, pos: ChunkPos, version: u32, mesh: ?Mesh };

const GenPool = threading.WorkerPool(GenJob, GenResult);
const MeshPool = threading.WorkerPool(MeshJob, MeshResult);
const WorkQueue = threading.WorkQueue;

const Entry = struct {
    /// Unique per load: detects results for a chunk unloaded then reloaded.
    id: u64,
    /// The chunk is owned by the gen job while generating, by the entry afterwards.
    state: union(enum) { generating, generated: *Chunk },
    /// Starts at 1 so that `displayed_version == 0` means "nothing displayed".
    version: u32 = 1,
    dirty: bool = true,
    mesh_in_flight: bool = false,
    displayed_version: u32 = 0,
};

gpa: Allocator,
io: Io,
config: Config,

gen_in: WorkQueue(GenJob) = .empty,
gen_out: WorkQueue(GenResult) = .empty,
mesh_in: WorkQueue(MeshJob) = .empty,
mesh_out: WorkQueue(MeshResult) = .empty,
gen_pool: GenPool,
mesh_pool: MeshPool,

entries: std.AutoHashMapUnmanaged(ChunkPos, Entry) = .empty,
// ponytail: grows without bound (one 32 KiB copy per edited chunk); spill to disk if it matters.
edited: std.AutoHashMapUnmanaged(ChunkPos, *Chunk) = .empty,
uploads: std.ArrayList(Upload) = .empty,
unloads: std.ArrayList(ChunkPos) = .empty,
next_id: u64 = 0,
/// Center column of the last completed streaming pass; null before the first one.
last_center: ?[2]i32 = null,
stats: Stats = .{},

/// Heap-allocated because the pools keep pointers to the queues.
/// `gpa` must be thread-safe: workers allocate and free with it.
pub fn create(gpa: Allocator, io: Io, config: Config) !*ChunkManager {
    const self = try gpa.create(ChunkManager);
    errdefer gpa.destroy(self);
    self.* = .{
        .gpa = gpa,
        .io = io,
        .config = config,
        .gen_pool = .init(&self.gen_in, &self.gen_out, config.gen_workers, genTask),
        .mesh_pool = .init(&self.mesh_in, &self.mesh_out, config.mesh_workers, meshTask),
    };
    // Nothing is queued yet: shutting down only joins the workers already spawned.
    errdefer self.gen_pool.shutdown(gpa, io);
    try self.gen_pool.start(gpa, io);
    errdefer self.mesh_pool.shutdown(gpa, io);
    try self.mesh_pool.start(gpa, io);
    return self;
}

pub fn destroy(self: *ChunkManager) void {
    const gpa = self.gpa;
    const io = self.io;
    // Queued jobs own memory: pop and free them instead of letting the pools drain them.
    self.gen_in.close(io);
    self.mesh_in.close(io);
    while (popClosed(GenJob, &self.gen_in, io)) |job| gpa.destroy(job.chunk);
    while (popClosed(MeshJob, &self.mesh_in, io)) |job| gpa.destroy(job.volume);
    // Output queues stay open until here: a worker pushing into a closed one leaks.
    self.gen_pool.shutdown(gpa, io);
    self.mesh_pool.shutdown(gpa, io);
    self.gen_out.close(io);
    self.mesh_out.close(io);
    while (popClosed(GenResult, &self.gen_out, io)) |r| gpa.destroy(r.chunk);
    while (popClosed(MeshResult, &self.mesh_out, io)) |r| if (r.mesh) |m| gpa.free(m.quads);

    self.gen_in.deinit(gpa);
    self.gen_out.deinit(gpa);
    self.mesh_in.deinit(gpa);
    self.mesh_out.deinit(gpa);
    var it = self.entries.valueIterator();
    while (it.next()) |e| switch (e.state) {
        .generating => {},
        .generated => |c| gpa.destroy(c),
    };
    self.entries.deinit(gpa);
    var edited_it = self.edited.valueIterator();
    while (edited_it.next()) |c| gpa.destroy(c.*);
    self.edited.deinit(gpa);
    self.freeUploads();
    self.uploads.deinit(gpa);
    self.unloads.deinit(gpa);
    gpa.destroy(self);
}

/// Pops from a closed queue, spinning past lock contention. Null once empty.
fn popClosed(comptime T: type, queue: *WorkQueue(T), io: Io) ?T {
    while (true) {
        if (queue.pop(io) catch return null) |item| return item;
    }
}

/// Once per frame, main thread. Invalidates the slices of `takeUploads` / `takeUnloads`.
pub fn update(self: *ChunkManager, center: BlockPos) !void {
    self.freeUploads();
    self.unloads.clearRetainingCapacity();

    while (try self.gen_out.pop(self.io)) |r| self.handleGenResult(r);
    while (try self.mesh_out.pop(self.io)) |r| try self.handleMeshResult(r);
    try self.dispatchMeshes();

    // The desired set only depends on the center column, so the streaming
    // pass runs when it changes. Recorded last: a failed pass is retried.
    const c = center.chunk();
    const column: [2]i32 = .{ c.x, c.z };
    if (self.last_center == null or !std.meta.eql(self.last_center.?, column)) {
        try self.enqueueMissing(c);
        try self.unloadFar(column);
        self.last_center = column;
    }
    self.stats.loaded = self.entries.count();
}

/// Meshes ready for the GPU. Valid until the next `update`, which frees them.
/// A mesh with 0 quads still means "replace": free the old allocation.
/// Apply uploads before unloads: a chunk may appear in both in the same frame.
pub fn takeUploads(self: *const ChunkManager) []const Upload {
    return self.uploads.items;
}

/// Chunks unloaded this frame. Valid until the next `update`.
pub fn takeUnloads(self: *const ChunkManager) []const ChunkPos {
    return self.unloads.items;
}

/// Main thread. Sets `pos` to air if its chunk is loaded and the block is not air.
/// Returns whether a block changed.
pub fn breakBlock(self: *ChunkManager, pos: BlockPos) !bool {
    const cpos = pos.chunk();
    const chunk = self.loadedChunk(cpos) orelse return false;
    const local = pos.local();
    if (chunk.get(local) == .air) return false;

    // Allocate before mutating: on OOM the world is left untouched.
    const slot = try self.edited.getOrPut(self.gpa, cpos);
    if (!slot.found_existing) {
        slot.value_ptr.* = self.gpa.create(Chunk) catch |err| {
            self.edited.removeByPtr(slot.key_ptr);
            return err;
        };
    }
    chunk.set(local, .air);
    slot.value_ptr.*.* = chunk.*;
    markEdited(self.entries.getPtr(cpos).?);

    // A border block exposes a face of the neighbour: remesh it too.
    const l = [3]u5{ local.x, local.y, local.z };
    const faces = [3][2]Face{ .{ .neg_x, .pos_x }, .{ .neg_y, .pos_y }, .{ .neg_z, .pos_z } };
    for (l, faces) |v, f| {
        const face = switch (v) {
            0 => f[0],
            Chunk.size - 1 => f[1],
            else => continue,
        };
        if (self.entries.getPtr(neighbour(cpos, face))) |e| markEdited(e);
    }
    return true;
}

/// Main thread. Air when the chunk is not loaded. Makes `ChunkManager` a `raycast` lookup.
pub fn blockAt(self: *const ChunkManager, pos: BlockPos) world.Block {
    const chunk = self.loadedChunk(pos.chunk()) orelse return .air;
    return chunk.get(pos.local());
}

fn loadedChunk(self: *const ChunkManager, pos: ChunkPos) ?*Chunk {
    const e = self.entries.getPtr(pos) orelse return null;
    return switch (e.state) {
        .generating => null,
        .generated => |c| c,
    };
}

fn markEdited(e: *Entry) void {
    e.version += 1;
    e.dirty = true;
}

fn neighbour(pos: ChunkPos, face: Face) ChunkPos {
    const n = face.normal();
    return .{ .x = pos.x + n[0], .y = pos.y + n[1], .z = pos.z + n[2] };
}

fn freeUploads(self: *ChunkManager) void {
    for (self.uploads.items) |*u| u.mesh.deinit(self.gpa);
    self.uploads.clearRetainingCapacity();
}

fn handleGenResult(self: *ChunkManager, r: GenResult) void {
    self.stats.gen_queued -= 1;
    const e = self.entries.getPtr(r.pos) orelse return self.gpa.destroy(r.chunk);
    if (e.id != r.id) return self.gpa.destroy(r.chunk);
    e.state = .{ .generated = r.chunk };
}

fn handleMeshResult(self: *ChunkManager, r: MeshResult) !void {
    self.stats.mesh_in_flight -= 1;
    var mesh = r.mesh orelse {
        if (self.entries.getPtr(r.pos)) |e| if (e.id == r.id) {
            e.mesh_in_flight = false;
            e.dirty = true;
        };
        return;
    };
    errdefer mesh.deinit(self.gpa);
    const e = self.entries.getPtr(r.pos) orelse return mesh.deinit(self.gpa);
    if (e.id != r.id) return mesh.deinit(self.gpa);
    e.mesh_in_flight = false;
    if (r.version <= e.displayed_version) return mesh.deinit(self.gpa);
    try self.uploads.append(self.gpa, .{ .pos = r.pos, .mesh = mesh });
    e.displayed_version = r.version;
    self.stats.uploads += 1;
}

fn dispatchMeshes(self: *ChunkManager) !void {
    const center = self.last_center orelse return;
    const r: i64 = self.config.radius;
    var it = self.entries.iterator();
    // ponytail: scans every entry each frame; keep a dirty list if it shows in a profile.
    next: while (it.next()) |kv| {
        const pos = kv.key_ptr.*;
        const e = kv.value_ptr;
        if (!e.dirty or e.mesh_in_flight) continue;
        const chunk = switch (e.state) {
            .generating => continue,
            .generated => |c| c,
        };
        if (horizontalDist2(pos, center) > r * r) continue;
        // Never displayed and empty: nothing to draw. Once displayed, an
        // emptied chunk still sends a 0-quad mesh so the renderer frees it.
        if (e.displayed_version == 0 and chunk.isEmpty()) {
            e.dirty = false;
            continue;
        }
        var neighbours: [6]?*const Chunk = undefined;
        for (&neighbours, 0..) |*n, f| {
            const npos = neighbour(pos, @enumFromInt(f));
            n.* = if (!npos.inWorld()) null else self.loadedChunk(npos) orelse continue :next;
        }

        const volume = try self.gpa.create(Volume);
        errdefer self.gpa.destroy(volume);
        world.mesher.buildVolume(chunk, neighbours, volume);
        const job: MeshJob = .{ .id = e.id, .pos = pos, .version = e.version, .volume = volume };
        // Version > 1 means edited since load: the player is waiting, jump the queue.
        if (e.version > 1)
            try self.mesh_in.pushFront(self.gpa, self.io, job)
        else
            try self.mesh_in.push(self.gpa, self.io, job);
        e.dirty = false;
        e.mesh_in_flight = true;
        self.stats.mesh_in_flight += 1;
        self.stats.mesh_dispatched += 1;
    }
}

fn horizontalDist2(pos: ChunkPos, center: [2]i32) i64 {
    const dx: i64 = pos.x - center[0];
    const dz: i64 = pos.z - center[1];
    return dx * dx + dz * dz;
}

/// Loads every chunk within `radius + 1` of `center` (all cy), nearest first.
fn enqueueMissing(self: *ChunkManager, center: ChunkPos) !void {
    const gpa = self.gpa;
    const r: i32 = @as(i32, self.config.radius) + 1;
    const column: [2]i32 = .{ center.x, center.z };

    var missing: std.ArrayList(ChunkPos) = .empty;
    defer missing.deinit(gpa);
    var dz: i32 = -r;
    while (dz <= r) : (dz += 1) {
        var dx: i32 = -r;
        while (dx <= r) : (dx += 1) {
            for (0..world.height_chunks) |cy| {
                const pos: ChunkPos = .{ .x = center.x + dx, .y = @intCast(cy), .z = center.z + dz };
                if (horizontalDist2(pos, column) > r * r or self.entries.contains(pos)) continue;
                try missing.append(gpa, pos);
            }
        }
    }
    const Near = struct {
        c: ChunkPos,
        fn dist2(ctx: @This(), p: ChunkPos) i64 {
            const dy: i64 = p.y - ctx.c.y;
            return horizontalDist2(p, .{ ctx.c.x, ctx.c.z }) + dy * dy;
        }
        fn lessThan(ctx: @This(), a: ChunkPos, b: ChunkPos) bool {
            return ctx.dist2(a) < ctx.dist2(b);
        }
    };
    std.mem.sort(ChunkPos, missing.items, Near{ .c = center }, Near.lessThan);

    // ponytail: jobs queued for an area the camera left still run before newer
    // ones; re-prioritise gen_in if fast travel makes that visible.
    for (missing.items) |pos| {
        try self.entries.ensureUnusedCapacity(gpa, 1);
        const id = self.next_id;
        self.next_id += 1;
        if (self.edited.get(pos)) |saved| {
            const chunk = try gpa.create(Chunk);
            chunk.* = saved.*;
            self.entries.putAssumeCapacityNoClobber(pos, .{ .id = id, .state = .{ .generated = chunk } });
            continue;
        }
        const chunk = try gpa.create(Chunk);
        errdefer gpa.destroy(chunk);
        try self.gen_in.push(gpa, self.io, .{ .id = id, .pos = pos, .seed = self.config.seed, .chunk = chunk });
        // Capacity reserved above: nothing can fail once the job owns the chunk.
        self.entries.putAssumeCapacityNoClobber(pos, .{ .id = id, .state = .generating });
        self.stats.gen_queued += 1;
    }
}

/// Unloads chunks beyond `radius + unload_margin` of `center` and records them in `unloads`.
fn unloadFar(self: *ChunkManager, center: [2]i32) !void {
    const r: i64 = @as(i64, self.config.radius) + self.config.unload_margin;
    var it = self.entries.keyIterator();
    while (it.next()) |pos| {
        if (horizontalDist2(pos.*, center) > r * r) try self.unloads.append(self.gpa, pos.*);
    }
    // A generating chunk is freed when its result comes back with an unknown id;
    // an in-flight mesh likewise.
    for (self.unloads.items) |pos| {
        const e = self.entries.fetchRemove(pos).?.value;
        switch (e.state) {
            .generating => {},
            .generated => |c| self.gpa.destroy(c),
        }
    }
}

/// Builds the generator in the task: `Generator.init` is a few field stores, and it
/// keeps the job a plain value with no pointer into the manager.
fn genTask(_: Allocator, _: Io, job: GenJob) anyerror!?GenResult {
    const generator: world.Generator = .init(job.seed);
    world.generate(&generator, job.pos, job.chunk);
    return .{ .id = job.id, .pos = job.pos, .chunk = job.chunk };
}

/// Never returns an error: the pool would drop the result and leak what it owns.
fn meshTask(gpa: Allocator, _: Io, job: MeshJob) anyerror!?MeshResult {
    defer gpa.destroy(job.volume);
    const mesh = world.mesher.mesh(gpa, job.volume) catch null;
    return .{ .id = job.id, .pos = job.pos, .version = job.version, .mesh = mesh };
}

// ---
// Tests
// ---

const testing = std.testing;

const PosSet = std.AutoHashMapUnmanaged(ChunkPos, void);

const test_config: Config = .{ .seed = 42, .radius = 2, .unload_margin = 2, .gen_workers = 2, .mesh_workers = 2 };

fn createTest(config: Config) !*ChunkManager {
    return create(testing.allocator, testing.io, config);
}

fn blockPos(cx: i32, cz: i32) BlockPos {
    return .{ .x = cx * Chunk.size + 16, .y = 100, .z = cz * Chunk.size + 16 };
}

/// Pumps `update` until nothing is generating or meshing, recording every
/// upload and unload position seen on the way. Fails after ~10 s.
fn settle(cm: *ChunkManager, center: BlockPos, uploaded: ?*PosSet, unloaded: ?*PosSet) !void {
    for (0..10_000) |_| {
        try cm.update(center);
        if (uploaded) |set| for (cm.takeUploads()) |u| try set.put(testing.allocator, u.pos, {});
        if (unloaded) |set| for (cm.takeUnloads()) |p| try set.put(testing.allocator, p, {});
        if (cm.stats.gen_queued == 0 and cm.stats.mesh_in_flight == 0) return;
        try testing.io.sleep(.fromMilliseconds(1), .awake);
    }
    return error.Timeout;
}

/// First non-air block of chunk `c` whose local coordinates satisfy `pred`.
fn findSolid(cm: *const ChunkManager, c: ChunkPos, comptime pred: fn (world.LocalPos) bool) ?BlockPos {
    const o = c.origin();
    for (0..Chunk.size) |y| for (0..Chunk.size) |z| for (0..Chunk.size) |x| {
        const l: world.LocalPos = .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) };
        const p: BlockPos = .{ .x = o.x + l.x, .y = o.y + l.y, .z = o.z + l.z };
        if (pred(l) and cm.blockAt(p) != .air) return p;
    };
    return null;
}

fn interior(l: world.LocalPos) bool {
    for ([_]u5{ l.x, l.y, l.z }) |v| if (v == 0 or v == Chunk.size - 1) return false;
    return true;
}

test "generates radius + 1, meshes every non-empty chunk within radius" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    var uploaded: PosSet = .empty;
    defer uploaded.deinit(testing.allocator);
    try settle(cm, blockPos(0, 0), &uploaded, null);

    const r: i32 = test_config.radius;
    var dx: i32 = -r - 2;
    while (dx <= r + 2) : (dx += 1) {
        var dz: i32 = -r - 2;
        while (dz <= r + 2) : (dz += 1) {
            const d2 = dx * dx + dz * dz;
            for (0..world.height_chunks) |cy| {
                const pos: ChunkPos = .{ .x = dx, .y = @intCast(cy), .z = dz };
                const chunk = cm.loadedChunk(pos);
                try testing.expectEqual(d2 <= (r + 1) * (r + 1), chunk != null);
                const expect_upload = d2 <= r * r and !chunk.?.isEmpty();
                try testing.expectEqual(expect_upload, uploaded.contains(pos));
            }
        }
    }
    try testing.expectEqual(cm.entries.count(), cm.stats.loaded);
}

test "unload hysteresis" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    var unloaded: PosSet = .empty;
    defer unloaded.deinit(testing.allocator);
    try settle(cm, blockPos(0, 0), null, &unloaded);

    // Moving by margin - 1 keeps every chunk within radius + margin.
    try settle(cm, blockPos(1, 0), null, &unloaded);
    try testing.expectEqual(0, unloaded.count());

    var before: PosSet = .empty;
    defer before.deinit(testing.allocator);
    var it = cm.entries.keyIterator();
    while (it.next()) |p| try before.put(testing.allocator, p.*, {});

    try cm.update(blockPos(100, 0));
    try testing.expectEqual(before.count(), cm.takeUnloads().len);
    for (cm.takeUnloads()) |p| try testing.expect(before.contains(p));
}

test "breakBlock and blockAt" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    try settle(cm, blockPos(0, 0), null, null);

    // Unloaded, and air (terrain tops out well below y = 250).
    try testing.expect(!try cm.breakBlock(.{ .x = 10_000, .y = 50, .z = 0 }));
    try testing.expect(!try cm.breakBlock(.{ .x = 0, .y = 250, .z = 0 }));

    // A border block (local x = 31) of chunk (0, 0, 0): its +X neighbour is remeshed too.
    const p = findSolid(cm, .{ .x = 0, .y = 0, .z = 0 }, struct {
        fn f(l: world.LocalPos) bool {
            return l.x == Chunk.size - 1 and l.y > 0 and l.y < Chunk.size - 1 and l.z > 0 and l.z < Chunk.size - 1;
        }
    }.f).?;
    const n: ChunkPos = .{ .x = 1, .y = 0, .z = 0 };
    const n_version = cm.entries.get(n).?.version;
    try testing.expect(try cm.breakBlock(p));
    try testing.expectEqual(world.Block.air, cm.blockAt(p));
    try testing.expect(!try cm.breakBlock(p));
    try testing.expectEqual(n_version + 1, cm.entries.get(n).?.version);

    var uploaded: PosSet = .empty;
    defer uploaded.deinit(testing.allocator);
    try settle(cm, blockPos(0, 0), &uploaded, null);
    try testing.expectEqual(2, uploaded.count());
    try testing.expect(uploaded.contains(p.chunk()));
    try testing.expect(uploaded.contains(n));
}

test "N edits during a mesh give exactly one remesh" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    const center = blockPos(0, 0);
    try settle(cm, center, null, null);

    const c: ChunkPos = .{ .x = 0, .y = 0, .z = 0 };
    const before = cm.stats.mesh_dispatched;
    try testing.expect(try cm.breakBlock(findSolid(cm, c, interior).?));
    try cm.update(center);
    try testing.expectEqual(before + 1, cm.stats.mesh_dispatched);
    for (0..5) |_| try testing.expect(try cm.breakBlock(findSolid(cm, c, interior).?));
    try settle(cm, center, null, null);
    // Either the first result is still in flight at the next update (the chunk
    // stays dirty until the result clears mesh_in_flight), or it already came
    // back (step 3 clears mesh_in_flight before step 4 dispatches). In both
    // cases the 5 edits share a single dirty flag, hence a single dispatch.
    try testing.expectEqual(before + 2, cm.stats.mesh_dispatched);
}

test "mesh result older than the displayed one is dropped" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    try settle(cm, blockPos(0, 0), null, null);
    const pos: ChunkPos = .{ .x = 0, .y = 0, .z = 0 };
    const e = cm.entries.getPtr(pos).?;
    e.displayed_version = 5;

    const Case = struct { version: u32, id: u64, uploaded: bool };
    for ([_]Case{
        .{ .version = 3, .id = e.id, .uploaded = false },
        .{ .version = 5, .id = e.id, .uploaded = false },
        .{ .version = 6, .id = e.id + 1_000_000, .uploaded = false },
        .{ .version = 6, .id = e.id, .uploaded = true },
    }) |case| {
        cm.freeUploads();
        cm.stats.mesh_in_flight += 1; // as if dispatched
        const mesh: Mesh = .{ .quads = try testing.allocator.alloc(world.mesher.Quad, 1), .counts = @splat(0) };
        try cm.handleMeshResult(.{ .id = case.id, .pos = pos, .version = case.version, .mesh = mesh });
        try testing.expectEqual(@intFromBool(case.uploaded), cm.takeUploads().len);
    }
    try testing.expectEqual(6, e.displayed_version);
}

test "edited chunk survives unload and reload" {
    const cm = try createTest(.{ .seed = 42, .radius = 1, .unload_margin = 1, .gen_workers = 2, .mesh_workers = 2 });
    defer cm.destroy();
    try settle(cm, blockPos(0, 0), null, null);
    const c: ChunkPos = .{ .x = 0, .y = 0, .z = 0 };
    const p = findSolid(cm, c, interior).?;
    try testing.expect(try cm.breakBlock(p));

    try settle(cm, blockPos(50, 0), null, null);
    try testing.expect(!cm.entries.contains(c));
    try settle(cm, blockPos(0, 0), null, null);
    try testing.expectEqual(world.Block.air, cm.blockAt(p));
    try testing.expectEqualSlices(world.Block, &cm.edited.get(c).?.blocks, &cm.loadedChunk(c).?.blocks);
}

test "destroy with jobs still queued leaks nothing" {
    const cm = try createTest(.{ .seed = 7, .radius = 6, .gen_workers = 1, .mesh_workers = 1 });
    defer cm.destroy();
    for (0..3) |_| {
        try cm.update(blockPos(0, 0));
        try testing.io.sleep(.fromMilliseconds(5), .awake);
    }
    try testing.expect(cm.stats.gen_queued > 0);
}
