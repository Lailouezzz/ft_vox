//! GPU storage of every chunk mesh: one big quad buffer sub-allocated by a
//! FreeList, and a metadata buffer indexed by slot. Uploads are staged and
//! recorded into the frame's command buffer, within a per-frame byte budget.
//!
//! Synchronization: `record` starts with a barrier ordering every earlier
//! GPU read (culling, indirect draw, vertex pulling) before this frame's
//! transfer writes, so freed ranges and slots can be reused immediately.
const std = @import("std");
const vk = @import("vulkan");
const world = @import("world");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");
const Buffer = @import("Buffer.zig");
const gpu = @import("gpu.zig");

const ChunkBuffers = @This();
const Quad = world.mesher.Quad;

pub const max_quads = 16 * 1024 * 1024; // 128 MiB
pub const max_chunks = 16 * 1024;
/// Staging bytes per frame in flight; larger backlogs spill over to the next frames.
pub const staging_size = 16 * 1024 * 1024;

const Slot = struct { index: u32, range: world.FreeList.Range };
const Pending = struct { pos: world.ChunkPos, quads: []Quad, counts: [world.mesher.groups]u32 };

gpa: Allocator,
quads: Buffer,
metas: Buffer,
ranges: world.FreeList,
slots: std.AutoHashMapUnmanaged(world.ChunkPos, Slot) = .empty,
free_slots: std.ArrayList(u32) = .empty,
/// One past the highest slot index ever used: the culling dispatch size.
slot_high: u32 = 0,
pending: std.ArrayList(Pending) = .empty,
/// Metadata writes to record this frame, keyed by slot.
meta_writes: std.AutoArrayHashMapUnmanaged(u32, gpu.ChunkMeta) = .empty,
quad_count: u64 = 0,

pub fn init(ctx: *const Context, gpa: Allocator) !ChunkBuffers {
    var quads: Buffer = try .init(ctx, max_quads * @sizeOf(Quad), .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, false);
    errdefer quads.deinit(ctx);
    var metas: Buffer = try .init(ctx, max_chunks * @sizeOf(gpu.ChunkMeta), .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, false);
    errdefer metas.deinit(ctx);
    return .{ .gpa = gpa, .quads = quads, .metas = metas, .ranges = try .init(gpa, max_quads) };
}

pub fn deinit(self: *ChunkBuffers, ctx: *const Context) void {
    for (self.pending.items) |p| self.gpa.free(p.quads);
    self.pending.deinit(self.gpa);
    self.meta_writes.deinit(self.gpa);
    self.free_slots.deinit(self.gpa);
    self.slots.deinit(self.gpa);
    self.ranges.deinit(self.gpa);
    self.metas.deinit(ctx);
    self.quads.deinit(ctx);
    self.* = undefined;
}

/// Queues `quads` (copied) as the new mesh of `pos`, replacing any older one.
pub fn upload(self: *ChunkBuffers, pos: world.ChunkPos, quads: []const Quad, counts: [world.mesher.groups]u32) !void {
    self.dropPending(pos);
    const copy = try self.gpa.dupe(Quad, quads);
    errdefer self.gpa.free(copy);
    try self.pending.append(self.gpa, .{ .pos = pos, .quads = copy, .counts = counts });
}

/// Forgets the mesh of `pos` (pending or resident).
pub fn remove(self: *ChunkBuffers, pos: world.ChunkPos) !void {
    self.dropPending(pos);
    try self.release(pos);
}

// ponytail: linear scan per upload, O(n²) over an initial burst; a pos → index
// map if it ever shows in a profile (it does not next to meshing).
fn dropPending(self: *ChunkBuffers, pos: world.ChunkPos) void {
    var i: usize = 0;
    while (i < self.pending.items.len) {
        if (std.meta.eql(self.pending.items[i].pos, pos)) {
            self.gpa.free(self.pending.items[i].quads);
            _ = self.pending.orderedRemove(i);
        } else i += 1;
    }
}

fn release(self: *ChunkBuffers, pos: world.ChunkPos) !void {
    const kv = self.slots.fetchRemove(pos) orelse return;
    try self.ranges.free(self.gpa, kv.value.range);
    self.quad_count -= kv.value.range.len;
    try self.free_slots.append(self.gpa, kv.value.index);
    try self.meta_writes.put(self.gpa, kv.value.index, std.mem.zeroes(gpu.ChunkMeta));
}

/// Records this frame's transfers: pending meshes that fit in `staging`, then
/// metadata updates. The caller issues the barrier that makes them visible to
/// culling and drawing.
pub fn record(self: *ChunkBuffers, cmd: vk.CommandBufferProxy, staging: *const Buffer) !void {
    barrier(cmd, .{ .compute_shader_bit = true, .vertex_shader_bit = true, .draw_indirect_bit = true }, .{ .shader_storage_read_bit = true, .indirect_command_read_bit = true }, .{ .copy_bit = true, .clear_bit = true }, .{ .transfer_write_bit = true });

    const staged = staging.slice(Quad);
    var used: usize = 0;
    var copies: std.ArrayList(vk.BufferCopy) = .empty;
    defer copies.deinit(self.gpa);
    var done: usize = 0;
    for (self.pending.items) |p| {
        if (used + p.quads.len > staged.len) break;
        done += 1;
        if (p.quads.len == 0) {
            try self.release(p.pos);
            continue;
        }
        // Allocate before releasing: when the buffer is full the previous mesh stays visible.
        const range = self.ranges.alloc(@intCast(p.quads.len)) orelse {
            std.log.scoped(.render).warn("quad buffer full, keeping the previous mesh of chunk {any}", .{p.pos});
            continue;
        };
        const index = if (self.slots.get(p.pos)) |old| blk: {
            try self.ranges.free(self.gpa, old.range);
            self.quad_count -= old.range.len;
            break :blk old.index;
        } else self.free_slots.pop() orelse blk: {
            if (self.slot_high == max_chunks) {
                try self.ranges.free(self.gpa, range);
                std.log.scoped(.render).warn("chunk slots full, dropping chunk {any}", .{p.pos});
                continue;
            }
            self.slot_high += 1;
            break :blk self.slot_high - 1;
        };
        @memcpy(staged[used..][0..p.quads.len], p.quads);
        try copies.append(self.gpa, .{ .src_offset = used * @sizeOf(Quad), .dst_offset = @as(u64, range.offset) * @sizeOf(Quad), .size = p.quads.len * @sizeOf(Quad) });
        used += p.quads.len;
        try self.slots.put(self.gpa, p.pos, .{ .index = index, .range = range });
        self.quad_count += range.len;
        const o = p.pos.origin();
        try self.meta_writes.put(self.gpa, index, .{ .origin = .{ o.x, o.y, o.z }, .first_quad = range.offset, .counts = p.counts, .enabled = 1 });
    }
    for (self.pending.items[0..done]) |p| self.gpa.free(p.quads);
    self.pending.replaceRangeAssumeCapacity(0, done, &.{});

    if (copies.items.len > 0) cmd.copyBuffer(staging.handle, self.quads.handle, copies.items);
    var it = self.meta_writes.iterator();
    while (it.next()) |e| cmd.updateBuffer(self.metas.handle, @as(u64, e.key_ptr.*) * @sizeOf(gpu.ChunkMeta), @sizeOf(gpu.ChunkMeta), e.value_ptr);
    self.meta_writes.clearRetainingCapacity();
}

pub fn barrier(cmd: vk.CommandBufferProxy, src_stage: vk.PipelineStageFlags2, src_access: vk.AccessFlags2, dst_stage: vk.PipelineStageFlags2, dst_access: vk.AccessFlags2) void {
    const b: vk.MemoryBarrier2 = .{ .src_stage_mask = src_stage, .src_access_mask = src_access, .dst_stage_mask = dst_stage, .dst_access_mask = dst_access };
    cmd.pipelineBarrier2(&.{ .memory_barrier_count = 1, .p_memory_barriers = @ptrCast(&b) });
}

pub fn chunkCount(self: *const ChunkBuffers) u32 {
    return self.slots.count();
}
