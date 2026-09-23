//! A buffer with its own memory allocation, device address, and a persistent
//! mapping when host visible.
const std = @import("std");
const vk = @import("vulkan");
const Context = @import("Context.zig");

const Buffer = @This();

handle: vk.Buffer,
memory: vk.DeviceMemory,
size: vk.DeviceSize,
address: vk.DeviceAddress,
mapped: ?[*]u8,

pub fn init(ctx: *const Context, size: vk.DeviceSize, usage: vk.BufferUsageFlags, host_visible: bool) !Buffer {
    var u = usage;
    u.shader_device_address_bit = true;
    const handle = try ctx.device.createBuffer(&.{ .size = size, .usage = u, .sharing_mode = .exclusive }, null);
    errdefer ctx.device.destroyBuffer(handle, null);
    const req = ctx.device.getBufferMemoryRequirements(handle);
    const flags: vk.MemoryPropertyFlags = if (host_visible)
        .{ .host_visible_bit = true, .host_coherent_bit = true }
    else
        .{ .device_local_bit = true };
    const alloc_flags: vk.MemoryAllocateFlagsInfo = .{ .flags = .{ .device_address_bit = true }, .device_mask = 0 };
    const memory = try ctx.device.allocateMemory(&.{
        .p_next = &alloc_flags,
        .allocation_size = req.size,
        .memory_type_index = try ctx.findMemoryType(req.memory_type_bits, flags),
    }, null);
    errdefer ctx.device.freeMemory(memory, null);
    try ctx.device.bindBufferMemory(handle, memory, 0);
    const mapped: ?[*]u8 = if (host_visible) @ptrCast(try ctx.device.mapMemory(memory, 0, vk.WHOLE_SIZE, .{})) else null;
    return .{
        .handle = handle,
        .memory = memory,
        .size = size,
        .address = ctx.device.getBufferDeviceAddress(&.{ .buffer = handle }),
        .mapped = mapped,
    };
}

pub fn deinit(self: *Buffer, ctx: *const Context) void {
    ctx.device.destroyBuffer(self.handle, null);
    ctx.device.freeMemory(self.memory, null);
    self.* = undefined;
}

/// Typed view of a host-visible buffer.
pub fn slice(self: *const Buffer, comptime T: type) []T {
    const bytes = self.mapped.?[0..self.size];
    return @alignCast(std.mem.bytesAsSlice(T, bytes));
}
