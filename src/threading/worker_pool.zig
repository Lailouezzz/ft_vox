const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const WorkQueue = @import("work_queue.zig").WorkQueue;
const QueueError = @import("work_queue.zig").QueueError;

pub fn WorkerPool(comptime InType: type, comptime OutType: type) type {
    return struct {
        const Self = @This();
        const TaskFn = fn (Allocator, Io, InType) anyerror!?OutType;

        in_queue: *WorkQueue(InType),
        out_queue: *WorkQueue(OutType),
        nb_threads: usize,
        threads: std.ArrayList(std.Io.Future(anyerror!void)),
        task_fn: TaskFn,

        pub fn init(
            in_queue: *WorkQueue(InType),
            out_queue: *WorkQueue(OutType),
            nb_threads: usize,
            task_fn: TaskFn,
        ) Self {
            return .{
                .in_queue = in_queue,
                .out_queue = out_queue,
                .nb_threads = nb_threads,
                .threads = .empty,
                .task_fn = task_fn,
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator, io: Io) void {
            self.shutdown(io);
            self.in_queue.deinit(allocator);
            self.out_queue.deinit(allocator);
            self.threads.deinit(allocator);
            self.* = undefined;
        }

        pub fn start(self: *Self, allocator: Allocator, io: Io) !void {
            for (0..self.nb_threads) |_| {
                try self.threads.append(allocator, try io.concurrent(threadEntry, .{ self, allocator, io }));
            }
        }

        pub fn shutdown(self: *Self, io: Io) void {
            self.in_queue.close(io);
            for (self.threads.items) |*thread| {
                thread.await(io);
            }
        }

        pub fn cancel(self: *Self, io: Io) void {
            self.in_queue.close(io);
            for (self.threads.items) |*thread| {
                thread.cancel(io) catch {};
            }
        }

        fn threadEntry(self: *Self, allocator: Allocator, io: Io) anyerror!void {
            errdefer std.log.err("Thread error", .{});
            while (true) {
                if (self.in_queue.waitPop(io)) |in| {
                    if (try self.task_fn(in)) |out| {
                        try self.out_queue.push(allocator, io, out);
                    }
                } else |err| switch (err) {
                    error.Closed => return,
                    else => return err,
                }
            }
        }
    };
}

// ---
// Tests
// ---

const testing = std.testing;
