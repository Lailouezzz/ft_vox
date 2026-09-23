const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const testing = std.testing;

pub const QueueError = error{
    Closed,
    OutOfMemory,
} || Io.Cancelable;

pub fn WorkQueue(comptime T: type) type {
    return struct {
        const Self = @This();

        deque: std.Deque(T),
        mutex: Io.Mutex,
        closed: bool,
        not_empty: Io.Condition,

        pub const empty: Self = .{
            .deque = .empty,
            .mutex = .init,
            .closed = false,
            .not_empty = .init,
        };

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.deque.deinit(allocator);
            self.* = undefined;
        }

        pub fn push(self: *Self, allocator: Allocator, io: Io, item: T) QueueError!void {
            try self.mutex.lock(io);
            defer self.mutex.unlock(io);

            if (self.closed) return error.Closed;

            try self.deque.pushBack(allocator, item);

            self.not_empty.signal(io);
        }

        pub fn pushFront(self: *Self, allocator: Allocator, io: Io, item: T) QueueError!void {
            try self.mutex.lock(io);
            defer self.mutex.unlock(io);

            if (self.closed) return error.Closed;

            try self.deque.pushFront(allocator, item);

            self.not_empty.signal(io);
        }

        /// Non-blocking: returns null if the queue is empty or the lock is contended.
        /// Items pushed before `close` are still returned; error.Closed once closed and empty.
        pub fn pop(self: *Self, io: Io) QueueError!?T {
            if (!self.mutex.tryLock()) return null;
            defer self.mutex.unlock(io);
            if (self.deque.popFront()) |item| return item;
            if (self.closed) return error.Closed;
            return null;
        }

        /// Blocks until an item is available. Returns error.Closed once closed and empty.
        pub fn waitPop(self: *Self, io: Io) QueueError!T {
            try self.mutex.lock(io);
            defer self.mutex.unlock(io);
            while (self.deque.len == 0) {
                if (self.closed) return error.Closed;
                try self.not_empty.wait(io, &self.mutex);
            }
            return self.deque.popFront().?;
        }

        pub fn close(self: *Self, io: Io) void {
            self.mutex.lockUncancelable(io);
            self.closed = true;
            self.mutex.unlock(io);
            self.not_empty.broadcast(io);
        }

        /// Discards every queued item without freeing it: if items own memory,
        /// pop them one by one instead.
        pub fn drain(self: *Self, io: Io) void {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            while (self.deque.len != 0) _ = self.deque.popFront();
        }
    };
}

// ---
// Tests
// ---

test "1 thread 2 items" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);

    try queue.push(allocator, io, 42);
    try queue.push(allocator, io, 43);
    try testing.expectEqual(42, queue.pop(io));
    try testing.expectEqual(43, queue.pop(io));
    try testing.expectEqual(null, queue.pop(io));
}

test "drain" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);

    try queue.push(allocator, io, 42);
    queue.drain(io);
    try testing.expectEqual(null, queue.pop(io));
}

test "1 thread 0 items" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);

    try testing.expectEqual(null, queue.pop(io));
}

test "multi-threaded concurrent push pop 1:1" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);
    var popped_count = std.atomic.Value(usize).init(0);

    const n_producers = 4;
    const n_consumers = 4;
    const items_per_producer = 100000;

    var producers: [n_producers]Io.Future(QueueError!void) = undefined;
    for (&producers, 0..n_producers) |*p, k| {
        p.* = try io.concurrent(struct {
            fn run(q: *WorkQueue(u64), id: usize) QueueError!void {
                for (0..items_per_producer) |j| {
                    q.push(allocator, io, id * items_per_producer + j) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                }
            }
        }.run, .{ &queue, k });
    }

    var consumers: [n_consumers]Io.Future(QueueError!void) = undefined;
    for (&consumers) |*c| {
        c.* = try io.concurrent(struct {
            fn run(q: *WorkQueue(u64), count: *std.atomic.Value(usize)) QueueError!void {
                while (true) {
                    _ = q.waitPop(io) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                    _ = count.fetchAdd(1, .monotonic);
                }
            }
        }.run, .{ &queue, &popped_count });
    }

    for (&producers) |*p| {
        try p.await(io);
    }
    queue.close(io);
    for (&consumers) |*c| {
        try c.await(io);
    }
    try testing.expectEqual(items_per_producer * n_producers, popped_count.load(.acquire));
}

test "multi-threaded concurrent push pop 1:1 cancel" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);
    var push_pop_count = std.atomic.Value(usize).init(0);

    const n_producers = 4;
    const n_consumers = 4;

    var producers: [n_producers]Io.Future(QueueError!void) = undefined;
    for (&producers, 0..n_producers) |*p, k| {
        p.* = try io.concurrent(struct {
            fn run(q: *WorkQueue(u64), count: *std.atomic.Value(usize), id: usize) QueueError!void {
                for (0..std.math.maxInt(usize)) |j| {
                    q.push(allocator, io, id + j) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                    _ = count.fetchAdd(1, .acq_rel);
                }
            }
        }.run, .{ &queue, &push_pop_count, k });
    }

    var consumers: [n_consumers]Io.Future(QueueError!void) = undefined;
    for (&consumers) |*c| {
        c.* = try io.concurrent(struct {
            fn run(q: *WorkQueue(u64), count: *std.atomic.Value(usize)) QueueError!void {
                while (true) {
                    _ = q.waitPop(io) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                    _ = count.fetchSub(1, .acq_rel);
                }
            }
        }.run, .{ &queue, &push_pop_count });
    }

    try io.sleep(.fromMilliseconds(100), .awake);

    for (&producers) |*p| {
        try testing.expectError(error.Canceled, p.cancel(io));
    }
    queue.close(io);
    for (&consumers) |*c| {
        try c.await(io);
    }
    try testing.expectEqual(0, push_pop_count.load(.monotonic));
}

test "multi-threaded concurrent push pop 1:10" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);
    var popped_count = std.atomic.Value(usize).init(0);

    const n_producers = 1;
    const n_consumers = 10;
    const items_per_producer = 100000;

    var producers: [n_producers]Io.Future(QueueError!void) = undefined;
    for (&producers, 0..n_producers) |*p, k| {
        p.* = try io.concurrent(struct {
            fn run(q: *WorkQueue(u64), id: usize) QueueError!void {
                for (0..items_per_producer) |j| {
                    q.push(allocator, io, id * items_per_producer + j) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                }
            }
        }.run, .{ &queue, k });
    }

    var consumers: [n_consumers]Io.Future(QueueError!void) = undefined;
    for (&consumers) |*c| {
        c.* = try io.concurrent(struct {
            fn run(q: *WorkQueue(u64), count: *std.atomic.Value(usize)) QueueError!void {
                while (true) {
                    _ = q.waitPop(io) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                    _ = count.fetchAdd(1, .monotonic);
                }
            }
        }.run, .{ &queue, &popped_count });
    }

    for (&producers) |*p| {
        try p.await(io);
    }
    queue.close(io);
    for (&consumers) |*c| {
        try c.await(io);
    }
    try testing.expectEqual(items_per_producer * n_producers, popped_count.load(.acquire));
}

test "multi-threaded items concurrent push drain" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);
    var pushed_count = std.atomic.Value(usize).init(0);

    const n_producers = 4;
    const items_per_producer = 100000;

    var producers: [n_producers]Io.Future(QueueError!void) = undefined;
    for (&producers, 0..n_producers) |*p, k| {
        p.* = try io.concurrent(struct {
            fn run(q: *WorkQueue(u64), id: usize, count: *std.atomic.Value(usize)) QueueError!void {
                for (0..items_per_producer) |j| {
                    q.push(allocator, io, id * items_per_producer + j) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                    _ = count.fetchAdd(1, .monotonic);
                }
            }
        }.run, .{ &queue, k, &pushed_count });
    }
    try io.sleep(.fromMilliseconds(100), .awake);
    queue.drain(io);
    for (&producers) |*p| {
        try p.await(io);
    }
    try testing.expect(queue.deque.len != pushed_count.load(.acquire));
}

test "close pop" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);

    try queue.push(allocator, io, 42);
    queue.close(io);
    try testing.expectEqual(42, queue.waitPop(io));
    try testing.expectError(QueueError.Closed, queue.waitPop(io));
}

test "close push" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);

    queue.close(io);
    try testing.expectError(QueueError.Closed, queue.push(allocator, io, 32));
    try testing.expectError(QueueError.Closed, queue.waitPop(io));
    try testing.expectError(QueueError.Closed, queue.pop(io));
    try testing.expectError(QueueError.Closed, queue.waitPop(io));
}

test "close keeps queued items" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue: WorkQueue(u64) = .empty;
    defer queue.deinit(allocator);

    try queue.push(allocator, io, 1);
    try queue.push(allocator, io, 2);
    queue.close(io);
    try testing.expectEqual(1, queue.pop(io));
    try testing.expectEqual(2, queue.pop(io));
    try testing.expectError(QueueError.Closed, queue.pop(io));
}
