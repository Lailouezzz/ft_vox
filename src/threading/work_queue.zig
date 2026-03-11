const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

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
        not_empty: std.Io.Condition,

        pub fn init() Self {
            return .{
                .deque = .empty,
                .mutex = .init,
                .closed = false,
                .not_empty = .init,
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.deque.deinit(allocator);
            self.* = undefined;
        }

        pub fn push(self: *Self, allocator: Allocator, io: Io, item: T) QueueError!void {
            try self.mutex.lock(io);
            defer self.mutex.unlock(io);

            if (self.closed) return QueueError.Closed;

            try self.deque.pushBack(allocator, item);

            self.not_empty.signal(io);
        }

        // Non blocking pop
        pub fn pop(self: *Self, io: Io) QueueError!?T {
            if (self.closed) return QueueError.Closed;
            if (!self.mutex.tryLock()) return null;
            defer self.mutex.unlock(io);
            return self.deque.popFront();
        }

        // Return null if WorkQueue closed
        pub fn waitPop(self: *Self, io: Io) QueueError!T {
            try self.mutex.lock(io);
            defer self.mutex.unlock(io);
            while (self.deque.len == 0) {
                if (self.closed) return QueueError.Closed;
                self.not_empty.waitUncancelable(io, &self.mutex);
            }
            return self.deque.popFront() orelse unreachable;
        }

        pub fn close(self: *Self, io: Io) void {
            self.mutex.lockUncancelable(io);
            self.closed = true;
            self.mutex.unlock(io);
            self.not_empty.broadcast(io);
        }

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

const testing = std.testing;

test "1 thread 2 items" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue = WorkQueue(u64).init();
    defer queue.deinit(testing.allocator);

    try queue.push(allocator, io, 42);
    try queue.push(allocator, io, 43);
    try testing.expectEqual(42, queue.pop(io));
    try testing.expectEqual(43, queue.pop(io));
    try testing.expectEqual(null, queue.pop(io));
}

test "drain" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue = WorkQueue(u64).init();
    defer queue.deinit(allocator);

    try queue.push(allocator, io, 42);
    queue.drain(io);
    try testing.expectEqual(null, queue.pop(io));
}

test "1 thread 0 items" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue = WorkQueue(u64).init();
    defer queue.deinit(allocator);

    try testing.expectEqual(null, queue.pop(io));
}

test "multi-threaded concurent push pop 1:1" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue = WorkQueue(u64).init();
    defer queue.deinit(allocator);
    var poped_count = std.atomic.Value(usize).init(0);

    const n_producers = 4;
    const n_consumers = 4;
    const items_per_producer = 100000;

    var producers: [n_producers]Io.Future(QueueError!void) = undefined;
    for (&producers, 0..n_producers) |*p, k| {
        p.* = try io.concurrent(struct {
            fn run(_queue: *WorkQueue(u64), id: usize) QueueError!void {
                for (0..items_per_producer) |j| {
                    _queue.push(allocator, io, id * items_per_producer + j) catch |err| switch (err) {
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
            fn run(q: *WorkQueue(u64), _poped_count: *std.atomic.Value(usize)) QueueError!void {
                while (true) {
                    _ = q.waitPop(io) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                    _ = _poped_count.fetchAdd(1, .monotonic);
                }
            }
        }.run, .{ &queue, &poped_count });
    }

    for (&producers) |*p| {
        try p.await(io);
    }
    queue.close(io);
    for (&consumers) |*c| {
        try c.await(io);
    }
    try testing.expectEqual(items_per_producer * n_producers, poped_count.load(.acquire));
}

test "multi-threaded concurent push pop 1:1 cancel" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue = WorkQueue(u64).init();
    defer queue.deinit(allocator);
    var poped_count = std.atomic.Value(usize).init(0);

    const n_producers = 4;
    const n_consumers = 4;

    var producers: [n_producers]Io.Future(QueueError!void) = undefined;
    for (&producers, 0..n_producers) |*p, k| {
        p.* = try io.concurrent(struct {
            fn run(_queue: *WorkQueue(u64), id: usize) QueueError!void {
                for (0..std.math.maxInt(usize)) |j| {
                    _queue.push(allocator, io, id + j) catch |err| switch (err) {
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
            fn run(q: *WorkQueue(u64), _poped_count: *std.atomic.Value(usize)) QueueError!void {
                while (true) {
                    _ = q.waitPop(io) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                    _ = _poped_count.fetchAdd(1, .monotonic);
                }
            }
        }.run, .{ &queue, &poped_count });
    }

    try io.sleep(.fromMilliseconds(100), .awake);

    for (&producers) |*p| {
        try testing.expectError(error.Canceled, p.cancel(io));
    }
    queue.close(io);
    for (&consumers) |*c| {
        try c.await(io);
    }
}

test "multi-threaded concurent push pop 1:10" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue = WorkQueue(u64).init();
    defer queue.deinit(allocator);
    var poped_count = std.atomic.Value(usize).init(0);

    const n_producers = 1;
    const n_consumers = 10;
    const items_per_producer = 100000;

    var producers: [n_producers]Io.Future(QueueError!void) = undefined;
    for (&producers, 0..n_producers) |*p, k| {
        p.* = try io.concurrent(struct {
            fn run(_queue: *WorkQueue(u64), id: usize) QueueError!void {
                for (0..items_per_producer) |j| {
                    _queue.push(allocator, io, id * items_per_producer + j) catch |err| switch (err) {
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
            fn run(q: *WorkQueue(u64), _poped_count: *std.atomic.Value(usize)) QueueError!void {
                while (true) {
                    _ = q.waitPop(io) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                    _ = _poped_count.fetchAdd(1, .monotonic);
                }
            }
        }.run, .{ &queue, &poped_count });
    }

    for (&producers) |*p| {
        try p.await(io);
    }
    queue.close(io);
    for (&consumers) |*c| {
        try c.await(io);
    }
    try testing.expectEqual(items_per_producer * n_producers, poped_count.load(.acquire));
}

test "multi-threaded items concurent push drain" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue = WorkQueue(u64).init();
    defer queue.deinit(allocator);
    var pushed_count = std.atomic.Value(usize).init(0);

    const n_producers = 4;
    const items_per_producer = 100000;

    var producers: [n_producers]Io.Future(QueueError!void) = undefined;
    for (&producers, 0..n_producers) |*p, k| {
        p.* = try io.concurrent(struct {
            fn run(_queue: *WorkQueue(u64), id: usize, _pushed_count: *std.atomic.Value(usize)) QueueError!void {
                for (0..items_per_producer) |j| {
                    _queue.push(allocator, io, id * items_per_producer + j) catch |err| switch (err) {
                        error.Closed => return,
                        else => return err,
                    };
                    _ = _pushed_count.fetchAdd(1, .monotonic);
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
    var queue = WorkQueue(u64).init();
    defer queue.deinit(allocator);

    try queue.push(allocator, io, 42);
    queue.close(io);
    try testing.expectEqual(42, queue.waitPop(io));
    try testing.expectError(QueueError.Closed, queue.waitPop(io));
}

test "close push" {
    const io = testing.io;
    const allocator = testing.allocator;
    var queue = WorkQueue(u64).init();
    defer queue.deinit(allocator);

    queue.close(io);
    try testing.expectError(QueueError.Closed, queue.push(allocator, io, 32));
    try testing.expectError(QueueError.Closed, queue.waitPop(io));
    try testing.expectError(QueueError.Closed, queue.pop(io));
    try testing.expectError(QueueError.Closed, queue.waitPop(io));
}
