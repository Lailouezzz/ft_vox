const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const testing = std.testing;

const QueueError = @import("work_queue.zig").QueueError;
const WorkQueue = @import("work_queue.zig").WorkQueue;

const log = std.log.scoped(.worker_pool);

pub fn WorkerPool(comptime InType: type, comptime OutType: type) type {
    return struct {
        const Self = @This();
        const TaskFn = *const fn (Allocator, Io, InType) anyerror!?OutType;

        const Worker = struct {
            const State = enum(u8) {
                waiting,
                working,
                shutdown,
                ended,
            };

            future: Io.Future(anyerror!void),
            state: std.atomic.Value(State),
            id: usize,

            pub const empty: Worker = .{
                .future = undefined,
                .state = .init(.waiting),
                .id = undefined,
            };

            pub fn await(self: *Worker, io: Io) !void {
                errdefer |err| log.warn("await worker[{d}] error: {any}", .{ self.id, err });
                log.info("awaiting worker[{d}]", .{self.id});
                try self.future.await(io);
            }

            pub fn cancel(self: *Worker, io: Io) !void {
                errdefer |err| switch (err) {
                    error.Canceled => {},
                    else => log.warn("cancel worker[{d}] error: {any}", .{ self.id, err }),
                };
                log.info("canceling worker[{d}]", .{self.id});
                try self.future.cancel(io);
            }

            fn entry(self: *Worker, allocator: Allocator, io: Io, in_queue: *WorkQueue(InType), out_queue: ?*WorkQueue(OutType), task_fn: TaskFn) anyerror!void {
                log.info("worker[{d}] started", .{self.id});
                defer self.state.store(.ended, .release);
                while (true) {
                    if (self.state.cmpxchgStrong(.working, .waiting, .acq_rel, .acquire)) |actual| switch (actual) {
                        .shutdown => break,
                        .ended, .working => unreachable,
                        .waiting => {},
                    };
                    log.debug("worker[{d}] wait pop", .{self.id});
                    if (in_queue.waitPop(io)) |in| {
                        if (self.state.cmpxchgStrong(.waiting, .working, .acq_rel, .acquire)) |actual| switch (actual) {
                            .shutdown => {
                                return in_queue.pushFront(allocator, io, in);
                            },
                            .working, .waiting, .ended => unreachable,
                        };
                        log.debug("worker[{d}] working", .{self.id});
                        const maybe_out = task_fn(allocator, io, in) catch |err| switch (err) {
                            error.Canceled => return err,
                            else => {
                                log.warn("worker[{d}] task failed: {t}", .{ self.id, err });
                                continue;
                            },
                        };
                        if (maybe_out) |out| {
                            // ponytail: `out` leaks if the push fails (OOM or closed output queue).
                            if (out_queue) |queue|
                                try queue.push(allocator, io, out);
                        }
                    } else |err| switch (err) {
                        error.Closed => break,
                        else => return err,
                    }
                }
            }
        };

        in_queue: *WorkQueue(InType),
        out_queue: ?*WorkQueue(OutType),
        worker_count: usize,
        workers: std.ArrayList(*Worker),
        ending_workers: std.ArrayList(*Worker),
        task_fn: TaskFn,

        pub fn init(
            in_queue: *WorkQueue(InType),
            out_queue: ?*WorkQueue(OutType),
            worker_count: usize,
            task_fn: TaskFn,
        ) Self {
            return .{
                .in_queue = in_queue,
                .out_queue = out_queue,
                .worker_count = worker_count,
                .workers = .empty,
                .ending_workers = .empty,
                .task_fn = task_fn,
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator, io: Io) void {
            self.cancel(allocator, io);
            self.workers.deinit(allocator);
            self.ending_workers.deinit(allocator);
            self.* = undefined;
        }

        pub fn start(self: *Self, allocator: Allocator, io: Io) !void {
            try self.update(allocator, io);
        }

        /// Blocking: closes the input queue and waits for workers to finish pending items.
        pub fn shutdown(self: *Self, allocator: Allocator, io: Io) void {
            self.in_queue.close(io);
            for (self.workers.items) |worker| {
                worker.await(io) catch {};
                allocator.destroy(worker);
            }
            self.workers.clearAndFree(allocator);
            for (self.ending_workers.items) |worker| {
                worker.await(io) catch {};
                allocator.destroy(worker);
            }
            self.ending_workers.clearAndFree(allocator);
        }

        /// Blocking: closes and drains the input queue, then cancels workers.
        pub fn cancel(self: *Self, allocator: Allocator, io: Io) void {
            self.in_queue.close(io);
            self.in_queue.drain(io);
            for (self.workers.items) |worker| {
                worker.cancel(io) catch {};
                allocator.destroy(worker);
            }
            self.workers.clearAndFree(allocator);
            for (self.ending_workers.items) |worker| {
                worker.cancel(io) catch {};
                allocator.destroy(worker);
            }
            self.ending_workers.clearAndFree(allocator);
        }

        pub fn setWorkerCount(self: *Self, worker_count: usize) void {
            self.worker_count = worker_count;
        }

        /// Call regularly to spawn or retire workers until `worker_count` is matched.
        pub fn update(self: *Self, allocator: Allocator, io: Io) !void {
            if (self.worker_count > self.workers.items.len) {
                try self.workers.ensureTotalCapacity(allocator, self.worker_count);
                for (self.workers.items.len..self.worker_count) |_| {
                    try self.spawnOne(allocator, io);
                }
            } else if (self.worker_count < self.workers.items.len) {
                for (self.worker_count..self.workers.items.len) |_| {
                    const worker = self.workers.pop().?;
                    if (worker.state.swap(.shutdown, .acq_rel) == .ended) {
                        log.warn("worker[{d}] was ended with: {any}", .{ worker.id, worker.cancel(io) });
                        allocator.destroy(worker);
                        continue;
                    }
                    log.info("shutting down worker[{d}]", .{worker.id});
                    try self.ending_workers.append(allocator, worker);
                }
            }
            self.cleanEnded(allocator, io);
        }

        fn spawnOne(self: *Self, allocator: Allocator, io: Io) !void {
            const id = self.workers.items.len;
            const worker = try allocator.create(Worker);
            worker.* = .empty;
            worker.id = id;
            self.workers.appendAssumeCapacity(worker);
            worker.future = try io.concurrent(Worker.entry, .{ worker, allocator, io, self.in_queue, self.out_queue, self.task_fn });
        }

        fn cleanEnded(self: *Self, allocator: Allocator, io: Io) void {
            var k: usize = 0;
            while (k < self.ending_workers.items.len) {
                const worker = self.ending_workers.items[k];
                if (worker.state.load(.acquire) == .ended) {
                    worker.await(io) catch |err| log.warn("worker[{d}] ended with error: {any}", .{ worker.id, err });
                    log.debug("worker[{d}] removed from ending workers", .{worker.id});
                    allocator.destroy(worker);
                    _ = self.ending_workers.swapRemove(k);
                } else {
                    log.debug("worker[{d}] waiting in ending workers", .{worker.id});
                    k += 1;
                }
            }
        }
    };
}

// ---
// Tests
// ---

fn adder(_: Allocator, _: Io, in: u64) anyerror!?u64 {
    return in + 1;
}

fn subber(_: Allocator, _: Io, in: u64) anyerror!?u64 {
    return in - 1;
}

fn hardWorkCancelable(_: Allocator, io: Io, in: u64) anyerror!?u64 {
    try io.sleep(.fromMilliseconds(1), .awake);
    return in + 1;
}

fn oneMsJobUncancelable(_: Allocator, io: Io, in: u64) anyerror!?u64 {
    const old_cancel_protect = io.swapCancelProtection(.blocked);
    defer _ = io.swapCancelProtection(old_cancel_protect);
    io.sleep(.fromMilliseconds(1), .awake) catch unreachable;
    return in + 1;
}

test "0 producer 1 consumer" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, adder);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
}

test "0 producer 1 consumer shutdown" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, adder);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    pool.shutdown(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
}

test "0 producer 1 consumer cancel" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, adder);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    pool.cancel(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
}

test "0 producer 5 consumer" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 5, adder);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
}

test "0 producer 5 consumer shutdown" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 5, adder);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    pool.shutdown(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
}

test "0 producer 5 consumer cancel" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 5, adder);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    pool.cancel(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
}

test "0 producer 1 consumer adder cancel (default queue 50 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    for (0..50) |k| {
        try in_queue.push(allocator, io, k);
    }
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, adder);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake);
    pool.cancel(allocator, io);
}

test "0 producer 1 consumer one ms job cancelable cancel (default queue 50 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    for (0..50) |k| {
        try in_queue.push(allocator, io, k);
    }
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, hardWorkCancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake);
    pool.cancel(allocator, io);
}

test "0 producer 1 consumer one ms job uncancelable cancel (default queue 50 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    for (0..50) |k| {
        try in_queue.push(allocator, io, k);
    }
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, oneMsJobUncancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake);
    pool.cancel(allocator, io);
}

test "0 producer 1 consumer adder shutdown (default queue 50 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    for (0..50) |k| {
        try in_queue.push(allocator, io, k);
    }
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, adder);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake);
    pool.shutdown(allocator, io);
    try testing.expectEqual(50, out_queue.deque.len);
}

test "0 producer 1 consumer one ms job cancelable shutdown (default queue 50 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    for (0..50) |k| {
        try in_queue.push(allocator, io, k);
    }
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, hardWorkCancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake);
    pool.shutdown(allocator, io);
    try testing.expectEqual(50, out_queue.deque.len);
}

test "0 producer 1 consumer one ms job uncancelable shutdown (default queue 50 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    for (0..50) |k| {
        try in_queue.push(allocator, io, k);
    }
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, oneMsJobUncancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake);
    pool.shutdown(allocator, io);
    try testing.expectEqual(50, out_queue.deque.len);
}

test "0 producer 10 consumer one ms job uncancelable shutdown (default queue 500 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    for (0..500) |k| {
        try in_queue.push(allocator, io, k);
    }
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 10, oneMsJobUncancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
    pool.shutdown(allocator, io);
    try testing.expectEqual(500, out_queue.deque.len);
}

test "0 producer 1->10 consumer one ms job uncancelable shutdown (default queue 200 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    for (0..200) |k| {
        try in_queue.push(allocator, io, k);
    }
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, oneMsJobUncancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake); // 10 job done theoretically
    pool.setWorkerCount(10);
    try pool.update(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake); // 100 job done theoretically
    pool.setWorkerCount(1);
    try pool.update(allocator, io);
    try io.sleep(.fromMilliseconds(2), .awake); // 2 job done theoretically
    try pool.update(allocator, io); // Clean up finished job
    pool.shutdown(allocator, io);
    try testing.expectEqual(200, out_queue.deque.len);
}

test "0 producer 1->10 consumer one ms job uncancelable shutdown with ending workers (default queue 200 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    for (0..200) |k| {
        try in_queue.push(allocator, io, k);
    }
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, oneMsJobUncancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake); // 10 job done theoretically
    pool.setWorkerCount(10);
    try pool.update(allocator, io);
    try io.sleep(.fromMilliseconds(10), .awake); // 100 job done theoretically
    pool.setWorkerCount(1);
    try pool.update(allocator, io);
    pool.shutdown(allocator, io);
    try testing.expectEqual(200, out_queue.deque.len);
}

test "0 producer 1->0 consumer adder shutdown with ending workers (default queue 0->1 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, oneMsJobUncancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
    pool.setWorkerCount(0);
    try pool.update(allocator, io); // 1 ending worker
    try io.sleep(.fromMilliseconds(1), .awake);
    try in_queue.push(allocator, io, 2);
    try io.sleep(.fromMilliseconds(1), .awake);
    pool.shutdown(allocator, io);
    try testing.expectEqual(1, in_queue.deque.len);
}

test "0 producer 1->0 consumer adder (default queue 0->1 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, oneMsJobUncancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
    pool.setWorkerCount(0);
    try pool.update(allocator, io); // 1 ending worker
    try io.sleep(.fromMilliseconds(1), .awake);
    try in_queue.push(allocator, io, 2);
    try io.sleep(.fromMilliseconds(1), .awake);
    try pool.update(allocator, io); // 1 ending worker cleared
    try io.sleep(.fromMilliseconds(1), .awake);
    pool.shutdown(allocator, io);
    try testing.expectEqual(1, in_queue.deque.len);
}

test "1 producer 10 consumer one ms job uncancelable shutdown (500 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 10, oneMsJobUncancelable);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
    for (0..500) |k| {
        try in_queue.push(allocator, io, k);
    }
    var sum: usize = 0;
    while (true) {
        sum += try out_queue.waitPop(io);
        if (sum == ((500 * 501) / 2))
            break;
    }
    try testing.expectEqual(0, out_queue.deque.len);
}

test "10 producer 10 consumer one ms job uncancelable shutdown (500 items)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var added_queue: WorkQueue(u64) = .empty;
    defer added_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    for (0..500) |k| {
        try in_queue.push(allocator, io, k);
    }
    var adder_pool = WorkerPool(u64, u64).init(&in_queue, &added_queue, 10, oneMsJobUncancelable);
    var subber_pool = WorkerPool(u64, u64).init(&added_queue, &out_queue, 10, subber);
    defer subber_pool.deinit(allocator, io);
    defer adder_pool.deinit(allocator, io);
    try subber_pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
    try adder_pool.start(allocator, io);
    try io.sleep(.fromMilliseconds(1), .awake);
    adder_pool.shutdown(allocator, io);
    subber_pool.shutdown(allocator, io);
    try testing.expectEqual(500, out_queue.deque.len);
    try testing.expectEqual(0, added_queue.deque.len);
    try testing.expectEqual(0, in_queue.deque.len);
    var sum: usize = 0;
    while (true) {
        sum += try out_queue.waitPop(io);
        if (sum == ((499 * 500) / 2))
            break;
    }
}

fn failOnZero(_: Allocator, _: Io, in: u64) anyerror!?u64 {
    if (in == 0) return error.Boom;
    return in;
}

test "failing task does not kill its worker" {
    const io = testing.io;
    const allocator = testing.allocator;
    var in_queue: WorkQueue(u64) = .empty;
    defer in_queue.deinit(allocator);
    var out_queue: WorkQueue(u64) = .empty;
    defer out_queue.deinit(allocator);
    for ([_]u64{ 0, 1, 2 }) |k| try in_queue.push(allocator, io, k);
    var pool = WorkerPool(u64, u64).init(&in_queue, &out_queue, 1, failOnZero);
    defer pool.deinit(allocator, io);
    try pool.start(allocator, io);
    pool.shutdown(allocator, io);
    try testing.expectEqual(2, out_queue.deque.len);
}
