pub const WorkQueue = @import("work_queue.zig").WorkQueue;
pub const WorkerPool = @import("worker_pool.zig").WorkerPool;

test {
    @import("std").testing.log_level = .debug;
    _ = @import("work_queue.zig");
    _ = @import("worker_pool.zig");
}
