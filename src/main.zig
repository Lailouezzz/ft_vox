const std = @import("std");

fn testing(io: std.Io, id: usize) anyerror!bool {
    try io.sleep(.fromSeconds(1), .awake);
    std.log.debug("{d} non", .{id});
    try io.sleep(.fromSeconds(1), .awake);
    std.log.debug("{d} non", .{id});
    try io.sleep(.fromSeconds(1), .awake);
    std.log.debug("{d} OUI", .{id});
    return true;
}

pub fn main(init: std.process.Init) !void {
    var futures: [20]std.Io.Future(anyerror!bool) = undefined;
    for (&futures, 0..) |*future, k| {
        future.* = try init.io.concurrent(testing, .{ init.io, k });
    }
    try init.io.sleep(.fromSeconds(1), .awake);
    std.log.debug("FROM MAIN", .{});
    try init.io.sleep(.fromSeconds(1), .awake);
    std.log.debug("FROM MAIN", .{});
    try init.io.sleep(.fromSeconds(1), .awake);
    std.log.debug("FROM MAIN", .{});
    for (&futures, 0..) |*future, k| {
        if (try future.await(init.io)) {
            if (try future.result) {
                std.log.debug("DONE {d}", .{k});
            }
        }
    }
    return;
}
