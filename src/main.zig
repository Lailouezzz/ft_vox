const std = @import("std");

const vk = @import("vulkan");

pub fn main(init: std.process.Init) !void {
    _ = init;
    return;
}

test {
    _ = @import("Camera.zig");
    _ = @import("Sun.zig");
}
