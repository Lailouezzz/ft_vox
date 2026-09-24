//! Command-line settings: `ft_vox [--seed N] [--radius N] [--shadow-res N] [--day-length S]`.
//! Defaults suit an integrated GPU.
const std = @import("std");

const Settings = @This();

seed: u64 = 42,
/// Render distance in chunks (horizontal).
radius: u16 = 16,
/// Resolution of each shadow cascade (power of two).
shadow_resolution: u32 = 2048,
/// Seconds for a full day/night cycle.
day_length: f32 = 240,

pub const usage =
    \\usage: ft_vox [--seed N] [--radius 4..32] [--shadow-res 512..4096] [--day-length SECONDS]
    \\
;

pub const ParseError = error{ UnknownOption, MissingValue, InvalidValue };

/// Parses `args` (without the program name).
pub fn parse(args: []const []const u8) ParseError!Settings {
    var s: Settings = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const name = args[i];
        if (i + 1 >= args.len) return if (isKnown(name)) error.MissingValue else error.UnknownOption;
        const value = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, name, "--seed")) {
            s.seed = std.fmt.parseInt(u64, value, 10) catch return error.InvalidValue;
        } else if (std.mem.eql(u8, name, "--radius")) {
            s.radius = std.fmt.parseInt(u16, value, 10) catch return error.InvalidValue;
            if (s.radius < 4 or s.radius > 32) return error.InvalidValue;
        } else if (std.mem.eql(u8, name, "--shadow-res")) {
            s.shadow_resolution = std.fmt.parseInt(u32, value, 10) catch return error.InvalidValue;
            if (s.shadow_resolution < 512 or s.shadow_resolution > 4096 or !std.math.isPowerOfTwo(s.shadow_resolution)) return error.InvalidValue;
        } else if (std.mem.eql(u8, name, "--day-length")) {
            s.day_length = std.fmt.parseFloat(f32, value) catch return error.InvalidValue;
            if (!std.math.isFinite(s.day_length) or s.day_length <= 0) return error.InvalidValue;
        } else return error.UnknownOption;
    }
    return s;
}

fn isKnown(name: []const u8) bool {
    for ([_][]const u8{ "--seed", "--radius", "--shadow-res", "--day-length" }) |k| if (std.mem.eql(u8, name, k)) return true;
    return false;
}

const testing = std.testing;

test "defaults" {
    try testing.expectEqual(Settings{}, try parse(&.{}));
}

test "all options" {
    const s = try parse(&.{ "--seed", "7", "--radius", "8", "--shadow-res", "1024", "--day-length", "60" });
    try testing.expectEqual(@as(u64, 7), s.seed);
    try testing.expectEqual(@as(u16, 8), s.radius);
    try testing.expectEqual(@as(u32, 1024), s.shadow_resolution);
    try testing.expectEqual(@as(f32, 60), s.day_length);
}

test "errors" {
    try testing.expectError(error.UnknownOption, parse(&.{"--nope"}));
    try testing.expectError(error.MissingValue, parse(&.{"--seed"}));
    try testing.expectError(error.InvalidValue, parse(&.{ "--radius", "99" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--shadow-res", "1000" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--seed", "-1" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--day-length", "inf" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--day-length", "-inf" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--day-length", "nan" }));
}
