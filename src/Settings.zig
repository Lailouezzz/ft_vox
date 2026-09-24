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
/// Distance fog; disabled by `--no-fog`.
fog: bool = true,

pub const usage =
    \\usage: ft_vox [--seed N] [--radius 4..24] [--shadow-res 512..4096] [--day-length SECONDS] [--no-fog] [--help]
    \\
;

pub const help =
    \\usage: ft_vox [options]
    \\  --seed N             world seed (default 42)
    \\  --radius N           render distance in chunks, 4..24 (default 16)
    \\  --shadow-res N       shadow map size per cascade, power of two in 512..4096 (default 2048)
    \\  --day-length S       seconds per day/night cycle (default 240)
    \\  --no-fog             disable distance fog (the edge of the loaded world becomes visible)
    \\  -h, --help           show this help
    \\
;

pub const ParseError = error{ UnknownOption, MissingValue, InvalidValue, HelpRequested };

/// Parses `args` (without the program name). Value-less flags (`--no-fog`, `-h`/`--help`) may
/// appear anywhere; the remaining, unrecognised arguments must come in `--option value` pairs.
pub fn parse(args: []const []const u8) ParseError!Settings {
    var s: Settings = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const name = args[i];
        if (std.mem.eql(u8, name, "--no-fog")) {
            s.fog = false;
            continue;
        }
        if (std.mem.eql(u8, name, "--help") or std.mem.eql(u8, name, "-h")) return error.HelpRequested;

        if (i + 1 >= args.len) return if (isKnown(name)) error.MissingValue else error.UnknownOption;
        const value = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, name, "--seed")) {
            s.seed = std.fmt.parseInt(u64, value, 10) catch return error.InvalidValue;
        } else if (std.mem.eql(u8, name, "--radius")) {
            s.radius = std.fmt.parseInt(u16, value, 10) catch return error.InvalidValue;
            // Cap at 24 to keep the GPU chunk-meta/quad buffers within their fixed capacity.
            if (s.radius < 4 or s.radius > 24) return error.InvalidValue;
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
    try testing.expectError(error.InvalidValue, parse(&.{ "--radius", "25" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--shadow-res", "1000" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--seed", "-1" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--day-length", "inf" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--day-length", "-inf" }));
    try testing.expectError(error.InvalidValue, parse(&.{ "--day-length", "nan" }));
}

test "fog defaults to true" {
    try testing.expectEqual(true, (try parse(&.{})).fog);
}

test "--no-fog disables fog" {
    try testing.expectEqual(false, (try parse(&.{"--no-fog"})).fog);
}

test "--no-fog mixed with a value option" {
    {
        const s = try parse(&.{ "--no-fog", "--radius", "8" });
        try testing.expectEqual(false, s.fog);
        try testing.expectEqual(@as(u16, 8), s.radius);
    }
    {
        const s = try parse(&.{ "--radius", "8", "--no-fog" });
        try testing.expectEqual(false, s.fog);
        try testing.expectEqual(@as(u16, 8), s.radius);
    }
}

test "--help and -h request help" {
    try testing.expectError(error.HelpRequested, parse(&.{"--help"}));
    try testing.expectError(error.HelpRequested, parse(&.{"-h"}));
    try testing.expectError(error.HelpRequested, parse(&.{ "--radius", "8", "--help" }));
}

test "known option as last argument still needs a value" {
    try testing.expectError(error.MissingValue, parse(&.{ "--radius", "8", "--seed" }));
}

test "help text mentions each default" {
    const defaults: Settings = .{};
    var buf: [32]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, help, std.fmt.bufPrint(&buf, "{d}", .{defaults.seed}) catch unreachable) != null);
    try testing.expect(std.mem.indexOf(u8, help, std.fmt.bufPrint(&buf, "{d}", .{defaults.radius}) catch unreachable) != null);
    try testing.expect(std.mem.indexOf(u8, help, std.fmt.bufPrint(&buf, "{d}", .{defaults.shadow_resolution}) catch unreachable) != null);
    try testing.expect(std.mem.indexOf(u8, help, std.fmt.bufPrint(&buf, "{d}", .{defaults.day_length}) catch unreachable) != null);
}
