//! Day/night cycle: sun direction and light colors from the time of day.
const std = @import("std");

const Sun = @This();

/// Seconds for a full day.
day_length: f32 = 240,
/// 0 = midnight, 0.25 = sunrise, 0.5 = noon, 0.75 = sunset.
time: f32 = 0.3,

pub const Lighting = struct {
    dir: [3]f32, // towards the light
    color: [3]f32,
    ambient: [3]f32,
};

pub fn advance(s: *Sun, dt: f32) void {
    s.time = @mod(s.time + dt / s.day_length, 1);
}

/// Direction towards the sun: rises in +X, sets in -X, tilted towards -Z.
pub fn direction(s: Sun) [3]f32 {
    const a = (s.time - 0.25) * 2 * std.math.pi;
    const v = [3]f32{ @cos(a), @sin(a), -0.35 };
    const len = @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return .{ v[0] / len, v[1] / len, v[2] / len };
}

pub fn lighting(s: Sun) Lighting {
    const dir = s.direction();
    const day = std.math.clamp(dir[1] * 4 + 0.3, 0, 1);
    // Warmer and dimmer near the horizon.
    const warm = std.math.clamp(1 - dir[1] * 3, 0, 1);
    return .{
        .dir = if (dir[1] > -0.1) dir else .{ -dir[0], -dir[1], -dir[2] }, // moon opposite the sun at night
        .color = if (dir[1] > -0.1)
            .{ 2.2 * day, (2.0 - 0.6 * warm) * day, (1.8 - 1.0 * warm) * day }
        else
            .{ 0.05, 0.07, 0.12 },
        .ambient = .{ 0.05 + 0.25 * day, 0.06 + 0.30 * day, 0.10 + 0.40 * day },
    };
}

test "noon sun is overhead, midnight sun is below" {
    try std.testing.expect((Sun{ .time = 0.5 }).direction()[1] > 0.9);
    try std.testing.expect((Sun{ .time = 0 }).direction()[1] < -0.9);
}

test "time wraps around" {
    var s: Sun = .{ .day_length = 10, .time = 0.95 };
    s.advance(1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), s.time, 1e-5);
}
