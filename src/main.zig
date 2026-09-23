const std = @import("std");
const glfw = @import("zglfw");
const vk = @import("vulkan");
const Context = @import("render/Context.zig");
const Renderer = @import("render/Renderer.zig");
const Camera = @import("Camera.zig");
const Sun = @import("Sun.zig");

const walk_speed: f32 = 12; // blocks per second
const sprint_factor: f32 = 6;
const mouse_sensitivity: f32 = 0.0025;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    try glfw.init();
    defer glfw.terminate();
    glfw.windowHint(.client_api, .no_api);
    const window = try glfw.Window.create(1280, 720, "ft_vox", null, null);
    defer window.destroy();
    try glfw.setInputMode(window, .cursor, .disabled);

    var ctx: Context = try .init(gpa, window);
    defer ctx.deinit(gpa);
    var renderer: Renderer = try .init(gpa, &ctx, framebufferExtent(window));
    defer renderer.deinit();

    var camera: Camera = .{};
    var sun: Sun = .{};
    var last_cursor = window.getCursorPos();
    var last_time = glfw.getTime();

    while (!window.shouldClose()) {
        glfw.pollEvents();
        const now = glfw.getTime();
        const dt: f32 = @floatCast(now - last_time);
        last_time = now;
        if (window.getKey(.escape) == .press) window.setShouldClose(true);

        // Mouse look.
        const cursor = window.getCursorPos();
        camera.yaw -= @as(f32, @floatCast(cursor[0] - last_cursor[0])) * mouse_sensitivity;
        camera.pitch -= @as(f32, @floatCast(cursor[1] - last_cursor[1])) * mouse_sensitivity;
        camera.pitch = std.math.clamp(camera.pitch, -1.55, 1.55);
        last_cursor = cursor;

        // Free fly; keys are physical positions (W/A/S/D = Z/Q/S/D on AZERTY).
        const f = camera.forward();
        const r = camera.right();
        var move: [3]f32 = .{ 0, 0, 0 };
        const axes = [_]struct { key: glfw.Key, dir: [3]f32, sign: f32 }{
            .{ .key = .w, .dir = f, .sign = 1 },
            .{ .key = .s, .dir = f, .sign = -1 },
            .{ .key = .d, .dir = r, .sign = 1 },
            .{ .key = .a, .dir = r, .sign = -1 },
            .{ .key = .space, .dir = .{ 0, 1, 0 }, .sign = 1 },
            .{ .key = .left_control, .dir = .{ 0, 1, 0 }, .sign = -1 },
        };
        for (axes) |a| if (window.getKey(a.key) == .press) {
            for (0..3) |i| move[i] += a.dir[i] * a.sign;
        };
        const speed: f32 = walk_speed * (if (window.getKey(.left_shift) == .press) sprint_factor else 1);
        for (0..3) |i| camera.pos[i] += move[i] * speed * dt;

        sun.advance(dt);

        const extent = framebufferExtent(window);
        if (extent.width == 0 or extent.height == 0) continue;
        const aspect = @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
        try renderer.drawFrame(extent, .{
            .view_proj = camera.viewProj(aspect),
            .sun_dir = sun.direction(),
        });
    }
}

fn framebufferExtent(window: *glfw.Window) vk.Extent2D {
    const size = window.getFramebufferSize();
    return .{ .width = @intCast(size[0]), .height = @intCast(size[1]) };
}

test {
    _ = @import("Camera.zig");
    _ = @import("Sun.zig");
    _ = @import("ChunkManager.zig");
}
