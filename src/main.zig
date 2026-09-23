const std = @import("std");
const glfw = @import("zglfw");
const vk = @import("vulkan");
const world = @import("world");
const Context = @import("render/Context.zig");
const Renderer = @import("render/Renderer.zig");
const Camera = @import("Camera.zig");
const ChunkManager = @import("ChunkManager.zig");
const Sun = @import("Sun.zig");

pub const std_options: std.Options = .{
    // Worker pool state changes are too chatty at debug level.
    .log_scope_levels = &.{.{ .scope = .worker_pool, .level = .info }},
};

const walk_speed: f32 = 12; // blocks per second
const sprint_factor: f32 = 6;
const mouse_sensitivity: f32 = 0.0025;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    try glfw.init();
    defer glfw.terminate();
    glfw.windowHint(.client_api, .no_api);
    const window = try glfw.Window.create(1280, 720, "ft_vox", null, null);
    defer window.destroy();
    try glfw.setInputMode(window, .cursor, .disabled);

    var ctx: Context = try .init(gpa, window);
    defer ctx.deinit(gpa);
    var renderer: Renderer = try .init(gpa, &ctx, framebufferExtent(window), .{});
    defer renderer.deinit();

    // One core stays free for the render thread: oversubscribing starves it while loading.
    const workers = @max(2, (std.Thread.getCpuCount() catch 3) - 1);
    const mesh_workers = @max(1, workers / 3);
    const chunks = try ChunkManager.create(gpa, io, .{ .seed = 42, .gen_workers = workers - mesh_workers, .mesh_workers = mesh_workers });
    defer chunks.destroy();

    var camera: Camera = .{};
    var sun: Sun = .{};
    var last_cursor = window.getCursorPos();
    var last_time = glfw.getTime();
    var title_timer: f64 = 0;
    var frames: u32 = 0;

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

        // Streaming: hand finished meshes to the GPU, uploads before unloads.
        try chunks.update(.{ .x = @intFromFloat(@floor(camera.pos[0])), .y = @intFromFloat(@floor(camera.pos[1])), .z = @intFromFloat(@floor(camera.pos[2])) });
        for (chunks.takeUploads()) |u| try renderer.chunks.upload(u.pos, u.mesh.quads, u.mesh.counts);
        for (chunks.takeUnloads()) |p| try renderer.chunks.remove(p);

        const extent = framebufferExtent(window);
        if (extent.width == 0 or extent.height == 0) continue;
        const far: f32 = @floatFromInt(@as(u32, 16) * world.chunk_size);
        try renderer.drawFrame(extent, .{
            .camera = camera,
            .light = sun.lighting(),
            .shadows = sun.direction()[1] > 0.02,
            .fog_start = far * 0.6,
            .fog_end = far * 0.95,
        });

        frames += 1;
        title_timer += dt;
        if (title_timer >= 0.5) {
            var buf: [160]u8 = undefined;
            const title = std.fmt.bufPrintZ(&buf, "ft_vox | {d:.0} fps | {d} chunks | {d} quads | pos {d:.0} {d:.0} {d:.0}", .{
                @as(f64, @floatFromInt(frames)) / title_timer,
                renderer.chunks.chunkCount(),
                renderer.chunks.quad_count,
                camera.pos[0],
                camera.pos[1],
                camera.pos[2],
            }) catch "ft_vox";
            window.setTitle(title);
            frames = 0;
            title_timer = 0;
        }
    }
}

fn framebufferExtent(window: *glfw.Window) vk.Extent2D {
    const size = window.getFramebufferSize();
    return .{ .width = @intCast(size[0]), .height = @intCast(size[1]) };
}

test {
    _ = @import("Camera.zig");
    _ = @import("ChunkManager.zig");
    _ = @import("Sun.zig");
    _ = @import("render/gpu.zig");
    _ = @import("render/Shadows.zig");
}
