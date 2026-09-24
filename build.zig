const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Deps externes
    const vulkan_headers = b.dependency("vulkan_headers", .{});
    const vulkan = b.dependency("vulkan-zig", .{
        .registry = vulkan_headers.path("registry/vk.xml"),
    }).module("vulkan-zig");

    const zglfw = b.dependency("zglfw", .{ .target = target, .optimize = optimize, .import_vulkan = true });
    const zglfw_mod = zglfw.module("root");
    zglfw_mod.addImport("vulkan", vulkan);

    const zmath = b.dependency("zmath", .{ .target = target, .optimize = optimize }).module("root");
    const znoise = b.dependency("znoise", .{ .target = target, .optimize = optimize });

    // Modules internes
    const threading = b.createModule(.{
        .root_source_file = b.path("src/threading/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const world = b.createModule(.{
        .root_source_file = b.path("src/world/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "znoise", .module = znoise.module("root") },
            .{ .name = "zmath", .module = zmath },
        },
    });
    world.linkLibrary(znoise.artifact("FastNoiseLite"));

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "threading", .module = threading },
            .{ .name = "world", .module = world },
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "zglfw", .module = zglfw_mod },
            .{ .name = "zmath", .module = zmath },
        },
    });
    exe_mod.linkLibrary(zglfw.artifact("glfw"));

    // Shaders: GLSL -> SPIR-V with glslc, embedded with @embedFile("<name>").
    const shaders = [_][]const u8{ "fullscreen.vert", "sky.frag", "cull.comp", "chunk.vert", "chunk.frag", "shadow.vert", "outline.vert", "outline.frag", "water.frag" };
    for (shaders) |name| {
        const glslc = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.4", "-O", "-o" });
        const spv = glslc.addOutputFileArg(b.fmt("{s}.spv", .{name}));
        glslc.addFileArg(b.path(b.fmt("src/render/shaders/{s}", .{name})));
        glslc.addFileInput(b.path("src/render/shaders/common.glsl"));
        glslc.addFileInput(b.path("src/render/shaders/gpu.glsl"));
        glslc.addFileInput(b.path("src/render/shaders/pull.glsl"));
        glslc.addFileInput(b.path("src/render/shaders/lighting.glsl"));
        exe_mod.addAnonymousImport(name, .{ .root_source_file = spv });
    }

    const exe = b.addExecutable(.{ .name = "ft_vox", .root_module = exe_mod });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the app").dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run all tests");
    for ([_]*std.Build.Module{ exe_mod, threading, world }) |m| {
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }

    // Benchmark du mesher, toujours en ReleaseFast
    const znoise_fast = b.dependency("znoise", .{ .target = target, .optimize = .ReleaseFast });
    const world_fast = b.createModule(.{
        .root_source_file = b.path("src/world/root.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "znoise", .module = znoise_fast.module("root") },
            .{ .name = "zmath", .module = zmath },
        },
    });
    world_fast.linkLibrary(znoise_fast.artifact("FastNoiseLite"));
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "world", .module = world_fast }},
        }),
    });
    b.step("bench", "Benchmark the mesher (ReleaseFast)").dependOn(&b.addRunArtifact(bench).step);
    test_step.dependOn(&bench.step); // compile only, don't run
}
