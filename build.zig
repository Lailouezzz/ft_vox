const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const vulkan_headers = b.dependency("vulkan_headers", .{});
    const vulkan = b.dependency("vulkan-zig", .{
        .registry = vulkan_headers.path("registry/vk.xml"),
    }).module("vulkan-zig");

    const zglfw = b.dependency("zglfw", .{ .target = target, .optimize = optimize, .import_vulkan = true });
    const zglfw_mod = zglfw.module("root");
    zglfw_mod.addImport("vulkan", vulkan);

    const zmath = b.dependency("zmath", .{ .target = target, .optimize = optimize }).module("root");
    const znoise = b.dependency("znoise", .{ .target = target, .optimize = optimize }).module("root");

    // Internal modules
    const math = b.createModule(.{
        .root_source_file = b.path("src/math/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "zmath", .module = zmath }},
    });
    const threading = b.createModule(.{
        .root_source_file = b.path("src/threading/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "math", .module = math },
            .{ .name = "threading", .module = threading },
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "zglfw", .module = zglfw_mod },
            .{ .name = "zmath", .module = zmath },
            .{ .name = "znoise", .module = znoise },
        },
    });
    exe_mod.linkLibrary(zglfw.artifact("glfw"));

    const exe = b.addExecutable(.{ .name = "ft_vox", .root_module = exe_mod });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the app").dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run all tests");
    for ([_]*std.Build.Module{ exe_mod, math, threading }) |m| {
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
}
