const std = @import("std");

const modules = [_]struct { name: [:0]const u8, path: [:0]const u8 }{
    .{ .name = "math", .path = "src/math/root.zig" },
    .{ .name = "threading", .path = "src/threading/root.zig" },
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    var mods: [modules.len]*std.Build.Module = undefined;
    var imports: [modules.len]std.Build.Module.Import = undefined;
    for (modules, 0..) |m, i| {
        mods[i] = b.createModule(.{
            .root_source_file = b.path(m.path),
            .target = target,
            .optimize = optimize,
        });
        imports[i] = .{ .name = m.name, .module = mods[i] };
    }

    const exe = b.addExecutable(.{
        .name = "ft_vox",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &imports,
        }),
    });

    // Vulkan
    const vulkan_headers = b.dependency("vulkan_headers", .{});
    const vulkan = b.dependency("vulkan-zig", .{
        .registry = vulkan_headers.path("registry/vk.xml"),
    }).module("vulkan-zig");

    // Glfw
    const zglfw = b.dependency("zglfw", .{
        .target = target,
        .optimize = optimize,
        .import_vulkan = true,
    });
    const zglfw_mod = zglfw.module("root");
    if (target.result.os.tag != .emscripten) {
        exe.root_module.linkLibrary(zglfw.artifact("glfw"));
    }
    zglfw_mod.addImport("vulkan", vulkan);

    // Math
    const zmath = b.dependency("zmath", .{
        .target = target,
        .optimize = optimize,
    });
    const zmath_mod = zmath.module("root");

    // Noise
    const znoise = b.dependency("znoise", .{
        .target = target,
        .optimize = optimize,
    });
    const znoise_mod = znoise.module("root");

    exe.root_module.addImport("zglfw", zglfw_mod);
    exe.root_module.addImport("vulkan", vulkan);
    exe.root_module.addImport("zmath", zmath_mod);
    exe.root_module.addImport("znoise", znoise_mod);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = exe.root_module })).step);
    for (mods) |mod| {
        const t = b.addTest(.{ .root_module = mod });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
