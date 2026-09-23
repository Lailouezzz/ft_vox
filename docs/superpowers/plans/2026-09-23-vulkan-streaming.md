# ft_vox — Vulkan bootstrap, ChunkManager, GPU-driven chunks (jalons 3–5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the procedural world on screen with a modern Vulkan 1.4 renderer: streaming chunks through two worker pools, GPU frustum culling, one `vkCmdDrawIndirectCount` for the whole world, vertex pulling through buffer device addresses, sky, sun lighting and fog.

**Architecture:** `render/` holds a thin Vulkan layer (Context, Swapchain, Buffer, Image, pipeline helpers) and a `Renderer` that owns frames in flight and `ChunkBuffers` (one 128 MiB quad buffer sub-allocated by `world.FreeList`, one metadata buffer indexed by slot). Every chunk pipeline reads its data through device addresses passed in push constants; no descriptor sets. `ChunkManager` (CPU only) streams chunks around the camera with a generation pool and a meshing pool and hands finished meshes to the renderer.

**Tech Stack:** Zig 0.16.0, vulkan-zig (`zig-0.16-compat` branch), zglfw, zmath, znoise, GLSL compiled by `glslc`. Spec: `docs/superpowers/specs/2026-09-23-voxel-engine-design.md` (revision r2).

**Provenance:** every code block below comes from a prototype that was built, tested (77/77 tests) and run on the dev GPU (AMD Radeon Renoir, RADV, Vulkan 1.4) with validation layers on and zero validation messages; each intermediate state (after Tasks 8, 9, 10, 11) was replayed from a clean checkout. Measured at the end of Task 11: ~240 fps, ~3000 chunks, ~2.6 M quads in view.

## Global Constraints

- Zig 0.16.0 exactly; run every command from the repo root `/home/lailouezzz/Documents/git/ft_vox`.
- vulkan-zig pinned to branch `zig-0.16-compat`, commit `b496a6a561ffbbeb530b0f9ed4e059f88c0723a5` (the previous pin targets Zig 0.17-dev and does not compile with 0.16.0).
- Vulkan 1.4 device with: dynamicRendering, synchronization2, maintenance4, maintenance5, pushDescriptor, bufferDeviceAddress, scalarBlockLayout, timelineSemaphore, drawIndirectCount, multiDrawIndirect, drawIndirectFirstInstance, shaderInt64, depthClamp.
- Shaders: GLSL, compiled with `glslc --target-env=vulkan1.4 -O`, embedded with `@embedFile("<file name>")`, handed to pipelines through maintenance5 (no `VkShaderModule`).
- Reverse-Z infinite perspective: depth 1 at the near plane, 0 at infinity, compare `GREATER_OR_EQUAL`, clear depth 0.
- zmath matrices (row-vector convention) are uploaded as-is and used as `mat4 * vec4` in GLSL.
- Quad bit layout (from `world.mesher.Quad`): x, y, z, w, h (6 bits each), face (3 bits), block (8 bits); width/height axes ±X → (z, y), ±Y → (x, z), ±Z → (x, y).
- `world` must not import `vulkan`, `zglfw` or `threading`.
- `zig fmt --check src build.zig` must pass before every commit; test logs at level `err` fail the test runner.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` (or the model that actually wrote the commit).

## File Structure

| File | Responsibility |
|---|---|
| `build.zig.zon` | vulkan-zig pin moved to `zig-0.16-compat` |
| `build.zig` | GLSL → SPIR-V with glslc, embedded as anonymous imports |
| `src/Camera.zig` | free-fly camera, view and reverse-Z projection |
| `src/Sun.zig` | day/night cycle: sun direction, light and ambient colors |
| `src/ChunkManager.zig` | chunk streaming: two worker pools, states, edits, uploads/unloads |
| `src/main.zig` | window, input, main loop, wiring |
| `src/render/Context.zig` | instance, debug messenger, surface, physical/logical device, queue |
| `src/render/Swapchain.zig` | swapchain, image views, per-image "render done" semaphores |
| `src/render/Image.zig` | image + memory + view |
| `src/render/Buffer.zig` | buffer + memory + device address + persistent mapping |
| `src/render/pipeline.zig` | SPIR-V embedding, graphics/compute pipeline and layout creation |
| `src/render/gpu.zig` | CPU mirrors of GPU structs, frustum planes |
| `src/render/ChunkBuffers.zig` | quad/metadata buffers, slots, staged uploads |
| `src/render/Renderer.zig` | frames in flight, culling dispatch, sky and chunk passes |
| `src/render/shaders/common.glsl` | sky color, tone mapping |
| `src/render/shaders/gpu.glsl` | buffer-reference declarations shared by chunk pipelines |
| `src/render/shaders/fullscreen.vert`, `sky.frag` | sky |
| `src/render/shaders/cull.comp` | per-chunk frustum + face culling, writes indirect draws |
| `src/render/shaders/chunk.vert`, `chunk.frag` | vertex pulling, sun + ambient lighting, fog |

Tasks 9 and 10 are independent (renderer bootstrap vs. CPU streaming) and can run in parallel; both edit the `test { ... }` block of `src/main.zig`, so merge that block by keeping both imports.

---

### Task 8: vulkan-zig for Zig 0.16, camera and sun

**Files:**
- Modify: `build.zig.zon` (vulkan-zig pin)
- Create: `src/Camera.zig`, `src/Sun.zig`
- Modify: `src/main.zig` (add a `test` block)

**Interfaces:**
- Produces: `Camera { pos: [3]f32, yaw, pitch, fov_y, near }` with `forward()`, `right()`, `view()`, `projection(aspect)`, `viewProj(aspect) zm.Mat`; `Sun { day_length, time }` with `advance(dt)`, `direction() [3]f32`, `lighting() Lighting { dir, color, ambient }`.

- [ ] **Step 1: Move vulkan-zig to the Zig 0.16 compatibility branch**

In `build.zig.zon`, replace the `.@"vulkan-zig"` entry with:

```zig
        .@"vulkan-zig" = .{
            .url = "git+https://github.com/Snektron/vulkan-zig#b496a6a561ffbbeb530b0f9ed4e059f88c0723a5",
            .hash = "vulkan-0.0.0-r7Ytx7N9AwD7IZt5_XNHtkJ4G9qY0pCH-cpcOsbL8wzD",
        },
```

Run: `zig build`
Expected: succeeds (it fetches the new package into `zig-pkg/` on first run).

- [ ] **Step 2: Write the camera and sun (tests included)**

Create `src/Camera.zig`:

```zig
//! Free-fly camera. Matrices use zmath's row-vector convention: their memory
//! layout is exactly what GLSL's column-major `mat4 * vec4` expects.
const std = @import("std");
const zm = @import("zmath");

const Camera = @This();

pos: [3]f32 = .{ 0, 110, 0 },
yaw: f32 = 0, // radians, 0 = looking towards -Z
pitch: f32 = -0.3,
fov_y: f32 = std.math.degreesToRadians(70),
near: f32 = 0.1,

pub fn forward(c: Camera) [3]f32 {
    return .{ -@sin(c.yaw) * @cos(c.pitch), @sin(c.pitch), -@cos(c.yaw) * @cos(c.pitch) };
}

pub fn right(c: Camera) [3]f32 {
    return .{ @cos(c.yaw), 0, -@sin(c.yaw) };
}

pub fn view(c: Camera) zm.Mat {
    const f = c.forward();
    return zm.lookToRh(zm.f32x4(c.pos[0], c.pos[1], c.pos[2], 1), zm.f32x4(f[0], f[1], f[2], 0), zm.f32x4(0, 1, 0, 0));
}

/// Infinite reverse-Z perspective for Vulkan (clip Y down, depth 1 at `near`, 0 at infinity).
/// Depth test GREATER, clear depth 0.
pub fn projection(c: Camera, aspect: f32) zm.Mat {
    const f = 1 / @tan(c.fov_y / 2);
    return .{
        zm.f32x4(f / aspect, 0, 0, 0),
        zm.f32x4(0, -f, 0, 0),
        zm.f32x4(0, 0, 0, -1),
        zm.f32x4(0, 0, c.near, 0),
    };
}

pub fn viewProj(c: Camera, aspect: f32) zm.Mat {
    return zm.mul(c.view(), c.projection(aspect));
}

const testing = std.testing;

test "reverse-Z maps near to 1 and far to ~0" {
    const c: Camera = .{ .pos = .{ 0, 0, 0 }, .yaw = 0, .pitch = 0 };
    const vp = c.viewProj(1);
    const at_near = zm.mul(zm.f32x4(0, 0, -c.near, 1), vp);
    const at_far = zm.mul(zm.f32x4(0, 0, -1e6, 1), vp);
    try testing.expectApproxEqAbs(@as(f32, 1), at_near[2] / at_near[3], 1e-5);
    try testing.expect(at_far[2] / at_far[3] < 1e-6);
}

test "up in the world is up on screen (Vulkan clip Y points down)" {
    const c: Camera = .{ .pos = .{ 0, 0, 0 }, .yaw = 0, .pitch = 0 };
    const p = zm.mul(zm.f32x4(0, 1, -5, 1), c.viewProj(1));
    try testing.expect(p[1] / p[3] < 0);
}
```

Create `src/Sun.zig`:

```zig
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
```

Append to `src/main.zig`:

```zig
test {
    _ = @import("Camera.zig");
    _ = @import("Sun.zig");
}
```

- [ ] **Step 3: Run the tests**

Run: `zig build test --summary all`
Expected: all tests pass, including `reverse-Z maps near to 1 and far to ~0`, `up in the world is up on screen (Vulkan clip Y points down)`, `noon sun is overhead, midnight sun is below` and `time wraps around`.

- [ ] **Step 4: Commit**

```bash
zig fmt --check src build.zig
git add build.zig.zon src/Camera.zig src/Sun.zig src/main.zig
git commit -m "feat: vulkan-zig 0.16 compat pin, reverse-Z camera, day/night sun

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Vulkan 1.4 bootstrap — window, device, swapchain, sky

**Files:**
- Modify: `build.zig` (full replacement: shader compilation)
- Create: `src/render/Context.zig`, `src/render/Swapchain.zig`, `src/render/Image.zig`, `src/render/pipeline.zig`, `src/render/Renderer.zig`
- Create: `src/render/shaders/common.glsl`, `src/render/shaders/fullscreen.vert`, `src/render/shaders/sky.frag`
- Modify: `src/main.zig` (full replacement)

**Interfaces:**
- Consumes: `Camera`, `Sun` (Task 8).
- Produces: `Context.init(gpa, window) !Context`, fields `device: vk.DeviceProxy`, `queue: vk.QueueProxy`, `vkd: *vk.DeviceWrapper`, `queue_family`, `findMemoryType(type_bits, flags) !u32`; `Swapchain`; `Image.init(ctx, extent, format, usage, aspect, layers)`, `Image.initDepth`; `pipeline.spirv(name) []const u32`, `pipeline.createGraphics(ctx, GraphicsDesc) !vk.Pipeline`, `pipeline.createCompute(ctx, layout, code)`, `pipeline.createLayout(ctx, push_size, stages, set_layouts)`; `Renderer.init(gpa, *const Context, extent)`, `Renderer.drawFrame(extent, FrameInput{ view_proj, sun_dir })`, `Renderer.imageBarrier(...)`, `Renderer.frames_in_flight = 2`. Task 11 extends `Renderer` and `FrameInput`.

Design notes the code relies on:
- A "render done" semaphore per swapchain image (not per frame): present gives no signal when it releases a semaphore, so reusing one per frame can race.
- Mailbox present mode when available, FIFO otherwise; the swapchain is recreated on `OUT_OF_DATE`, `SUBOPTIMAL` or a framebuffer size change, and skipped while minimized.
- The sky is a fullscreen triangle drawn first without depth; it reconstructs the view ray from the inverse view-projection.

- [ ] **Step 1: Compile shaders in the build**

Replace `build.zig` with:

```zig
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
    const shaders = [_][]const u8{ "fullscreen.vert", "sky.frag" };
    for (shaders) |name| {
        const glslc = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.4", "-O", "-o" });
        const spv = glslc.addOutputFileArg(b.fmt("{s}.spv", .{name}));
        glslc.addFileArg(b.path(b.fmt("src/render/shaders/{s}", .{name})));
        glslc.addFileInput(b.path("src/render/shaders/common.glsl"));
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
```

- [ ] **Step 2: Write the shaders**

Create `src/render/shaders/common.glsl`:

```glsl
// Shared lighting helpers.
#extension GL_EXT_scalar_block_layout : require

vec3 skyColor(vec3 dir, vec3 sun) {
    float day = clamp(sun.y * 4.0 + 0.5, 0.0, 1.0);
    vec3 zenith = mix(vec3(0.01, 0.01, 0.04), vec3(0.20, 0.45, 0.90), day);
    vec3 horizon = mix(vec3(0.05, 0.05, 0.10), vec3(0.70, 0.80, 0.95), day);
    // Warm horizon at sunrise/sunset.
    float dusk = clamp(1.0 - abs(sun.y) * 5.0, 0.0, 1.0);
    horizon = mix(horizon, vec3(1.0, 0.45, 0.20), dusk * 0.6);
    float t = pow(clamp(dir.y, 0.0, 1.0), 0.5);
    vec3 col = mix(horizon, zenith, t);
    if (dir.y < 0.0) col = mix(horizon, horizon * 0.3, clamp(-dir.y * 3.0, 0.0, 1.0));
    float sd = max(dot(dir, sun), 0.0);
    col += vec3(1.0, 0.9, 0.7) * (smoothstep(0.9995, 0.9998, sd) * 20.0 + pow(sd, 64.0) * 0.5 * day);
    return col;
}

// ACES filmic approximation (Narkowicz).
vec3 tonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
```

Create `src/render/shaders/fullscreen.vert`:

```glsl
#version 460
// Fullscreen triangle, no vertex buffer.
layout(location = 0) out vec2 ndc;

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    ndc = uv * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
```

Create `src/render/shaders/sky.frag`:

```glsl
#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec2 ndc;
layout(location = 0) out vec4 out_color;

layout(push_constant, scalar) uniform Push {
    mat4 inv_view_proj;
    vec4 sun_dir; // xyz: direction towards the sun
} pc;

void main() {
    // Reverse-Z: depth 0 is infinitely far.
    vec4 far = pc.inv_view_proj * vec4(ndc, 1e-6, 1.0);
    vec4 near = pc.inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 dir = normalize(far.xyz / far.w - near.xyz / near.w);
    out_color = vec4(tonemap(skyColor(dir, pc.sun_dir.xyz)), 1.0);
}
```

- [ ] **Step 3: Write the Vulkan layer**

Create `src/render/Context.zig`:

```zig
//! Vulkan 1.4 instance, surface, physical device and logical device.
const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vulkan");
const glfw = @import("zglfw");
const Allocator = std.mem.Allocator;

const log = std.log.scoped(.vulkan);

const Context = @This();

const validation_layer = "VK_LAYER_KHRONOS_validation";

vkb: vk.BaseWrapper,
vki: *vk.InstanceWrapper,
vkd: *vk.DeviceWrapper,
instance: vk.InstanceProxy,
debug_messenger: vk.DebugUtilsMessengerEXT,
surface: vk.SurfaceKHR,
pdev: vk.PhysicalDevice,
props: vk.PhysicalDeviceProperties,
mem_props: vk.PhysicalDeviceMemoryProperties,
device: vk.DeviceProxy,
queue_family: u32,
queue: vk.QueueProxy,

pub fn init(gpa: Allocator, window: *glfw.Window) !Context {
    var self: Context = undefined;
    self.vkb = vk.BaseWrapper.load(glfw.getInstanceProcAddress);

    // Instance: GLFW's surface extensions + debug utils; validation in Debug if installed.
    const glfw_exts = try glfw.getRequiredInstanceExtensions();
    var exts: std.ArrayList([*:0]const u8) = .empty;
    defer exts.deinit(gpa);
    try exts.appendSlice(gpa, glfw_exts);
    const debug = builtin.mode == .Debug and try hasLayer(gpa, self.vkb, validation_layer);
    if (debug) try exts.append(gpa, vk.extensions.ext_debug_utils.name);
    const layers: []const [*:0]const u8 = if (debug) &.{validation_layer} else &.{};
    log.info("validation layers: {s}", .{if (debug) "on" else "off"});

    const debug_info: vk.DebugUtilsMessengerCreateInfoEXT = .{
        .message_severity = .{ .warning_bit_ext = true, .error_bit_ext = true },
        .message_type = .{ .general_bit_ext = true, .validation_bit_ext = true, .performance_bit_ext = true },
        .pfn_user_callback = debugCallback,
    };
    const instance = try self.vkb.createInstance(&.{
        .p_next = if (debug) &debug_info else null,
        .p_application_info = &.{
            .p_application_name = "ft_vox",
            .application_version = 0,
            .p_engine_name = "ft_vox",
            .engine_version = 0,
            .api_version = @bitCast(vk.API_VERSION_1_4),
        },
        .enabled_layer_count = @intCast(layers.len),
        .pp_enabled_layer_names = layers.ptr,
        .enabled_extension_count = @intCast(exts.items.len),
        .pp_enabled_extension_names = exts.items.ptr,
    }, null);

    self.vki = try gpa.create(vk.InstanceWrapper);
    errdefer gpa.destroy(self.vki);
    self.vki.* = vk.InstanceWrapper.load(instance, self.vkb.dispatch.vkGetInstanceProcAddr.?);
    self.instance = vk.InstanceProxy.init(instance, self.vki);
    errdefer self.instance.destroyInstance(null);

    self.debug_messenger = if (debug) try self.instance.createDebugUtilsMessengerEXT(&debug_info, null) else .null_handle;
    errdefer if (debug) self.instance.destroyDebugUtilsMessengerEXT(self.debug_messenger, null);

    try glfw.createWindowSurface(instance, window, null, &self.surface);
    errdefer self.instance.destroySurfaceKHR(self.surface, null);

    try self.pickPhysicalDevice(gpa);
    self.props = self.instance.getPhysicalDeviceProperties(self.pdev);
    self.mem_props = self.instance.getPhysicalDeviceMemoryProperties(self.pdev);
    log.info("GPU: {s}", .{std.mem.sliceTo(&self.props.device_name, 0)});

    // Device: one graphics+present queue, the features the renderer relies on.
    var f14: vk.PhysicalDeviceVulkan14Features = .{ .push_descriptor = .true, .maintenance_5 = .true };
    var f13: vk.PhysicalDeviceVulkan13Features = .{ .p_next = &f14, .dynamic_rendering = .true, .synchronization_2 = .true, .maintenance_4 = .true };
    var f12: vk.PhysicalDeviceVulkan12Features = .{ .p_next = &f13, .buffer_device_address = .true, .draw_indirect_count = .true, .timeline_semaphore = .true, .scalar_block_layout = .true };
    const f10: vk.PhysicalDeviceFeatures2 = .{ .p_next = &f12, .features = .{ .shader_int_64 = .true, .multi_draw_indirect = .true, .draw_indirect_first_instance = .true, .depth_clamp = .true } };
    const priority = [_]f32{1};
    const dev = try self.instance.createDevice(self.pdev, &.{
        .p_next = &f10,
        .queue_create_info_count = 1,
        .p_queue_create_infos = &.{.{ .queue_family_index = self.queue_family, .queue_count = 1, .p_queue_priorities = &priority }},
        .enabled_extension_count = 1,
        .pp_enabled_extension_names = &.{vk.extensions.khr_swapchain.name},
    }, null);
    self.vkd = try gpa.create(vk.DeviceWrapper);
    errdefer gpa.destroy(self.vkd);
    self.vkd.* = vk.DeviceWrapper.load(dev, self.vki.dispatch.vkGetDeviceProcAddr.?);
    self.device = vk.DeviceProxy.init(dev, self.vkd);
    self.queue = vk.QueueProxy.init(self.device.getDeviceQueue(self.queue_family, 0), self.vkd);
    return self;
}

pub fn deinit(self: *Context, gpa: Allocator) void {
    self.device.destroyDevice(null);
    self.instance.destroySurfaceKHR(self.surface, null);
    if (self.debug_messenger != .null_handle) self.instance.destroyDebugUtilsMessengerEXT(self.debug_messenger, null);
    self.instance.destroyInstance(null);
    gpa.destroy(self.vkd);
    gpa.destroy(self.vki);
    self.* = undefined;
}

/// First device exposing Vulkan 1.4, the required features and a graphics+present queue.
fn pickPhysicalDevice(self: *Context, gpa: Allocator) !void {
    const pdevs = try self.instance.enumeratePhysicalDevicesAlloc(gpa);
    defer gpa.free(pdevs);
    for (pdevs) |pdev| {
        const props = self.instance.getPhysicalDeviceProperties(pdev);
        const name = std.mem.sliceTo(&props.device_name, 0);
        if (@as(vk.Version, @bitCast(props.api_version)).minor < 4) {
            log.info("skipping {s}: Vulkan 1.4 required", .{name});
            continue;
        }
        if (missingFeature(self.instance, pdev)) |missing| {
            log.info("skipping {s}: missing feature {s}", .{ name, missing });
            continue;
        }
        const families = try self.instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(pdev, gpa);
        defer gpa.free(families);
        for (families, 0..) |fam, i| {
            const idx: u32 = @intCast(i);
            if (!fam.queue_flags.graphics_bit or !fam.queue_flags.compute_bit) continue;
            if (try self.instance.getPhysicalDeviceSurfaceSupportKHR(pdev, idx, self.surface) != .true) continue;
            self.pdev = pdev;
            self.queue_family = idx;
            return;
        }
    }
    return error.NoSuitableGpu;
}

fn missingFeature(instance: vk.InstanceProxy, pdev: vk.PhysicalDevice) ?[]const u8 {
    var f14: vk.PhysicalDeviceVulkan14Features = .{};
    var f13: vk.PhysicalDeviceVulkan13Features = .{ .p_next = &f14 };
    var f12: vk.PhysicalDeviceVulkan12Features = .{ .p_next = &f13 };
    var f10: vk.PhysicalDeviceFeatures2 = .{ .p_next = &f12, .features = .{} };
    instance.getPhysicalDeviceFeatures2(pdev, &f10);
    const checks = .{
        .{ f14.push_descriptor, "pushDescriptor" },
        .{ f14.maintenance_5, "maintenance5" },
        .{ f13.dynamic_rendering, "dynamicRendering" },
        .{ f13.synchronization_2, "synchronization2" },
        .{ f13.maintenance_4, "maintenance4" },
        .{ f12.buffer_device_address, "bufferDeviceAddress" },
        .{ f12.draw_indirect_count, "drawIndirectCount" },
        .{ f12.timeline_semaphore, "timelineSemaphore" },
        .{ f12.scalar_block_layout, "scalarBlockLayout" },
        .{ f10.features.shader_int_64, "shaderInt64" },
        .{ f10.features.multi_draw_indirect, "multiDrawIndirect" },
        .{ f10.features.draw_indirect_first_instance, "drawIndirectFirstInstance" },
        .{ f10.features.depth_clamp, "depthClamp" },
    };
    inline for (checks) |c| if (c[0] != .true) return c[1];
    return null;
}

fn hasLayer(gpa: Allocator, vkb: vk.BaseWrapper, name: []const u8) !bool {
    const layers = try vkb.enumerateInstanceLayerPropertiesAlloc(gpa);
    defer gpa.free(layers);
    for (layers) |l| if (std.mem.eql(u8, std.mem.sliceTo(&l.layer_name, 0), name)) return true;
    return false;
}

fn debugCallback(
    severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    _: vk.DebugUtilsMessageTypeFlagsEXT,
    data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    _: ?*anyopaque,
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    const msg = if (data) |d| d.p_message orelse "?" else "?";
    if (severity.error_bit_ext) log.err("{s}", .{msg}) else log.warn("{s}", .{msg});
    return .false;
}

/// Index of a memory type allowed by `type_bits` with all `flags`.
pub fn findMemoryType(self: *const Context, type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
    for (self.mem_props.memory_types[0..self.mem_props.memory_type_count], 0..) |t, i| {
        if (type_bits & (@as(u32, 1) << @intCast(i)) != 0 and t.property_flags.contains(flags)) return @intCast(i);
    }
    return error.NoSuitableMemoryType;
}
```

Create `src/render/Swapchain.zig`:

```zig
//! Swapchain + image views. Recreated on resize or when presentation reports out of date.
const std = @import("std");
const vk = @import("vulkan");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");

const Swapchain = @This();

handle: vk.SwapchainKHR,
format: vk.Format,
extent: vk.Extent2D,
images: []vk.Image,
views: []vk.ImageView,
/// One per image: signalled when rendering to that image is done, waited on by present.
/// Per image rather than per frame, since present gives no signal of when it releases it.
render_done: []vk.Semaphore,

pub fn init(ctx: *const Context, gpa: Allocator, extent: vk.Extent2D, old: vk.SwapchainKHR) !Swapchain {
    const caps = try ctx.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(ctx.pdev, ctx.surface);
    const format = try pickFormat(ctx, gpa);
    const present_mode = try pickPresentMode(ctx, gpa);
    const actual: vk.Extent2D = if (caps.current_extent.width != std.math.maxInt(u32)) caps.current_extent else .{
        .width = std.math.clamp(extent.width, caps.min_image_extent.width, caps.max_image_extent.width),
        .height = std.math.clamp(extent.height, caps.min_image_extent.height, caps.max_image_extent.height),
    };
    var image_count = caps.min_image_count + 1;
    if (caps.max_image_count > 0) image_count = @min(image_count, caps.max_image_count);

    const handle = try ctx.device.createSwapchainKHR(&.{
        .surface = ctx.surface,
        .min_image_count = image_count,
        .image_format = format.format,
        .image_color_space = format.color_space,
        .image_extent = actual,
        .image_array_layers = 1,
        .image_usage = .{ .color_attachment_bit = true },
        .image_sharing_mode = .exclusive,
        .pre_transform = caps.current_transform,
        .composite_alpha = .{ .opaque_bit_khr = true },
        .present_mode = present_mode,
        .clipped = .true,
        .old_swapchain = old,
    }, null);
    errdefer ctx.device.destroySwapchainKHR(handle, null);
    if (old != .null_handle) ctx.device.destroySwapchainKHR(old, null);

    const images = try ctx.device.getSwapchainImagesAllocKHR(handle, gpa);
    errdefer gpa.free(images);
    const views = try gpa.alloc(vk.ImageView, images.len);
    errdefer gpa.free(views);
    const render_done = try gpa.alloc(vk.Semaphore, images.len);
    errdefer gpa.free(render_done);
    for (images, views, render_done) |img, *view, *sem| {
        view.* = try ctx.device.createImageView(&.{
            .image = img,
            .view_type = .@"2d",
            .format = format.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{ .aspect_mask = .{ .color_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
        }, null);
        sem.* = try ctx.device.createSemaphore(&.{}, null);
    }
    return .{ .handle = handle, .format = format.format, .extent = actual, .images = images, .views = views, .render_done = render_done };
}

/// Destroys views and semaphores; keeps `handle` alive when it is handed to `init` as `old`.
pub fn deinitKeepHandle(self: *Swapchain, ctx: *const Context, gpa: Allocator) void {
    for (self.views, self.render_done) |v, s| {
        ctx.device.destroyImageView(v, null);
        ctx.device.destroySemaphore(s, null);
    }
    gpa.free(self.views);
    gpa.free(self.render_done);
    gpa.free(self.images);
}

pub fn deinit(self: *Swapchain, ctx: *const Context, gpa: Allocator) void {
    self.deinitKeepHandle(ctx, gpa);
    ctx.device.destroySwapchainKHR(self.handle, null);
    self.* = undefined;
}

fn pickFormat(ctx: *const Context, gpa: Allocator) !vk.SurfaceFormatKHR {
    const formats = try ctx.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(ctx.pdev, ctx.surface, gpa);
    defer gpa.free(formats);
    for (formats) |f| {
        if (f.format == .b8g8r8a8_srgb and f.color_space == .srgb_nonlinear_khr) return f;
    }
    return formats[0];
}

/// Mailbox (low latency, no tearing) when available, otherwise FIFO (always supported).
fn pickPresentMode(ctx: *const Context, gpa: Allocator) !vk.PresentModeKHR {
    const modes = try ctx.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(ctx.pdev, ctx.surface, gpa);
    defer gpa.free(modes);
    for (modes) |m| if (m == .mailbox_khr) return m;
    return .fifo_khr;
}
```

Create `src/render/Image.zig`:

```zig
//! A device-local image with its own memory allocation and one view.
const std = @import("std");
const vk = @import("vulkan");
const Context = @import("Context.zig");

const Image = @This();

image: vk.Image,
memory: vk.DeviceMemory,
view: vk.ImageView,

pub fn init(ctx: *const Context, extent: vk.Extent2D, format: vk.Format, usage: vk.ImageUsageFlags, aspect: vk.ImageAspectFlags, layers: u32) !Image {
    const image = try ctx.device.createImage(&.{
        .image_type = .@"2d",
        .format = format,
        .extent = .{ .width = extent.width, .height = extent.height, .depth = 1 },
        .mip_levels = 1,
        .array_layers = layers,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = usage,
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, null);
    errdefer ctx.device.destroyImage(image, null);
    const req = ctx.device.getImageMemoryRequirements(image);
    const memory = try ctx.device.allocateMemory(&.{
        .allocation_size = req.size,
        .memory_type_index = try ctx.findMemoryType(req.memory_type_bits, .{ .device_local_bit = true }),
    }, null);
    errdefer ctx.device.freeMemory(memory, null);
    try ctx.device.bindImageMemory(image, memory, 0);
    const view = try ctx.device.createImageView(&.{
        .image = image,
        .view_type = if (layers > 1) .@"2d_array" else .@"2d",
        .format = format,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = .{ .aspect_mask = aspect, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = layers },
    }, null);
    return .{ .image = image, .memory = memory, .view = view };
}

pub fn initDepth(ctx: *const Context, extent: vk.Extent2D, format: vk.Format) !Image {
    return init(ctx, extent, format, .{ .depth_stencil_attachment_bit = true }, .{ .depth_bit = true }, 1);
}

pub fn deinit(self: *Image, ctx: *const Context) void {
    ctx.device.destroyImageView(self.view, null);
    ctx.device.destroyImage(self.image, null);
    ctx.device.freeMemory(self.memory, null);
    self.* = undefined;
}
```

Create `src/render/pipeline.zig`:

```zig
//! Pipeline helpers. SPIR-V goes straight into the stage via maintenance5
//! (VkShaderModuleCreateInfo chained on the stage): no VkShaderModule objects.
const std = @import("std");
const vk = @import("vulkan");
const Context = @import("Context.zig");

/// Embedded SPIR-V of a shader built by build.zig (see `shaders` there).
pub fn spirv(comptime name: []const u8) []const u32 {
    const bytes align(@alignOf(u32)) = @embedFile(name).*;
    return std.mem.bytesAsSlice(u32, &bytes);
}

pub const Stage = struct {
    stage: vk.ShaderStageFlags,
    code: []const u32,
};

pub const GraphicsDesc = struct {
    layout: vk.PipelineLayout,
    vertex: []const u32,
    fragment: ?[]const u32,
    color_format: ?vk.Format,
    depth_format: vk.Format,
    depth_test: bool = true,
    depth_write: bool = true,
    depth_clamp: bool = false,
    cull_back: bool = true,
    topology: vk.PrimitiveTopology = .triangle_list,
    depth_bias: bool = false,
};

pub fn createGraphics(ctx: *const Context, d: GraphicsDesc) !vk.Pipeline {
    var modules: [2]vk.ShaderModuleCreateInfo = undefined;
    var stages: [2]vk.PipelineShaderStageCreateInfo = undefined;
    var n: u32 = 0;
    for ([_]?[]const u32{ d.vertex, d.fragment }, [_]vk.ShaderStageFlags{ .{ .vertex_bit = true }, .{ .fragment_bit = true } }) |code, stage| {
        const c = code orelse continue;
        modules[n] = .{ .code_size = c.len * 4, .p_code = c.ptr };
        stages[n] = .{ .p_next = &modules[n], .stage = stage, .module = .null_handle, .p_name = "main" };
        n += 1;
    }
    const color_formats: []const vk.Format = if (d.color_format) |*f| f[0..1] else &.{};
    const rendering: vk.PipelineRenderingCreateInfo = .{
        .view_mask = 0,
        .color_attachment_count = @intCast(color_formats.len),
        .p_color_attachment_formats = color_formats.ptr,
        .depth_attachment_format = d.depth_format,
        .stencil_attachment_format = .undefined,
    };
    const dynamic = [_]vk.DynamicState{ .viewport, .scissor, .depth_bias };
    const blend = vk.PipelineColorBlendAttachmentState{
        .blend_enable = .false,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };
    var pipeline: vk.Pipeline = undefined;
    _ = try ctx.device.createGraphicsPipelines(.null_handle, &.{.{
        .p_next = &rendering,
        .stage_count = n,
        .p_stages = &stages,
        .p_vertex_input_state = &.{},
        .p_input_assembly_state = &.{ .topology = d.topology, .primitive_restart_enable = .false },
        .p_viewport_state = &.{ .viewport_count = 1, .scissor_count = 1 },
        .p_rasterization_state = &.{
            .depth_clamp_enable = if (d.depth_clamp) .true else .false,
            .rasterizer_discard_enable = .false,
            .polygon_mode = .fill,
            .cull_mode = if (d.cull_back) .{ .back_bit = true } else .{},
            .front_face = .counter_clockwise,
            .depth_bias_enable = if (d.depth_bias) .true else .false,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1,
        },
        .p_multisample_state = &.{ .rasterization_samples = .{ .@"1_bit" = true }, .sample_shading_enable = .false, .min_sample_shading = 1, .alpha_to_coverage_enable = .false, .alpha_to_one_enable = .false },
        .p_depth_stencil_state = &.{
            .depth_test_enable = if (d.depth_test) .true else .false,
            .depth_write_enable = if (d.depth_write) .true else .false,
            .depth_compare_op = .greater_or_equal, // reverse-Z
            .depth_bounds_test_enable = .false,
            .stencil_test_enable = .false,
            .front = std.mem.zeroes(vk.StencilOpState),
            .back = std.mem.zeroes(vk.StencilOpState),
            .min_depth_bounds = 0,
            .max_depth_bounds = 1,
        },
        .p_color_blend_state = &.{ .logic_op_enable = .false, .logic_op = .copy, .attachment_count = @intCast(color_formats.len), .p_attachments = @ptrCast(&blend), .blend_constants = .{ 0, 0, 0, 0 } },
        .p_dynamic_state = &.{ .dynamic_state_count = dynamic.len, .p_dynamic_states = &dynamic },
        .layout = d.layout,
        .render_pass = .null_handle,
        .subpass = 0,
        .base_pipeline_index = -1,
    }}, null, (&pipeline)[0..1]);
    return pipeline;
}

pub fn createCompute(ctx: *const Context, layout: vk.PipelineLayout, code: []const u32) !vk.Pipeline {
    const module: vk.ShaderModuleCreateInfo = .{ .code_size = code.len * 4, .p_code = code.ptr };
    var pipeline: vk.Pipeline = undefined;
    _ = try ctx.device.createComputePipelines(.null_handle, &.{.{
        .stage = .{ .p_next = &module, .stage = .{ .compute_bit = true }, .module = .null_handle, .p_name = "main" },
        .layout = layout,
        .base_pipeline_index = -1,
    }}, null, (&pipeline)[0..1]);
    return pipeline;
}

pub fn createLayout(ctx: *const Context, push_size: u32, stages: vk.ShaderStageFlags, set_layouts: []const vk.DescriptorSetLayout) !vk.PipelineLayout {
    const range: vk.PushConstantRange = .{ .stage_flags = stages, .offset = 0, .size = push_size };
    return ctx.device.createPipelineLayout(&.{
        .set_layout_count = @intCast(set_layouts.len),
        .p_set_layouts = set_layouts.ptr,
        .push_constant_range_count = if (push_size > 0) 1 else 0,
        .p_push_constant_ranges = @ptrCast(&range),
    }, null);
}
```

Create `src/render/Renderer.zig`:

```zig
//! Frame loop: frames in flight, depth buffer, passes.
const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");
const Swapchain = @import("Swapchain.zig");
const pipeline = @import("pipeline.zig");
const Image = @import("Image.zig");

const Renderer = @This();

pub const frames_in_flight = 2;
const depth_format: vk.Format = .d32_sfloat;

const Frame = struct {
    pool: vk.CommandPool,
    cmd: vk.CommandBuffer,
    image_acquired: vk.Semaphore,
    fence: vk.Fence,
};

pub const FrameInput = struct {
    view_proj: zm.Mat,
    sun_dir: [3]f32,
};

gpa: Allocator,
ctx: *const Context,
swapchain: Swapchain,
depth: Image,
frames: [frames_in_flight]Frame,
frame_index: usize = 0,
sky_layout: vk.PipelineLayout,
sky_pipeline: vk.Pipeline,

pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D) !Renderer {
    var swapchain: Swapchain = try .init(ctx, gpa, extent, .null_handle);
    errdefer swapchain.deinit(ctx, gpa);
    var depth: Image = try .initDepth(ctx, swapchain.extent, depth_format);
    errdefer depth.deinit(ctx);

    var frames: [frames_in_flight]Frame = undefined;
    for (&frames) |*f| {
        f.pool = try ctx.device.createCommandPool(&.{ .flags = .{ .reset_command_buffer_bit = true }, .queue_family_index = ctx.queue_family }, null);
        try ctx.device.allocateCommandBuffers(&.{ .command_pool = f.pool, .level = .primary, .command_buffer_count = 1 }, @ptrCast(&f.cmd));
        f.image_acquired = try ctx.device.createSemaphore(&.{}, null);
        f.fence = try ctx.device.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
    }

    const sky_layout = try pipeline.createLayout(ctx, @sizeOf(SkyPush), .{ .fragment_bit = true }, &.{});
    const sky_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = sky_layout,
        .vertex = pipeline.spirv("fullscreen.vert"),
        .fragment = pipeline.spirv("sky.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
        .depth_test = false,
        .depth_write = false,
        .cull_back = false,
    });
    return .{
        .gpa = gpa,
        .ctx = ctx,
        .swapchain = swapchain,
        .depth = depth,
        .frames = frames,
        .sky_layout = sky_layout,
        .sky_pipeline = sky_pipeline,
    };
}

pub fn deinit(self: *Renderer) void {
    const d = self.ctx.device;
    d.deviceWaitIdle() catch {};
    d.destroyPipeline(self.sky_pipeline, null);
    d.destroyPipelineLayout(self.sky_layout, null);
    for (self.frames) |f| {
        d.destroyFence(f.fence, null);
        d.destroySemaphore(f.image_acquired, null);
        d.destroyCommandPool(f.pool, null);
    }
    self.depth.deinit(self.ctx);
    self.swapchain.deinit(self.ctx, self.gpa);
}

const SkyPush = extern struct {
    inv_view_proj: zm.Mat,
    sun_dir: [4]f32,
};

/// Renders one frame. `extent` is the current framebuffer size (for resizes).
pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
    const d = self.ctx.device;
    const frame = &self.frames[self.frame_index];
    _ = try d.waitForFences(&.{frame.fence}, .true, std.math.maxInt(u64));

    const acquired = d.acquireNextImageKHR(self.swapchain.handle, std.math.maxInt(u64), frame.image_acquired, .null_handle) catch |err| switch (err) {
        error.OutOfDateKHR => return self.recreate(extent),
        else => return err,
    };
    try d.resetFences(&.{frame.fence});
    const image_index = acquired.image_index;

    const cmd: vk.CommandBufferProxy = .init(frame.cmd, self.ctx.vkd);
    try cmd.resetCommandBuffer(.{});
    try cmd.beginCommandBuffer(&.{ .flags = .{ .one_time_submit_bit = true } });

    const image = self.swapchain.images[image_index];
    imageBarrier(cmd, image, .{ .color_bit = true }, .undefined, .color_attachment_optimal, .{ .color_attachment_output_bit = true }, .{}, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true });
    imageBarrier(cmd, self.depth.image, .{ .depth_bit = true }, .undefined, .depth_attachment_optimal, .{ .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .early_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true });

    const ext = self.swapchain.extent;
    const color_att: vk.RenderingAttachmentInfo = .{
        .image_view = self.swapchain.views[image_index],
        .image_layout = .color_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .dont_care,
        .store_op = .store,
        .clear_value = .{ .color = .{ .float_32 = .{ 0, 0, 0, 1 } } },
    };
    const depth_att: vk.RenderingAttachmentInfo = .{
        .image_view = self.depth.view,
        .image_layout = .depth_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .clear,
        .store_op = .dont_care,
        .clear_value = .{ .depth_stencil = .{ .depth = 0, .stencil = 0 } },
    };
    cmd.beginRendering(&.{
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = ext },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_att),
        .p_depth_attachment = &depth_att,
    });
    setViewport(cmd, ext);

    const sky: SkyPush = .{ .inv_view_proj = zm.inverse(in.view_proj), .sun_dir = .{ in.sun_dir[0], in.sun_dir[1], in.sun_dir[2], 0 } };
    cmd.bindPipeline(.graphics, self.sky_pipeline);
    cmd.pushConstants(self.sky_layout, .{ .fragment_bit = true }, 0, @sizeOf(SkyPush), &sky);
    cmd.draw(3, 1, 0, 0);

    cmd.endRendering();
    imageBarrier(cmd, image, .{ .color_bit = true }, .color_attachment_optimal, .present_src_khr, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true }, .{}, .{});
    try cmd.endCommandBuffer();

    const render_done = self.swapchain.render_done[image_index];
    try self.ctx.queue.submit2(&.{.{
        .wait_semaphore_info_count = 1,
        .p_wait_semaphore_infos = &.{.{ .semaphore = frame.image_acquired, .value = 0, .stage_mask = .{ .color_attachment_output_bit = true }, .device_index = 0 }},
        .command_buffer_info_count = 1,
        .p_command_buffer_infos = &.{.{ .command_buffer = frame.cmd, .device_mask = 0 }},
        .signal_semaphore_info_count = 1,
        .p_signal_semaphore_infos = &.{.{ .semaphore = render_done, .value = 0, .stage_mask = .{ .all_commands_bit = true }, .device_index = 0 }},
    }}, frame.fence);

    self.frame_index = (self.frame_index + 1) % frames_in_flight;
    const present = self.ctx.queue.presentKHR(&.{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&render_done),
        .swapchain_count = 1,
        .p_swapchains = @ptrCast(&self.swapchain.handle),
        .p_image_indices = @ptrCast(&image_index),
    }) catch |err| switch (err) {
        error.OutOfDateKHR => return self.recreate(extent),
        else => return err,
    };
    if (present == .suboptimal_khr or extent.width != self.swapchain.extent.width or extent.height != self.swapchain.extent.height)
        try self.recreate(extent);
}

fn recreate(self: *Renderer, extent: vk.Extent2D) !void {
    if (extent.width == 0 or extent.height == 0) return; // minimized
    try self.ctx.device.deviceWaitIdle();
    self.swapchain.deinitKeepHandle(self.ctx, self.gpa);
    self.swapchain = try .init(self.ctx, self.gpa, extent, self.swapchain.handle);
    self.depth.deinit(self.ctx);
    self.depth = try .initDepth(self.ctx, self.swapchain.extent, depth_format);
}

fn setViewport(cmd: vk.CommandBufferProxy, ext: vk.Extent2D) void {
    cmd.setViewport(0, &.{.{ .x = 0, .y = 0, .width = @floatFromInt(ext.width), .height = @floatFromInt(ext.height), .min_depth = 0, .max_depth = 1 }});
    cmd.setScissor(0, &.{.{ .offset = .{ .x = 0, .y = 0 }, .extent = ext }});
}

pub fn imageBarrier(
    cmd: vk.CommandBufferProxy,
    image: vk.Image,
    aspect: vk.ImageAspectFlags,
    old: vk.ImageLayout,
    new: vk.ImageLayout,
    src_stage: vk.PipelineStageFlags2,
    src_access: vk.AccessFlags2,
    dst_stage: vk.PipelineStageFlags2,
    dst_access: vk.AccessFlags2,
) void {
    const b: vk.ImageMemoryBarrier2 = .{
        .src_stage_mask = src_stage,
        .src_access_mask = src_access,
        .dst_stage_mask = dst_stage,
        .dst_access_mask = dst_access,
        .old_layout = old,
        .new_layout = new,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{ .aspect_mask = aspect, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = vk.REMAINING_ARRAY_LAYERS },
    };
    cmd.pipelineBarrier2(&.{ .image_memory_barrier_count = 1, .p_image_memory_barriers = @ptrCast(&b) });
}
```

- [ ] **Step 4: Replace the main loop**

Replace `src/main.zig` with (the old `Io.concurrent` demo goes away):

```zig
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
}
```

- [ ] **Step 5: Build, test, run**

Run: `zig build test --summary all`
Expected: all tests pass.

Run: `zig build && timeout 6 ./zig-out/bin/ft_vox 2>&1 | grep -v worker_pool`
Expected: exactly these two lines (the window stays open about 6 seconds), and no `error(vulkan)` or `warning(vulkan)` line:

```
info(vulkan): validation layers: on
info(vulkan): GPU: <your GPU name>
```

If `validation layers: off` is printed, install the Khronos validation layer (`vulkan-validation-layers` on Arch) and rerun: this check is meaningless without it.

Visually (if a display is available): a sky gradient with a sun disc; the mouse turns the view; the sky darkens and warms as the sun moves (a full day lasts 240 s); resizing the window keeps rendering.

- [ ] **Step 6: Commit**

```bash
zig fmt --check src build.zig
git add build.zig src/render src/main.zig
git commit -m "feat(render): Vulkan 1.4 bootstrap with dynamic rendering and sky

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: ChunkManager — streaming with two worker pools

**Files:**
- Create: `src/ChunkManager.zig`
- Modify: `src/main.zig` (`test` block)

**Interfaces:**
- Consumes: `world.Generator`, `world.generate`, `world.Chunk`, `world.ChunkPos`, `world.BlockPos`, `world.mesher.{Volume, buildVolume, mesh, Mesh}`, `threading.{WorkQueue, WorkerPool}`.
- Produces: `ChunkManager.create(gpa, io, Config) !*ChunkManager`, `destroy()`, `update(center: world.BlockPos) !void`, `takeUploads() []const Upload` (`Upload = { pos: ChunkPos, mesh: Mesh }`), `takeUnloads() []const ChunkPos`, `breakBlock(pos) !bool`, `blockAt(pos) world.Block`, field `stats: Stats`. `Config = { seed: u64, radius: u16 = 16, unload_margin: u16 = 2, gen_workers: usize, mesh_workers: usize }`. Lists returned by `take*` are valid until the next `update`; the manager frees the meshes itself. Callers must apply uploads before unloads.

Behavior (spec r2, "Streaming et édition"): generation ring radius + 1, meshing within radius once the 6 face neighbours exist; gen tasks cannot fail (sender allocates the chunk); mesh tasks catch errors and return `mesh = null` (chunk goes back to dirty) and free their volume; at most one mesh in flight per chunk, so N edits during a mesh give exactly one extra remesh; edit remeshes go to the front of the mesh queue; edited chunks survive unload in an edited table; `destroy` pops and frees every queued job before shutting the pools down.

- [ ] **Step 1: Write the ChunkManager (tests included)**

Create `src/ChunkManager.zig`:

```zig
//! Chunk streaming and editing. Lives on the main thread between two worker
//! pools: gen_in → GenPool → gen_out → ChunkManager → mesh_in → MeshPool → mesh_out.
//! No GPU code: the renderer consumes `takeUploads` / `takeUnloads` each frame
//! and keys its allocations by `ChunkPos`.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const threading = @import("threading");
const world = @import("world");
const Chunk = world.Chunk;
const ChunkPos = world.ChunkPos;
const BlockPos = world.BlockPos;
const Face = world.Face;
const Mesh = world.mesher.Mesh;
const Volume = world.mesher.Volume;

const ChunkManager = @This();

pub const Config = struct {
    seed: u64,
    /// Horizontal meshing radius in chunks, Euclidean on (x, z). Chunks are
    /// generated up to `radius + 1` so every meshed chunk has its 6 neighbours.
    radius: u16 = 16,
    /// Chunks are unloaded beyond `radius + unload_margin` (hysteresis).
    unload_margin: u16 = 2,
    gen_workers: usize,
    mesh_workers: usize,
};

pub const Upload = struct { pos: ChunkPos, mesh: Mesh };

pub const Stats = struct {
    loaded: usize = 0,
    /// Generation jobs sent and not yet received back.
    gen_queued: usize = 0,
    /// Mesh jobs sent and not yet received back.
    mesh_in_flight: usize = 0,
    mesh_dispatched: u64 = 0,
    uploads: u64 = 0,
};

/// The sender allocates `chunk`; ownership travels with the job and comes back in the result.
const GenJob = struct { id: u64, pos: ChunkPos, seed: u64, chunk: *Chunk };
const GenResult = struct { id: u64, pos: ChunkPos, chunk: *Chunk };
/// The mesh task owns and frees `volume`: nobody needs the snapshot afterwards.
const MeshJob = struct { id: u64, pos: ChunkPos, version: u32, volume: *Volume };
/// `mesh == null`: meshing failed (OOM), the chunk goes back to dirty.
const MeshResult = struct { id: u64, pos: ChunkPos, version: u32, mesh: ?Mesh };

const GenPool = threading.WorkerPool(GenJob, GenResult);
const MeshPool = threading.WorkerPool(MeshJob, MeshResult);
const WorkQueue = threading.WorkQueue;

const Entry = struct {
    /// Unique per load: detects results for a chunk unloaded then reloaded.
    id: u64,
    /// The chunk is owned by the gen job while generating, by the entry afterwards.
    state: union(enum) { generating, generated: *Chunk },
    /// Starts at 1 so that `displayed_version == 0` means "nothing displayed".
    version: u32 = 1,
    dirty: bool = true,
    mesh_in_flight: bool = false,
    displayed_version: u32 = 0,
};

gpa: Allocator,
io: Io,
config: Config,

gen_in: WorkQueue(GenJob) = .empty,
gen_out: WorkQueue(GenResult) = .empty,
mesh_in: WorkQueue(MeshJob) = .empty,
mesh_out: WorkQueue(MeshResult) = .empty,
gen_pool: GenPool,
mesh_pool: MeshPool,

entries: std.AutoHashMapUnmanaged(ChunkPos, Entry) = .empty,
// ponytail: grows without bound (one 32 KiB copy per edited chunk); spill to disk if it matters.
edited: std.AutoHashMapUnmanaged(ChunkPos, *Chunk) = .empty,
uploads: std.ArrayList(Upload) = .empty,
unloads: std.ArrayList(ChunkPos) = .empty,
next_id: u64 = 0,
/// Center column of the last completed streaming pass; null before the first one.
last_center: ?[2]i32 = null,
stats: Stats = .{},

/// Heap-allocated because the pools keep pointers to the queues.
/// `gpa` must be thread-safe: workers allocate and free with it.
pub fn create(gpa: Allocator, io: Io, config: Config) !*ChunkManager {
    const self = try gpa.create(ChunkManager);
    errdefer gpa.destroy(self);
    self.* = .{
        .gpa = gpa,
        .io = io,
        .config = config,
        .gen_pool = .init(&self.gen_in, &self.gen_out, config.gen_workers, genTask),
        .mesh_pool = .init(&self.mesh_in, &self.mesh_out, config.mesh_workers, meshTask),
    };
    // Nothing is queued yet: shutting down only joins the workers already spawned.
    errdefer self.gen_pool.shutdown(gpa, io);
    try self.gen_pool.start(gpa, io);
    errdefer self.mesh_pool.shutdown(gpa, io);
    try self.mesh_pool.start(gpa, io);
    return self;
}

pub fn destroy(self: *ChunkManager) void {
    const gpa = self.gpa;
    const io = self.io;
    // Queued jobs own memory: pop and free them instead of letting the pools drain them.
    self.gen_in.close(io);
    self.mesh_in.close(io);
    while (popClosed(GenJob, &self.gen_in, io)) |job| gpa.destroy(job.chunk);
    while (popClosed(MeshJob, &self.mesh_in, io)) |job| gpa.destroy(job.volume);
    // Output queues stay open until here: a worker pushing into a closed one leaks.
    self.gen_pool.shutdown(gpa, io);
    self.mesh_pool.shutdown(gpa, io);
    self.gen_out.close(io);
    self.mesh_out.close(io);
    while (popClosed(GenResult, &self.gen_out, io)) |r| gpa.destroy(r.chunk);
    while (popClosed(MeshResult, &self.mesh_out, io)) |r| if (r.mesh) |m| gpa.free(m.quads);

    self.gen_in.deinit(gpa);
    self.gen_out.deinit(gpa);
    self.mesh_in.deinit(gpa);
    self.mesh_out.deinit(gpa);
    var it = self.entries.valueIterator();
    while (it.next()) |e| switch (e.state) {
        .generating => {},
        .generated => |c| gpa.destroy(c),
    };
    self.entries.deinit(gpa);
    var edited_it = self.edited.valueIterator();
    while (edited_it.next()) |c| gpa.destroy(c.*);
    self.edited.deinit(gpa);
    self.freeUploads();
    self.uploads.deinit(gpa);
    self.unloads.deinit(gpa);
    gpa.destroy(self);
}

/// Pops from a closed queue, spinning past lock contention. Null once empty.
fn popClosed(comptime T: type, queue: *WorkQueue(T), io: Io) ?T {
    while (true) {
        if (queue.pop(io) catch return null) |item| return item;
    }
}

/// Once per frame, main thread. Invalidates the slices of `takeUploads` / `takeUnloads`.
pub fn update(self: *ChunkManager, center: BlockPos) !void {
    self.freeUploads();
    self.unloads.clearRetainingCapacity();

    while (try self.gen_out.pop(self.io)) |r| self.handleGenResult(r);
    while (try self.mesh_out.pop(self.io)) |r| try self.handleMeshResult(r);
    try self.dispatchMeshes();

    // The desired set only depends on the center column, so the streaming
    // pass runs when it changes. Recorded last: a failed pass is retried.
    const c = center.chunk();
    const column: [2]i32 = .{ c.x, c.z };
    if (self.last_center == null or !std.meta.eql(self.last_center.?, column)) {
        try self.enqueueMissing(c);
        try self.unloadFar(column);
        self.last_center = column;
    }
    self.stats.loaded = self.entries.count();
}

/// Meshes ready for the GPU. Valid until the next `update`, which frees them.
/// A mesh with 0 quads still means "replace": free the old allocation.
/// Apply uploads before unloads: a chunk may appear in both in the same frame.
pub fn takeUploads(self: *const ChunkManager) []const Upload {
    return self.uploads.items;
}

/// Chunks unloaded this frame. Valid until the next `update`.
pub fn takeUnloads(self: *const ChunkManager) []const ChunkPos {
    return self.unloads.items;
}

/// Main thread. Sets `pos` to air if its chunk is loaded and the block is not air.
/// Returns whether a block changed.
pub fn breakBlock(self: *ChunkManager, pos: BlockPos) !bool {
    const cpos = pos.chunk();
    const chunk = self.loadedChunk(cpos) orelse return false;
    const local = pos.local();
    if (chunk.get(local) == .air) return false;

    // Allocate before mutating: on OOM the world is left untouched.
    const slot = try self.edited.getOrPut(self.gpa, cpos);
    if (!slot.found_existing) {
        slot.value_ptr.* = self.gpa.create(Chunk) catch |err| {
            self.edited.removeByPtr(slot.key_ptr);
            return err;
        };
    }
    chunk.set(local, .air);
    slot.value_ptr.*.* = chunk.*;
    markEdited(self.entries.getPtr(cpos).?);

    // A border block exposes a face of the neighbour: remesh it too.
    const l = [3]u5{ local.x, local.y, local.z };
    const faces = [3][2]Face{ .{ .neg_x, .pos_x }, .{ .neg_y, .pos_y }, .{ .neg_z, .pos_z } };
    for (l, faces) |v, f| {
        const face = switch (v) {
            0 => f[0],
            Chunk.size - 1 => f[1],
            else => continue,
        };
        if (self.entries.getPtr(neighbour(cpos, face))) |e| markEdited(e);
    }
    return true;
}

/// Main thread. Air when the chunk is not loaded. Makes `ChunkManager` a `raycast` lookup.
pub fn blockAt(self: *const ChunkManager, pos: BlockPos) world.Block {
    const chunk = self.loadedChunk(pos.chunk()) orelse return .air;
    return chunk.get(pos.local());
}

fn loadedChunk(self: *const ChunkManager, pos: ChunkPos) ?*Chunk {
    const e = self.entries.getPtr(pos) orelse return null;
    return switch (e.state) {
        .generating => null,
        .generated => |c| c,
    };
}

fn markEdited(e: *Entry) void {
    e.version += 1;
    e.dirty = true;
}

fn neighbour(pos: ChunkPos, face: Face) ChunkPos {
    const n = face.normal();
    return .{ .x = pos.x + n[0], .y = pos.y + n[1], .z = pos.z + n[2] };
}

fn freeUploads(self: *ChunkManager) void {
    for (self.uploads.items) |*u| u.mesh.deinit(self.gpa);
    self.uploads.clearRetainingCapacity();
}

fn handleGenResult(self: *ChunkManager, r: GenResult) void {
    self.stats.gen_queued -= 1;
    const e = self.entries.getPtr(r.pos) orelse return self.gpa.destroy(r.chunk);
    if (e.id != r.id) return self.gpa.destroy(r.chunk);
    e.state = .{ .generated = r.chunk };
}

fn handleMeshResult(self: *ChunkManager, r: MeshResult) !void {
    self.stats.mesh_in_flight -= 1;
    var mesh = r.mesh orelse {
        if (self.entries.getPtr(r.pos)) |e| if (e.id == r.id) {
            e.mesh_in_flight = false;
            e.dirty = true;
        };
        return;
    };
    errdefer mesh.deinit(self.gpa);
    const e = self.entries.getPtr(r.pos) orelse return mesh.deinit(self.gpa);
    if (e.id != r.id) return mesh.deinit(self.gpa);
    e.mesh_in_flight = false;
    if (r.version <= e.displayed_version) return mesh.deinit(self.gpa);
    try self.uploads.append(self.gpa, .{ .pos = r.pos, .mesh = mesh });
    e.displayed_version = r.version;
    self.stats.uploads += 1;
}

fn dispatchMeshes(self: *ChunkManager) !void {
    const center = self.last_center orelse return;
    const r: i64 = self.config.radius;
    var it = self.entries.iterator();
    // ponytail: scans every entry each frame; keep a dirty list if it shows in a profile.
    next: while (it.next()) |kv| {
        const pos = kv.key_ptr.*;
        const e = kv.value_ptr;
        if (!e.dirty or e.mesh_in_flight) continue;
        const chunk = switch (e.state) {
            .generating => continue,
            .generated => |c| c,
        };
        if (horizontalDist2(pos, center) > r * r) continue;
        // Never displayed and empty: nothing to draw. Once displayed, an
        // emptied chunk still sends a 0-quad mesh so the renderer frees it.
        if (e.displayed_version == 0 and chunk.isEmpty()) {
            e.dirty = false;
            continue;
        }
        var neighbours: [6]?*const Chunk = undefined;
        for (&neighbours, 0..) |*n, f| {
            const npos = neighbour(pos, @enumFromInt(f));
            n.* = if (!npos.inWorld()) null else self.loadedChunk(npos) orelse continue :next;
        }

        const volume = try self.gpa.create(Volume);
        errdefer self.gpa.destroy(volume);
        world.mesher.buildVolume(chunk, neighbours, volume);
        const job: MeshJob = .{ .id = e.id, .pos = pos, .version = e.version, .volume = volume };
        // Version > 1 means edited since load: the player is waiting, jump the queue.
        if (e.version > 1)
            try self.mesh_in.pushFront(self.gpa, self.io, job)
        else
            try self.mesh_in.push(self.gpa, self.io, job);
        e.dirty = false;
        e.mesh_in_flight = true;
        self.stats.mesh_in_flight += 1;
        self.stats.mesh_dispatched += 1;
    }
}

fn horizontalDist2(pos: ChunkPos, center: [2]i32) i64 {
    const dx: i64 = pos.x - center[0];
    const dz: i64 = pos.z - center[1];
    return dx * dx + dz * dz;
}

/// Loads every chunk within `radius + 1` of `center` (all cy), nearest first.
fn enqueueMissing(self: *ChunkManager, center: ChunkPos) !void {
    const gpa = self.gpa;
    const r: i32 = @as(i32, self.config.radius) + 1;
    const column: [2]i32 = .{ center.x, center.z };

    var missing: std.ArrayList(ChunkPos) = .empty;
    defer missing.deinit(gpa);
    var dz: i32 = -r;
    while (dz <= r) : (dz += 1) {
        var dx: i32 = -r;
        while (dx <= r) : (dx += 1) {
            for (0..world.height_chunks) |cy| {
                const pos: ChunkPos = .{ .x = center.x + dx, .y = @intCast(cy), .z = center.z + dz };
                if (horizontalDist2(pos, column) > r * r or self.entries.contains(pos)) continue;
                try missing.append(gpa, pos);
            }
        }
    }
    const Near = struct {
        c: ChunkPos,
        fn dist2(ctx: @This(), p: ChunkPos) i64 {
            const dy: i64 = p.y - ctx.c.y;
            return horizontalDist2(p, .{ ctx.c.x, ctx.c.z }) + dy * dy;
        }
        fn lessThan(ctx: @This(), a: ChunkPos, b: ChunkPos) bool {
            return ctx.dist2(a) < ctx.dist2(b);
        }
    };
    std.mem.sort(ChunkPos, missing.items, Near{ .c = center }, Near.lessThan);

    // ponytail: jobs queued for an area the camera left still run before newer
    // ones; re-prioritise gen_in if fast travel makes that visible.
    for (missing.items) |pos| {
        try self.entries.ensureUnusedCapacity(gpa, 1);
        const id = self.next_id;
        self.next_id += 1;
        if (self.edited.get(pos)) |saved| {
            const chunk = try gpa.create(Chunk);
            chunk.* = saved.*;
            self.entries.putAssumeCapacityNoClobber(pos, .{ .id = id, .state = .{ .generated = chunk } });
            continue;
        }
        const chunk = try gpa.create(Chunk);
        errdefer gpa.destroy(chunk);
        try self.gen_in.push(gpa, self.io, .{ .id = id, .pos = pos, .seed = self.config.seed, .chunk = chunk });
        // Capacity reserved above: nothing can fail once the job owns the chunk.
        self.entries.putAssumeCapacityNoClobber(pos, .{ .id = id, .state = .generating });
        self.stats.gen_queued += 1;
    }
}

/// Unloads chunks beyond `radius + unload_margin` of `center` and records them in `unloads`.
fn unloadFar(self: *ChunkManager, center: [2]i32) !void {
    const r: i64 = @as(i64, self.config.radius) + self.config.unload_margin;
    var it = self.entries.keyIterator();
    while (it.next()) |pos| {
        if (horizontalDist2(pos.*, center) > r * r) try self.unloads.append(self.gpa, pos.*);
    }
    // A generating chunk is freed when its result comes back with an unknown id;
    // an in-flight mesh likewise.
    for (self.unloads.items) |pos| {
        const e = self.entries.fetchRemove(pos).?.value;
        switch (e.state) {
            .generating => {},
            .generated => |c| self.gpa.destroy(c),
        }
    }
}

/// Builds the generator in the task: `Generator.init` is a few field stores, and it
/// keeps the job a plain value with no pointer into the manager.
fn genTask(_: Allocator, _: Io, job: GenJob) anyerror!?GenResult {
    const generator: world.Generator = .init(job.seed);
    world.generate(&generator, job.pos, job.chunk);
    return .{ .id = job.id, .pos = job.pos, .chunk = job.chunk };
}

/// Never returns an error: the pool would drop the result and leak what it owns.
fn meshTask(gpa: Allocator, _: Io, job: MeshJob) anyerror!?MeshResult {
    defer gpa.destroy(job.volume);
    const mesh = world.mesher.mesh(gpa, job.volume) catch null;
    return .{ .id = job.id, .pos = job.pos, .version = job.version, .mesh = mesh };
}

// ---
// Tests
// ---

const testing = std.testing;

const PosSet = std.AutoHashMapUnmanaged(ChunkPos, void);

const test_config: Config = .{ .seed = 42, .radius = 2, .unload_margin = 2, .gen_workers = 2, .mesh_workers = 2 };

fn createTest(config: Config) !*ChunkManager {
    return create(testing.allocator, testing.io, config);
}

fn blockPos(cx: i32, cz: i32) BlockPos {
    return .{ .x = cx * Chunk.size + 16, .y = 100, .z = cz * Chunk.size + 16 };
}

/// Pumps `update` until nothing is generating or meshing, recording every
/// upload and unload position seen on the way. Fails after ~10 s.
fn settle(cm: *ChunkManager, center: BlockPos, uploaded: ?*PosSet, unloaded: ?*PosSet) !void {
    for (0..10_000) |_| {
        try cm.update(center);
        if (uploaded) |set| for (cm.takeUploads()) |u| try set.put(testing.allocator, u.pos, {});
        if (unloaded) |set| for (cm.takeUnloads()) |p| try set.put(testing.allocator, p, {});
        if (cm.stats.gen_queued == 0 and cm.stats.mesh_in_flight == 0) return;
        try testing.io.sleep(.fromMilliseconds(1), .awake);
    }
    return error.Timeout;
}

/// First non-air block of chunk `c` whose local coordinates satisfy `pred`.
fn findSolid(cm: *const ChunkManager, c: ChunkPos, comptime pred: fn (world.LocalPos) bool) ?BlockPos {
    const o = c.origin();
    for (0..Chunk.size) |y| for (0..Chunk.size) |z| for (0..Chunk.size) |x| {
        const l: world.LocalPos = .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) };
        const p: BlockPos = .{ .x = o.x + l.x, .y = o.y + l.y, .z = o.z + l.z };
        if (pred(l) and cm.blockAt(p) != .air) return p;
    };
    return null;
}

fn interior(l: world.LocalPos) bool {
    for ([_]u5{ l.x, l.y, l.z }) |v| if (v == 0 or v == Chunk.size - 1) return false;
    return true;
}

test "generates radius + 1, meshes every non-empty chunk within radius" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    var uploaded: PosSet = .empty;
    defer uploaded.deinit(testing.allocator);
    try settle(cm, blockPos(0, 0), &uploaded, null);

    const r: i32 = test_config.radius;
    var dx: i32 = -r - 2;
    while (dx <= r + 2) : (dx += 1) {
        var dz: i32 = -r - 2;
        while (dz <= r + 2) : (dz += 1) {
            const d2 = dx * dx + dz * dz;
            for (0..world.height_chunks) |cy| {
                const pos: ChunkPos = .{ .x = dx, .y = @intCast(cy), .z = dz };
                const chunk = cm.loadedChunk(pos);
                try testing.expectEqual(d2 <= (r + 1) * (r + 1), chunk != null);
                const expect_upload = d2 <= r * r and !chunk.?.isEmpty();
                try testing.expectEqual(expect_upload, uploaded.contains(pos));
            }
        }
    }
    try testing.expectEqual(cm.entries.count(), cm.stats.loaded);
}

test "unload hysteresis" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    var unloaded: PosSet = .empty;
    defer unloaded.deinit(testing.allocator);
    try settle(cm, blockPos(0, 0), null, &unloaded);

    // Moving by margin - 1 keeps every chunk within radius + margin.
    try settle(cm, blockPos(1, 0), null, &unloaded);
    try testing.expectEqual(0, unloaded.count());

    var before: PosSet = .empty;
    defer before.deinit(testing.allocator);
    var it = cm.entries.keyIterator();
    while (it.next()) |p| try before.put(testing.allocator, p.*, {});

    try cm.update(blockPos(100, 0));
    try testing.expectEqual(before.count(), cm.takeUnloads().len);
    for (cm.takeUnloads()) |p| try testing.expect(before.contains(p));
}

test "breakBlock and blockAt" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    try settle(cm, blockPos(0, 0), null, null);

    // Unloaded, and air (terrain tops out well below y = 250).
    try testing.expect(!try cm.breakBlock(.{ .x = 10_000, .y = 50, .z = 0 }));
    try testing.expect(!try cm.breakBlock(.{ .x = 0, .y = 250, .z = 0 }));

    // A border block (local x = 31) of chunk (0, 0, 0): its +X neighbour is remeshed too.
    const p = findSolid(cm, .{ .x = 0, .y = 0, .z = 0 }, struct {
        fn f(l: world.LocalPos) bool {
            return l.x == Chunk.size - 1 and l.y > 0 and l.y < Chunk.size - 1 and l.z > 0 and l.z < Chunk.size - 1;
        }
    }.f).?;
    const n: ChunkPos = .{ .x = 1, .y = 0, .z = 0 };
    const n_version = cm.entries.get(n).?.version;
    try testing.expect(try cm.breakBlock(p));
    try testing.expectEqual(world.Block.air, cm.blockAt(p));
    try testing.expect(!try cm.breakBlock(p));
    try testing.expectEqual(n_version + 1, cm.entries.get(n).?.version);

    var uploaded: PosSet = .empty;
    defer uploaded.deinit(testing.allocator);
    try settle(cm, blockPos(0, 0), &uploaded, null);
    try testing.expectEqual(2, uploaded.count());
    try testing.expect(uploaded.contains(p.chunk()));
    try testing.expect(uploaded.contains(n));
}

test "N edits during a mesh give exactly one remesh" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    const center = blockPos(0, 0);
    try settle(cm, center, null, null);

    const c: ChunkPos = .{ .x = 0, .y = 0, .z = 0 };
    const before = cm.stats.mesh_dispatched;
    try testing.expect(try cm.breakBlock(findSolid(cm, c, interior).?));
    try cm.update(center);
    try testing.expectEqual(before + 1, cm.stats.mesh_dispatched);
    for (0..5) |_| try testing.expect(try cm.breakBlock(findSolid(cm, c, interior).?));
    try settle(cm, center, null, null);
    // Either the first result is still in flight at the next update (the chunk
    // stays dirty until the result clears mesh_in_flight), or it already came
    // back (step 3 clears mesh_in_flight before step 4 dispatches). In both
    // cases the 5 edits share a single dirty flag, hence a single dispatch.
    try testing.expectEqual(before + 2, cm.stats.mesh_dispatched);
}

test "mesh result older than the displayed one is dropped" {
    const cm = try createTest(test_config);
    defer cm.destroy();
    try settle(cm, blockPos(0, 0), null, null);
    const pos: ChunkPos = .{ .x = 0, .y = 0, .z = 0 };
    const e = cm.entries.getPtr(pos).?;
    e.displayed_version = 5;

    const Case = struct { version: u32, id: u64, uploaded: bool };
    for ([_]Case{
        .{ .version = 3, .id = e.id, .uploaded = false },
        .{ .version = 5, .id = e.id, .uploaded = false },
        .{ .version = 6, .id = e.id + 1_000_000, .uploaded = false },
        .{ .version = 6, .id = e.id, .uploaded = true },
    }) |case| {
        cm.freeUploads();
        cm.stats.mesh_in_flight += 1; // as if dispatched
        const mesh: Mesh = .{ .quads = try testing.allocator.alloc(world.mesher.Quad, 1), .counts = @splat(0) };
        try cm.handleMeshResult(.{ .id = case.id, .pos = pos, .version = case.version, .mesh = mesh });
        try testing.expectEqual(@intFromBool(case.uploaded), cm.takeUploads().len);
    }
    try testing.expectEqual(6, e.displayed_version);
}

test "edited chunk survives unload and reload" {
    const cm = try createTest(.{ .seed = 42, .radius = 1, .unload_margin = 1, .gen_workers = 2, .mesh_workers = 2 });
    defer cm.destroy();
    try settle(cm, blockPos(0, 0), null, null);
    const c: ChunkPos = .{ .x = 0, .y = 0, .z = 0 };
    const p = findSolid(cm, c, interior).?;
    try testing.expect(try cm.breakBlock(p));

    try settle(cm, blockPos(50, 0), null, null);
    try testing.expect(!cm.entries.contains(c));
    try settle(cm, blockPos(0, 0), null, null);
    try testing.expectEqual(world.Block.air, cm.blockAt(p));
    try testing.expectEqualSlices(world.Block, &cm.edited.get(c).?.blocks, &cm.loadedChunk(c).?.blocks);
}

test "destroy with jobs still queued leaks nothing" {
    const cm = try createTest(.{ .seed = 7, .radius = 6, .gen_workers = 1, .mesh_workers = 1 });
    defer cm.destroy();
    for (0..3) |_| {
        try cm.update(blockPos(0, 0));
        try testing.io.sleep(.fromMilliseconds(5), .awake);
    }
    try testing.expect(cm.stats.gen_queued > 0);
}
```

Add `_ = @import("ChunkManager.zig");` to the `test { ... }` block of `src/main.zig`.

- [ ] **Step 2: Run the tests three times (they use real threads)**

Run: `for i in 1 2 3; do zig build test --summary all 2>&1 | grep "Build Summary"; done`
Expected: three lines, each reporting all tests passed. `std.testing.allocator` reports any leak, including jobs still queued at `destroy`.

- [ ] **Step 3: Commit**

```bash
zig fmt --check src build.zig
git add src/ChunkManager.zig src/main.zig
git commit -m "feat: ChunkManager streaming with generation and meshing worker pools

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: GPU-driven chunks on screen

**Files:**
- Create: `src/render/Buffer.zig`, `src/render/gpu.zig`, `src/render/ChunkBuffers.zig`
- Create: `src/render/shaders/gpu.glsl`, `src/render/shaders/cull.comp`, `src/render/shaders/chunk.vert`, `src/render/shaders/chunk.frag`
- Modify: `src/render/Renderer.zig` (full replacement), `build.zig` (full replacement), `src/main.zig` (full replacement)

**Interfaces:**
- Consumes: everything from Tasks 8–10; `world.FreeList` (zero-length ranges are no-ops), `world.Block.color`, `world.mesher.Quad`.
- Produces: `Buffer.init(ctx, size, usage, host_visible)` (always device-addressable), `Buffer.slice(T)`; `gpu.{FrameData, ChunkMeta, DrawCmd, Push, frustumPlanes}`; `ChunkBuffers.upload(pos, quads, counts) !void`, `remove(pos) !void`, `record(cmd, staging)`, `chunkCount()`, field `quad_count`; `Renderer.chunks: ChunkBuffers`; `FrameInput { view_proj, camera_pos, sun_dir, sun_color, ambient, fog_start, fog_end }`.

How a frame works (all in one command buffer):
1. `ChunkBuffers.record`: a barrier orders every earlier culling/indirect/vertex read before this frame's transfer writes (so freed quad ranges and slots are reused at once); pending meshes are copied into the frame's staging buffer up to 16 MiB and copied to the quad buffer; changed `ChunkMeta` entries are written with `vkCmdUpdateBuffer`.
2. The draw counter is cleared; barrier; `cull.comp` runs one invocation per slot: frustum test against the chunk's AABB, then one `VkDrawIndirectCommand` per face direction that can face the camera, with `firstInstance = slot * 8 + face`.
3. Barrier; the sky is drawn; then one `vkCmdDrawIndirectCount` draws every visible chunk face group. `chunk.vert` finds its quad from `gl_VertexIndex / 6` and its chunk from `gl_InstanceIndex >> 3`, through device addresses in the push constants.

`gpu.glsl` and `gpu.zig` must stay byte-compatible (scalar layout): change them together.

- [ ] **Step 1: Write the GPU data definitions and their test**

Create `src/render/gpu.zig`:

```zig
//! CPU mirrors of the GPU structures declared in shaders/gpu.glsl (scalar layout).
const zm = @import("zmath");

pub const FrameData = extern struct {
    view_proj: zm.Mat,
    planes: [6][4]f32,
    camera_pos: [4]f32,
    sun_dir: [4]f32,
    sun_color: [4]f32,
    ambient: [4]f32,
    fog: [4]f32,
    palette: [8][4]f32,
    chunk_capacity: u32,
};

pub const ChunkMeta = extern struct {
    origin: [3]i32,
    first_quad: u32,
    counts: [6]u32,
    enabled: u32,
};

pub const DrawCmd = extern struct {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

/// Push constants of every chunk pipeline: device addresses only.
pub const Push = extern struct {
    frame: u64,
    metas: u64,
    quads: u64,
    draws: u64,
    count: u64,
};

/// Frustum planes (normal pointing inside, xyz·p + w >= 0 inside) of a
/// row-vector view-projection matrix with reverse-Z depth in [0, 1].
pub fn frustumPlanes(vp: zm.Mat) [6][4]f32 {
    // Columns of the row-vector matrix are the rows of the column-vector one.
    const m = zm.transpose(vp);
    const raw = [6]zm.F32x4{
        m[3] + m[0], // left
        m[3] - m[0], // right
        m[3] + m[1], // top/bottom (Y flipped, both kept)
        m[3] - m[1],
        m[3] - m[2], // near (reverse-Z: depth <= w)
        m[2], // far at infinity: depth >= 0
    };
    var out: [6][4]f32 = undefined;
    for (raw, &out) |p, *o| {
        const len = @sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        o.* = if (len > 0) .{ p[0] / len, p[1] / len, p[2] / len, p[3] / len } else .{ 0, 0, 0, 1 };
    }
    return out;
}

const std = @import("std");
const Camera = @import("../Camera.zig");

test "frustum planes keep what is in front and reject what is behind" {
    const c: Camera = .{ .pos = .{ 0, 0, 0 }, .yaw = 0, .pitch = 0 };
    const planes = frustumPlanes(c.viewProj(16.0 / 9.0));
    const inside = struct {
        fn f(ps: [6][4]f32, p: [3]f32) bool {
            for (ps) |pl| if (pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3] < 0) return false;
            return true;
        }
    }.f;
    try std.testing.expect(inside(planes, .{ 0, 0, -10 })); // straight ahead
    try std.testing.expect(inside(planes, .{ 0, 0, -1e5 })); // far away: infinite far plane
    try std.testing.expect(!inside(planes, .{ 0, 0, 10 })); // behind
    try std.testing.expect(!inside(planes, .{ 100, 0, -10 })); // far to the right
    try std.testing.expect(!inside(planes, .{ 0, 100, -10 })); // far above
}
```

Create `src/render/shaders/gpu.glsl`:

```glsl
// GPU data shared by the chunk pipelines (cull, chunk, shadow).
#include "common.glsl"

#extension GL_EXT_buffer_reference : require

// Mirrors render/gpu.zig. All buffers are reached through device addresses.
layout(buffer_reference, scalar) readonly buffer FrameData {
    mat4 view_proj;
    vec4 planes[6];      // camera frustum, xyz normal pointing inside, w distance
    vec4 camera_pos;
    vec4 sun_dir;        // towards the sun
    vec4 sun_color;
    vec4 ambient;
    vec4 fog;            // x: start, y: end (blocks)
    vec4 palette[8];     // block albedo, indexed by Block
    uint chunk_capacity;
};

struct ChunkMeta {
    ivec3 origin;        // world block coordinates of the chunk's min corner
    uint first_quad;
    uint counts[6];      // quads per face, in Face order
    uint enabled;
};
layout(buffer_reference, scalar) readonly buffer Metas { ChunkMeta m[]; };
layout(buffer_reference, scalar) readonly buffer Quads { uvec2 q[]; };

struct DrawCmd { uint vertex_count; uint instance_count; uint first_vertex; uint first_instance; };
layout(buffer_reference, scalar) writeonly buffer Draws { DrawCmd d[]; };
layout(buffer_reference, scalar) buffer Count { uint n; };

// One push-constant block for every chunk pipeline.
layout(push_constant, scalar) uniform Push {
    FrameData frame;
    Metas metas;
    Quads quads;
    Draws draws;
    Count count;
} pc;

const vec3 face_normals[6] = vec3[6](
    vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));
```

- [ ] **Step 2: Write the culling and chunk shaders**

Create `src/render/shaders/cull.comp`:

```glsl
#version 460
#extension GL_GOOGLE_include_directive : require
#include "gpu.glsl"

// One invocation per chunk slot: frustum test, then one indirect draw per
// face direction that can face the camera.
layout(local_size_x = 64) in;

void main() {
    uint slot = gl_GlobalInvocationID.x;
    if (slot >= pc.frame.chunk_capacity) return;
    ChunkMeta m = pc.metas.m[slot];
    if (m.enabled == 0) return;

    vec3 lo = vec3(m.origin);
    vec3 hi = lo + 32.0;
    for (int i = 0; i < 6; i++) {
        vec4 p = pc.frame.planes[i];
        vec3 v = mix(lo, hi, greaterThan(p.xyz, vec3(0))); // corner furthest along the normal
        if (dot(p.xyz, v) + p.w < 0.0) return;
    }

    vec3 cam = pc.frame.camera_pos.xyz;
    // A +X face lies at x >= lo.x + 1, so it can only be seen from x > lo.x (and so on).
    bool visible[6] = bool[6](cam.x > lo.x, cam.x < hi.x, cam.y > lo.y, cam.y < hi.y, cam.z > lo.z, cam.z < hi.z);
    uint first = m.first_quad;
    for (uint f = 0; f < 6; f++) {
        uint n = m.counts[f];
        if (n != 0 && visible[f]) {
            uint i = atomicAdd(pc.count.n, 1);
            pc.draws.d[i] = DrawCmd(n * 6, 1, first * 6, slot * 8 + f);
        }
        first += n;
    }
}
```

Create `src/render/shaders/chunk.vert`:

```glsl
#version 460
#extension GL_GOOGLE_include_directive : require
#include "gpu.glsl"

layout(location = 0) out vec3 world_pos;
layout(location = 1) flat out uint face;
layout(location = 2) flat out uint block;

// Quad (u, v) corners of the two triangles, counter-clockwise seen from the front.
const vec2 corners[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));

void main() {
    uvec2 q = pc.quads.q[gl_VertexIndex / 6];
    uvec3 p = uvec3(q.x & 63u, (q.x >> 6) & 63u, (q.x >> 12) & 63u);
    vec2 size = vec2((q.x >> 18) & 63u, (q.x >> 24) & 63u);
    face = (q.x >> 30) | ((q.y & 1u) << 2);
    block = (q.y >> 1) & 255u;
    ChunkMeta m = pc.metas.m[gl_InstanceIndex >> 3];

    // Width/height axes: ±X -> (z, y), ±Y -> (x, z), ±Z -> (x, y).
    uint axis = face >> 1;
    bool positive = (face & 1u) == 0u;
    vec3 u = axis == 0u ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 v = axis == 1u ? vec3(0, 0, 1) : vec3(0, 1, 0);
    // cross(u, v) is -normal for ±X/±Y positive faces, +normal for +Z: swap to keep CCW.
    vec2 c = corners[gl_VertexIndex % 6];
    bool flip = (axis == 2u) != positive;
    if (flip) c = c.yx;

    vec3 base = vec3(p) + (positive ? face_normals[face] : vec3(0));
    world_pos = vec3(m.origin) + base + u * (c.x * size.x) + v * (c.y * size.y);
    gl_Position = pc.frame.view_proj * vec4(world_pos, 1.0);
}
```

Create `src/render/shaders/chunk.frag`:

```glsl
#version 460
#extension GL_GOOGLE_include_directive : require
#include "gpu.glsl"

layout(location = 0) in vec3 world_pos;
layout(location = 1) flat in uint face;
layout(location = 2) flat in uint block;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 n = face_normals[face];
    vec3 sun = pc.frame.sun_dir.xyz;
    vec3 albedo = pc.frame.palette[block].rgb;

    float ndl = max(dot(n, sun), 0.0);
    // Hemispheric ambient: brighter from the sky than from the ground.
    vec3 ambient = pc.frame.ambient.rgb * mix(0.5, 1.0, n.y * 0.5 + 0.5);
    vec3 color = albedo * (pc.frame.sun_color.rgb * ndl + ambient);

    vec3 to_frag = world_pos - pc.frame.camera_pos.xyz;
    float fog = smoothstep(pc.frame.fog.x, pc.frame.fog.y, length(to_frag));
    color = mix(color, skyColor(normalize(to_frag), sun), fog);
    out_color = vec4(tonemap(color), 1.0);
}
```

- [ ] **Step 3: Write the buffers**

Create `src/render/Buffer.zig`:

```zig
//! A buffer with its own memory allocation, device address, and a persistent
//! mapping when host visible.
const std = @import("std");
const vk = @import("vulkan");
const Context = @import("Context.zig");

const Buffer = @This();

handle: vk.Buffer,
memory: vk.DeviceMemory,
size: vk.DeviceSize,
address: vk.DeviceAddress,
mapped: ?[*]u8,

pub fn init(ctx: *const Context, size: vk.DeviceSize, usage: vk.BufferUsageFlags, host_visible: bool) !Buffer {
    var u = usage;
    u.shader_device_address_bit = true;
    const handle = try ctx.device.createBuffer(&.{ .size = size, .usage = u, .sharing_mode = .exclusive }, null);
    errdefer ctx.device.destroyBuffer(handle, null);
    const req = ctx.device.getBufferMemoryRequirements(handle);
    const flags: vk.MemoryPropertyFlags = if (host_visible)
        .{ .host_visible_bit = true, .host_coherent_bit = true }
    else
        .{ .device_local_bit = true };
    const alloc_flags: vk.MemoryAllocateFlagsInfo = .{ .flags = .{ .device_address_bit = true }, .device_mask = 0 };
    const memory = try ctx.device.allocateMemory(&.{
        .p_next = &alloc_flags,
        .allocation_size = req.size,
        .memory_type_index = try ctx.findMemoryType(req.memory_type_bits, flags),
    }, null);
    errdefer ctx.device.freeMemory(memory, null);
    try ctx.device.bindBufferMemory(handle, memory, 0);
    const mapped: ?[*]u8 = if (host_visible) @ptrCast(try ctx.device.mapMemory(memory, 0, vk.WHOLE_SIZE, .{})) else null;
    return .{
        .handle = handle,
        .memory = memory,
        .size = size,
        .address = ctx.device.getBufferDeviceAddress(&.{ .buffer = handle }),
        .mapped = mapped,
    };
}

pub fn deinit(self: *Buffer, ctx: *const Context) void {
    ctx.device.destroyBuffer(self.handle, null);
    ctx.device.freeMemory(self.memory, null);
    self.* = undefined;
}

/// Typed view of a host-visible buffer.
pub fn slice(self: *const Buffer, comptime T: type) []T {
    const bytes = self.mapped.?[0..self.size];
    return @alignCast(std.mem.bytesAsSlice(T, bytes));
}
```

Create `src/render/ChunkBuffers.zig`:

```zig
//! GPU storage of every chunk mesh: one big quad buffer sub-allocated by a
//! FreeList, and a metadata buffer indexed by slot. Uploads are staged and
//! recorded into the frame's command buffer, within a per-frame byte budget.
//!
//! Synchronization: `record` starts with a barrier ordering every earlier
//! GPU read (culling, indirect draw, vertex pulling) before this frame's
//! transfer writes, so freed ranges and slots can be reused immediately.
const std = @import("std");
const vk = @import("vulkan");
const world = @import("world");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");
const Buffer = @import("Buffer.zig");
const gpu = @import("gpu.zig");

const ChunkBuffers = @This();
const Quad = world.mesher.Quad;

pub const max_quads = 16 * 1024 * 1024; // 128 MiB
pub const max_chunks = 16 * 1024;
/// Staging bytes per frame in flight; larger backlogs spill over to the next frames.
pub const staging_size = 16 * 1024 * 1024;

const Slot = struct { index: u32, range: world.FreeList.Range };
const Pending = struct { pos: world.ChunkPos, quads: []Quad, counts: [6]u32 };

gpa: Allocator,
quads: Buffer,
metas: Buffer,
ranges: world.FreeList,
slots: std.AutoHashMapUnmanaged(world.ChunkPos, Slot) = .empty,
free_slots: std.ArrayList(u32) = .empty,
/// One past the highest slot index ever used: the culling dispatch size.
slot_high: u32 = 0,
pending: std.ArrayList(Pending) = .empty,
/// Metadata writes to record this frame, keyed by slot.
meta_writes: std.AutoArrayHashMapUnmanaged(u32, gpu.ChunkMeta) = .empty,
quad_count: u64 = 0,

pub fn init(ctx: *const Context, gpa: Allocator) !ChunkBuffers {
    var quads: Buffer = try .init(ctx, max_quads * @sizeOf(Quad), .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, false);
    errdefer quads.deinit(ctx);
    var metas: Buffer = try .init(ctx, max_chunks * @sizeOf(gpu.ChunkMeta), .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, false);
    errdefer metas.deinit(ctx);
    return .{ .gpa = gpa, .quads = quads, .metas = metas, .ranges = try .init(gpa, max_quads) };
}

pub fn deinit(self: *ChunkBuffers, ctx: *const Context) void {
    for (self.pending.items) |p| self.gpa.free(p.quads);
    self.pending.deinit(self.gpa);
    self.meta_writes.deinit(self.gpa);
    self.free_slots.deinit(self.gpa);
    self.slots.deinit(self.gpa);
    self.ranges.deinit(self.gpa);
    self.metas.deinit(ctx);
    self.quads.deinit(ctx);
    self.* = undefined;
}

/// Queues `quads` (copied) as the new mesh of `pos`, replacing any older one.
pub fn upload(self: *ChunkBuffers, pos: world.ChunkPos, quads: []const Quad, counts: [6]u32) !void {
    self.dropPending(pos);
    const copy = try self.gpa.dupe(Quad, quads);
    errdefer self.gpa.free(copy);
    try self.pending.append(self.gpa, .{ .pos = pos, .quads = copy, .counts = counts });
}

/// Forgets the mesh of `pos` (pending or resident).
pub fn remove(self: *ChunkBuffers, pos: world.ChunkPos) !void {
    self.dropPending(pos);
    try self.release(pos);
}

fn dropPending(self: *ChunkBuffers, pos: world.ChunkPos) void {
    var i: usize = 0;
    while (i < self.pending.items.len) {
        if (std.meta.eql(self.pending.items[i].pos, pos)) {
            self.gpa.free(self.pending.items[i].quads);
            _ = self.pending.orderedRemove(i);
        } else i += 1;
    }
}

fn release(self: *ChunkBuffers, pos: world.ChunkPos) !void {
    const kv = self.slots.fetchRemove(pos) orelse return;
    try self.ranges.free(self.gpa, kv.value.range);
    self.quad_count -= kv.value.range.len;
    try self.free_slots.append(self.gpa, kv.value.index);
    try self.meta_writes.put(self.gpa, kv.value.index, std.mem.zeroes(gpu.ChunkMeta));
}

/// Records this frame's transfers: pending meshes that fit in `staging`, then
/// metadata updates. Ends with a barrier making them visible to culling and drawing.
pub fn record(self: *ChunkBuffers, cmd: vk.CommandBufferProxy, staging: *const Buffer) !void {
    barrier(cmd, .{ .compute_shader_bit = true, .vertex_shader_bit = true, .draw_indirect_bit = true }, .{ .shader_storage_read_bit = true, .indirect_command_read_bit = true }, .{ .copy_bit = true, .clear_bit = true }, .{ .transfer_write_bit = true });

    const staged = staging.slice(Quad);
    var used: usize = 0;
    var copies: std.ArrayList(vk.BufferCopy) = .empty;
    defer copies.deinit(self.gpa);
    var done: usize = 0;
    for (self.pending.items) |p| {
        if (used + p.quads.len > staged.len) break;
        try self.release(p.pos);
        done += 1;
        if (p.quads.len == 0) continue;
        const range = self.ranges.alloc(@intCast(p.quads.len)) orelse {
            std.log.scoped(.render).warn("quad buffer full, dropping chunk {any}", .{p.pos});
            continue;
        };
        const index = self.free_slots.pop() orelse blk: {
            if (self.slot_high == max_chunks) {
                try self.ranges.free(self.gpa, range);
                std.log.scoped(.render).warn("chunk slots full, dropping chunk {any}", .{p.pos});
                continue;
            }
            self.slot_high += 1;
            break :blk self.slot_high - 1;
        };
        @memcpy(staged[used..][0..p.quads.len], p.quads);
        try copies.append(self.gpa, .{ .src_offset = used * @sizeOf(Quad), .dst_offset = @as(u64, range.offset) * @sizeOf(Quad), .size = p.quads.len * @sizeOf(Quad) });
        used += p.quads.len;
        try self.slots.put(self.gpa, p.pos, .{ .index = index, .range = range });
        self.quad_count += range.len;
        const o = p.pos.origin();
        try self.meta_writes.put(self.gpa, index, .{ .origin = .{ o.x, o.y, o.z }, .first_quad = range.offset, .counts = p.counts, .enabled = 1 });
    }
    for (self.pending.items[0..done]) |p| self.gpa.free(p.quads);
    self.pending.replaceRangeAssumeCapacity(0, done, &.{});

    if (copies.items.len > 0) cmd.copyBuffer(staging.handle, self.quads.handle, copies.items);
    var it = self.meta_writes.iterator();
    while (it.next()) |e| cmd.updateBuffer(self.metas.handle, @as(u64, e.key_ptr.*) * @sizeOf(gpu.ChunkMeta), @sizeOf(gpu.ChunkMeta), e.value_ptr);
    self.meta_writes.clearRetainingCapacity();
}

pub fn barrier(cmd: vk.CommandBufferProxy, src_stage: vk.PipelineStageFlags2, src_access: vk.AccessFlags2, dst_stage: vk.PipelineStageFlags2, dst_access: vk.AccessFlags2) void {
    const b: vk.MemoryBarrier2 = .{ .src_stage_mask = src_stage, .src_access_mask = src_access, .dst_stage_mask = dst_stage, .dst_access_mask = dst_access };
    cmd.pipelineBarrier2(&.{ .memory_barrier_count = 1, .p_memory_barriers = @ptrCast(&b) });
}

pub fn chunkCount(self: *const ChunkBuffers) u32 {
    return self.slots.count();
}
```

- [ ] **Step 4: Extend the renderer**

Replace `src/render/Renderer.zig` with:

```zig
//! Frame loop: frames in flight, depth buffer, passes.
const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");
const Swapchain = @import("Swapchain.zig");
const pipeline = @import("pipeline.zig");
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const ChunkBuffers = @import("ChunkBuffers.zig");
const gpu = @import("gpu.zig");
const world = @import("world");

const Renderer = @This();

pub const frames_in_flight = 2;
const depth_format: vk.Format = .d32_sfloat;

const Frame = struct {
    pool: vk.CommandPool,
    cmd: vk.CommandBuffer,
    image_acquired: vk.Semaphore,
    fence: vk.Fence,
    /// Written by the CPU while the frame is recorded, read by the GPU.
    frame_data: Buffer,
    staging: Buffer,
};

pub const FrameInput = struct {
    view_proj: zm.Mat,
    camera_pos: [3]f32,
    sun_dir: [3]f32,
    sun_color: [3]f32,
    ambient: [3]f32,
    fog_start: f32,
    fog_end: f32,
};

gpa: Allocator,
ctx: *const Context,
swapchain: Swapchain,
depth: Image,
frames: [frames_in_flight]Frame,
frame_index: usize = 0,
sky_layout: vk.PipelineLayout,
sky_pipeline: vk.Pipeline,
chunks: ChunkBuffers,
draws: Buffer,
draw_count: Buffer,
chunk_layout: vk.PipelineLayout,
cull_pipeline: vk.Pipeline,
chunk_pipeline: vk.Pipeline,

pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D) !Renderer {
    var swapchain: Swapchain = try .init(ctx, gpa, extent, .null_handle);
    errdefer swapchain.deinit(ctx, gpa);
    var depth: Image = try .initDepth(ctx, swapchain.extent, depth_format);
    errdefer depth.deinit(ctx);

    var frames: [frames_in_flight]Frame = undefined;
    for (&frames) |*f| {
        f.pool = try ctx.device.createCommandPool(&.{ .flags = .{ .reset_command_buffer_bit = true }, .queue_family_index = ctx.queue_family }, null);
        try ctx.device.allocateCommandBuffers(&.{ .command_pool = f.pool, .level = .primary, .command_buffer_count = 1 }, @ptrCast(&f.cmd));
        f.image_acquired = try ctx.device.createSemaphore(&.{}, null);
        f.fence = try ctx.device.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
        f.frame_data = try .init(ctx, @sizeOf(gpu.FrameData), .{ .storage_buffer_bit = true }, true);
        f.staging = try .init(ctx, ChunkBuffers.staging_size, .{ .transfer_src_bit = true }, true);
    }

    var chunks: ChunkBuffers = try .init(ctx, gpa);
    errdefer chunks.deinit(ctx);
    const max_draws = ChunkBuffers.max_chunks * 6;
    var draws: Buffer = try .init(ctx, max_draws * @sizeOf(gpu.DrawCmd), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true }, false);
    errdefer draws.deinit(ctx);
    var draw_count: Buffer = try .init(ctx, 4, .{ .storage_buffer_bit = true, .indirect_buffer_bit = true, .transfer_dst_bit = true }, false);
    errdefer draw_count.deinit(ctx);
    const chunk_layout = try pipeline.createLayout(ctx, @sizeOf(gpu.Push), .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true }, &.{});
    const cull_pipeline = try pipeline.createCompute(ctx, chunk_layout, pipeline.spirv("cull.comp"));

    const sky_layout = try pipeline.createLayout(ctx, @sizeOf(SkyPush), .{ .fragment_bit = true }, &.{});
    const sky_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = sky_layout,
        .vertex = pipeline.spirv("fullscreen.vert"),
        .fragment = pipeline.spirv("sky.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
        .depth_test = false,
        .depth_write = false,
        .cull_back = false,
    });
    const chunk_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = chunk_layout,
        .vertex = pipeline.spirv("chunk.vert"),
        .fragment = pipeline.spirv("chunk.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
    });
    return .{
        .gpa = gpa,
        .ctx = ctx,
        .swapchain = swapchain,
        .depth = depth,
        .frames = frames,
        .sky_layout = sky_layout,
        .sky_pipeline = sky_pipeline,
        .chunks = chunks,
        .draws = draws,
        .draw_count = draw_count,
        .chunk_layout = chunk_layout,
        .cull_pipeline = cull_pipeline,
        .chunk_pipeline = chunk_pipeline,
    };
}

pub fn deinit(self: *Renderer) void {
    const d = self.ctx.device;
    d.deviceWaitIdle() catch {};
    d.destroyPipeline(self.chunk_pipeline, null);
    d.destroyPipeline(self.cull_pipeline, null);
    d.destroyPipelineLayout(self.chunk_layout, null);
    self.draw_count.deinit(self.ctx);
    self.draws.deinit(self.ctx);
    self.chunks.deinit(self.ctx);
    d.destroyPipeline(self.sky_pipeline, null);
    d.destroyPipelineLayout(self.sky_layout, null);
    for (&self.frames) |*f| {
        f.staging.deinit(self.ctx);
        f.frame_data.deinit(self.ctx);
        d.destroyFence(f.fence, null);
        d.destroySemaphore(f.image_acquired, null);
        d.destroyCommandPool(f.pool, null);
    }
    self.depth.deinit(self.ctx);
    self.swapchain.deinit(self.ctx, self.gpa);
}

const SkyPush = extern struct {
    inv_view_proj: zm.Mat,
    sun_dir: [4]f32,
};

/// Renders one frame. `extent` is the current framebuffer size (for resizes).
pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
    const d = self.ctx.device;
    const frame = &self.frames[self.frame_index];
    _ = try d.waitForFences(&.{frame.fence}, .true, std.math.maxInt(u64));

    const acquired = d.acquireNextImageKHR(self.swapchain.handle, std.math.maxInt(u64), frame.image_acquired, .null_handle) catch |err| switch (err) {
        error.OutOfDateKHR => return self.recreate(extent),
        else => return err,
    };
    try d.resetFences(&.{frame.fence});
    const image_index = acquired.image_index;

    const cmd: vk.CommandBufferProxy = .init(frame.cmd, self.ctx.vkd);
    try cmd.resetCommandBuffer(.{});
    try cmd.beginCommandBuffer(&.{ .flags = .{ .one_time_submit_bit = true } });

    // Chunk uploads, then GPU culling into the indirect draw buffer.
    self.writeFrameData(frame, in);
    const push: gpu.Push = .{
        .frame = frame.frame_data.address,
        .metas = self.chunks.metas.address,
        .quads = self.chunks.quads.address,
        .draws = self.draws.address,
        .count = self.draw_count.address,
    };
    try self.chunks.record(cmd, &frame.staging);
    cmd.fillBuffer(self.draw_count.handle, 0, 4, 0);
    ChunkBuffers.barrier(cmd, .{ .copy_bit = true, .clear_bit = true }, .{ .transfer_write_bit = true }, .{ .compute_shader_bit = true, .vertex_shader_bit = true }, .{ .shader_storage_read_bit = true, .shader_storage_write_bit = true });
    cmd.bindPipeline(.compute, self.cull_pipeline);
    cmd.pushConstants(self.chunk_layout, .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(gpu.Push), &push);
    cmd.dispatch(std.math.divCeil(u32, self.chunks.slot_high, 64) catch unreachable, 1, 1);
    ChunkBuffers.barrier(cmd, .{ .compute_shader_bit = true }, .{ .shader_storage_write_bit = true }, .{ .draw_indirect_bit = true }, .{ .indirect_command_read_bit = true });

    const image = self.swapchain.images[image_index];
    imageBarrier(cmd, image, .{ .color_bit = true }, .undefined, .color_attachment_optimal, .{ .color_attachment_output_bit = true }, .{}, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true });
    imageBarrier(cmd, self.depth.image, .{ .depth_bit = true }, .undefined, .depth_attachment_optimal, .{ .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .early_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true });

    const ext = self.swapchain.extent;
    const color_att: vk.RenderingAttachmentInfo = .{
        .image_view = self.swapchain.views[image_index],
        .image_layout = .color_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .dont_care,
        .store_op = .store,
        .clear_value = .{ .color = .{ .float_32 = .{ 0, 0, 0, 1 } } },
    };
    const depth_att: vk.RenderingAttachmentInfo = .{
        .image_view = self.depth.view,
        .image_layout = .depth_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined,
        .load_op = .clear,
        .store_op = .dont_care,
        .clear_value = .{ .depth_stencil = .{ .depth = 0, .stencil = 0 } },
    };
    cmd.beginRendering(&.{
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = ext },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_att),
        .p_depth_attachment = &depth_att,
    });
    setViewport(cmd, ext);

    const sky: SkyPush = .{ .inv_view_proj = zm.inverse(in.view_proj), .sun_dir = .{ in.sun_dir[0], in.sun_dir[1], in.sun_dir[2], 0 } };
    cmd.bindPipeline(.graphics, self.sky_pipeline);
    cmd.pushConstants(self.sky_layout, .{ .fragment_bit = true }, 0, @sizeOf(SkyPush), &sky);
    cmd.draw(3, 1, 0, 0);

    cmd.bindPipeline(.graphics, self.chunk_pipeline);
    cmd.pushConstants(self.chunk_layout, .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(gpu.Push), &push);
    cmd.drawIndirectCount(self.draws.handle, 0, self.draw_count.handle, 0, ChunkBuffers.max_chunks * 6, @sizeOf(gpu.DrawCmd));

    cmd.endRendering();
    imageBarrier(cmd, image, .{ .color_bit = true }, .color_attachment_optimal, .present_src_khr, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true }, .{}, .{});
    try cmd.endCommandBuffer();

    const render_done = self.swapchain.render_done[image_index];
    try self.ctx.queue.submit2(&.{.{
        .wait_semaphore_info_count = 1,
        .p_wait_semaphore_infos = &.{.{ .semaphore = frame.image_acquired, .value = 0, .stage_mask = .{ .color_attachment_output_bit = true }, .device_index = 0 }},
        .command_buffer_info_count = 1,
        .p_command_buffer_infos = &.{.{ .command_buffer = frame.cmd, .device_mask = 0 }},
        .signal_semaphore_info_count = 1,
        .p_signal_semaphore_infos = &.{.{ .semaphore = render_done, .value = 0, .stage_mask = .{ .all_commands_bit = true }, .device_index = 0 }},
    }}, frame.fence);

    self.frame_index = (self.frame_index + 1) % frames_in_flight;
    const present = self.ctx.queue.presentKHR(&.{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&render_done),
        .swapchain_count = 1,
        .p_swapchains = @ptrCast(&self.swapchain.handle),
        .p_image_indices = @ptrCast(&image_index),
    }) catch |err| switch (err) {
        error.OutOfDateKHR => return self.recreate(extent),
        else => return err,
    };
    if (present == .suboptimal_khr or extent.width != self.swapchain.extent.width or extent.height != self.swapchain.extent.height)
        try self.recreate(extent);
}

fn writeFrameData(self: *Renderer, frame: *Frame, in: FrameInput) void {
    var palette: [8][4]f32 = undefined;
    for (&palette, 0..) |*c, i| {
        const rgb = @as(world.Block, @enumFromInt(i)).color();
        c.* = .{ rgb[0], rgb[1], rgb[2], 1 };
    }
    const data: *gpu.FrameData = @ptrCast(@alignCast(frame.frame_data.mapped.?));
    data.* = .{
        .view_proj = in.view_proj,
        .planes = gpu.frustumPlanes(in.view_proj),
        .camera_pos = .{ in.camera_pos[0], in.camera_pos[1], in.camera_pos[2], 1 },
        .sun_dir = .{ in.sun_dir[0], in.sun_dir[1], in.sun_dir[2], 0 },
        .sun_color = .{ in.sun_color[0], in.sun_color[1], in.sun_color[2], 0 },
        .ambient = .{ in.ambient[0], in.ambient[1], in.ambient[2], 0 },
        .fog = .{ in.fog_start, in.fog_end, 0, 0 },
        .palette = palette,
        .chunk_capacity = self.chunks.slot_high,
    };
}

fn recreate(self: *Renderer, extent: vk.Extent2D) !void {
    if (extent.width == 0 or extent.height == 0) return; // minimized
    try self.ctx.device.deviceWaitIdle();
    self.swapchain.deinitKeepHandle(self.ctx, self.gpa);
    self.swapchain = try .init(self.ctx, self.gpa, extent, self.swapchain.handle);
    self.depth.deinit(self.ctx);
    self.depth = try .initDepth(self.ctx, self.swapchain.extent, depth_format);
}

fn setViewport(cmd: vk.CommandBufferProxy, ext: vk.Extent2D) void {
    cmd.setViewport(0, &.{.{ .x = 0, .y = 0, .width = @floatFromInt(ext.width), .height = @floatFromInt(ext.height), .min_depth = 0, .max_depth = 1 }});
    cmd.setScissor(0, &.{.{ .offset = .{ .x = 0, .y = 0 }, .extent = ext }});
}

pub fn imageBarrier(
    cmd: vk.CommandBufferProxy,
    image: vk.Image,
    aspect: vk.ImageAspectFlags,
    old: vk.ImageLayout,
    new: vk.ImageLayout,
    src_stage: vk.PipelineStageFlags2,
    src_access: vk.AccessFlags2,
    dst_stage: vk.PipelineStageFlags2,
    dst_access: vk.AccessFlags2,
) void {
    const b: vk.ImageMemoryBarrier2 = .{
        .src_stage_mask = src_stage,
        .src_access_mask = src_access,
        .dst_stage_mask = dst_stage,
        .dst_access_mask = dst_access,
        .old_layout = old,
        .new_layout = new,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{ .aspect_mask = aspect, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = vk.REMAINING_ARRAY_LAYERS },
    };
    cmd.pipelineBarrier2(&.{ .image_memory_barrier_count = 1, .p_image_memory_barriers = @ptrCast(&b) });
}
```

- [ ] **Step 5: Add the new shaders to the build**

Replace `build.zig` with:

```zig
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
    const shaders = [_][]const u8{ "fullscreen.vert", "sky.frag", "cull.comp", "chunk.vert", "chunk.frag" };
    for (shaders) |name| {
        const glslc = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.4", "-O", "-o" });
        const spv = glslc.addOutputFileArg(b.fmt("{s}.spv", .{name}));
        glslc.addFileArg(b.path(b.fmt("src/render/shaders/{s}", .{name})));
        glslc.addFileInput(b.path("src/render/shaders/common.glsl"));
        glslc.addFileInput(b.path("src/render/shaders/gpu.glsl"));
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
```

- [ ] **Step 6: Wire streaming into the main loop**

Replace `src/main.zig` with:

```zig
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
    var renderer: Renderer = try .init(gpa, &ctx, framebufferExtent(window));
    defer renderer.deinit();

    const workers = @max(1, (std.Thread.getCpuCount() catch 2) - 1);
    const chunks = try ChunkManager.create(gpa, io, .{ .seed = 42, .gen_workers = workers, .mesh_workers = workers });
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
        const aspect = @as(f32, @floatFromInt(extent.width)) / @as(f32, @floatFromInt(extent.height));
        const light = sun.lighting();
        const far: f32 = @floatFromInt(@as(u32, 16) * world.chunk_size);
        try renderer.drawFrame(extent, .{
            .view_proj = camera.viewProj(aspect),
            .camera_pos = camera.pos,
            .sun_dir = light.dir,
            .sun_color = light.color,
            .ambient = light.ambient,
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
}
```

- [ ] **Step 7: Build, test, run**

Run: `zig build test --summary all`
Expected: all tests pass, including `frustum planes keep what is in front and reject what is behind`.

Run: `zig build && timeout 6 ./zig-out/bin/ft_vox 2>&1 | grep -v worker_pool`
Expected: exactly these two lines (the window stays open about 6 seconds), and no `error(vulkan)` or `warning(vulkan)` line:

```
info(vulkan): validation layers: on
info(vulkan): GPU: <your GPU name>
```

If `validation layers: off` is printed, install the Khronos validation layer (`vulkan-validation-layers` on Arch) and rerun: this check is meaningless without it.

Visually: terrain with grass, sand beaches, water, trees and caves streams in around the camera; the window title shows fps, chunk count and quad count (dev GPU: ~240 fps, ~3000 chunks, ~2.6 M quads at radius 16); W/A/S/D (Z/Q/S/D on AZERTY), Space, Ctrl move, Shift sprints, the mouse looks around, Escape quits; far terrain fades into the sky color.

- [ ] **Step 8: Commit**

```bash
zig fmt --check src build.zig
git add build.zig src/main.zig src/render
git commit -m "feat(render): GPU-driven chunk rendering with culling, indirect count and vertex pulling

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## After this plan

Next plan (jalons 6–8): cascaded shadow maps drawn through the same culling + indirect path from the sun's frustum, quality settings for integrated GPUs (render distance, shadow resolution) as command-line arguments, and block breaking (camera raycast through `ChunkManager.blockAt`, wireframe outline, `breakBlock`).
