# ft_vox — consolidation, shadows, settings, block breaking (jalons 6–9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the engine: fix what the reviews and the loading profile found, add cascaded shadow maps with the day/night cycle, command-line quality settings, and block breaking with an outline.

**Architecture:** CPU side, the ChunkManager caps the first meshes dispatched per frame and the worker pools leave a core to the render thread. GPU side, culling runs once per view (camera + 3 cascades) into per-view indirect ranges; each cascade is drawn depth-only through the same indirect path; the chunk fragment shader samples the cascade array through a push descriptor. Block breaking raycasts through `ChunkManager.blockAt` and draws a 24-vertex line outline.

**Tech Stack:** Zig 0.16.0, Vulkan 1.4 (vulkan-zig `zig-0.16-compat`), GLSL/glslc, zglfw, zmath. Spec: `docs/superpowers/specs/2026-09-23-voxel-engine-design.md` (revision r3).

**Provenance:** every code block and patch below comes from a prototype that was built, tested (82/82 tests at the end) and run on the dev GPU (AMD Radeon Renoir, RADV) with validation layers on and no validation message; each intermediate state (after Tasks 12, 13, 14, 15) was replayed from a clean checkout of `ai` at `fc34640`. Loading fps went from 1–4 to 110–140 (Debug build) with Task 12; shadows and block breaking were checked with `xdotool`-driven screenshots.

## Global Constraints

- Zig 0.16.0; run every command from the repo root `/home/lailouezzz/Documents/git/ft_vox`.
- Vulkan 1.4 as in the previous plan; shaders compiled with `glslc --target-env=vulkan1.4 -O`, embedded by file name, handed to pipelines through maintenance5.
- Reverse-Z everywhere, shadows included: depth 1 nearest (to the camera or to the sun), compare `GREATER_OR_EQUAL`, clear 0.
- `gpu.glsl` and `gpu.zig` stay byte-compatible (scalar layout); change them together.
- Worker threads: `n = cores - 1` in total, `mesh = max(1, n / 3)`, `gen = n - mesh` (spec r3; oversubscribing starves the render thread).
- `world` must not import `vulkan`, `zglfw` or `threading`.
- `zig fmt --check src build.zig` must pass before every commit; test logs at level `err` fail the test runner.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` (or the model that actually wrote the commit).

## File Structure

| File | Change |
|---|---|
| `src/world/mesher.zig` | `buildVolume` copies rows of 32 blocks |
| `src/ChunkManager.zig` | mesh dispatch budget, emptiness from the gen worker, OOM keeps the chunk dirty, `unload_margin >= 1`, stronger destroy test |
| `src/threading/worker_pool.zig` | a worker is listed only once its task is running |
| `src/Camera.zig` | doc comment: compare `GREATER_OR_EQUAL` |
| `src/main.zig` | worker split; then shadows input, settings, block breaking |
| `src/render/Context.zig` | wrappers allocated before their handles |
| `src/render/Swapchain.zig` | atomic rebuild (old swapchain only retired), `ZeroExtent`, errdefers |
| `src/render/pipeline.zig` | dead `Stage` type removed |
| `src/render/ChunkBuffers.zig` | allocate-before-release on replace, doc fix |
| `src/render/Shadows.zig` | new: cascade array, sampler, stable cascade fitting |
| `src/render/gpu.zig`, `shaders/gpu.glsl` | cascades, per-view counters, `view` push field |
| `src/render/shaders/pull.glsl` | new: vertex pulling shared by chunk and shadow vertex shaders |
| `src/render/shaders/cull.comp`, `chunk.vert`, `chunk.frag`, `shadow.vert` | per-view culling, shadow pass, PCF lighting |
| `src/render/Renderer.zig` | shadow passes, push descriptor, atomic `recreate`, errdefers; then outline |
| `src/Settings.zig` | new: command-line settings |
| `src/render/shaders/outline.vert`, `outline.frag` | new: block outline |
| `build.zig` | new shaders and includes |

---

### Task 12: Consolidation (CPU) and loading performance

**Files:**
- Modify: `src/world/mesher.zig`, `src/ChunkManager.zig`, `src/threading/worker_pool.zig`, `src/Camera.zig`, `src/main.zig` (all through one patch)

**Interfaces:**
- Produces: `ChunkManager.Config.max_mesh_dispatch: u16 = 64` (first meshes dispatched per `update`; edit remeshes are not limited); `GenResult.empty` (computed by the gen worker); `ChunkManager.create` asserts `unload_margin >= 1`.

Why: profiling the initial load (Debug build) showed 77 % of the main thread in `mesher.buildVolume` (one per mesh dispatch, hundreds per frame, block-by-block copies) and 1–4 fps; oversubscribed pools (7 gen + 7 mesh workers on 8 cores) also starved the render thread (10 fps even after the budget). Also folds in review findings: a failed `uploads.append` must leave the chunk dirty; a zero `unload_margin` would never mesh the edge ring; `WorkerPool.spawnOne` listed a worker before `io.concurrent` succeeded, so `shutdown` could await an undefined future.

- [ ] **Step 1: Apply the patch**

Save this as `/tmp/task12.patch`, then run `git apply /tmp/task12.patch`:

```diff
diff --git a/src/Camera.zig b/src/Camera.zig
index 71ea9e9..239b8e6 100644
--- a/src/Camera.zig
+++ b/src/Camera.zig
@@ -25,7 +25,7 @@ pub fn view(c: Camera) zm.Mat {
 }
 
 /// Infinite reverse-Z perspective for Vulkan (clip Y down, depth 1 at `near`, 0 at infinity).
-/// Depth test GREATER, clear depth 0.
+/// Depth test GREATER_OR_EQUAL, clear depth 0.
 pub fn projection(c: Camera, aspect: f32) zm.Mat {
     const f = 1 / @tan(c.fov_y / 2);
     return .{
diff --git a/src/ChunkManager.zig b/src/ChunkManager.zig
index 382ad4e..165d546 100644
--- a/src/ChunkManager.zig
+++ b/src/ChunkManager.zig
@@ -26,6 +26,9 @@ pub const Config = struct {
     unload_margin: u16 = 2,
     gen_workers: usize,
     mesh_workers: usize,
+    /// First meshes dispatched per `update` (each costs a 39 KB snapshot on the
+    /// main thread); the rest wait for the next frames. Edits are not limited.
+    max_mesh_dispatch: u16 = 64,
 };
 
 pub const Upload = struct { pos: ChunkPos, mesh: Mesh };
@@ -42,7 +45,7 @@ pub const Stats = struct {
 
 /// The sender allocates `chunk`; ownership travels with the job and comes back in the result.
 const GenJob = struct { id: u64, pos: ChunkPos, seed: u64, chunk: *Chunk };
-const GenResult = struct { id: u64, pos: ChunkPos, chunk: *Chunk };
+const GenResult = struct { id: u64, pos: ChunkPos, chunk: *Chunk, empty: bool };
 /// The mesh task owns and frees `volume`: nobody needs the snapshot afterwards.
 const MeshJob = struct { id: u64, pos: ChunkPos, version: u32, volume: *Volume };
 /// `mesh == null`: meshing failed (OOM), the chunk goes back to dirty.
@@ -61,6 +64,8 @@ const Entry = struct {
     version: u32 = 1,
     dirty: bool = true,
     mesh_in_flight: bool = false,
+    /// All air at load time (computed by the gen worker, not per frame).
+    empty: bool = false,
     displayed_version: u32 = 0,
 };
 
@@ -88,6 +93,9 @@ stats: Stats = .{},
 /// Heap-allocated because the pools keep pointers to the queues.
 /// `gpa` must be thread-safe: workers allocate and free with it.
 pub fn create(gpa: Allocator, io: Io, config: Config) !*ChunkManager {
+    // With no margin the radius + 1 ring would unload as soon as it loads, and
+    // chunks at the edge of the radius would never get their 6 neighbours.
+    std.debug.assert(config.unload_margin >= 1);
     const self = try gpa.create(ChunkManager);
     errdefer gpa.destroy(self);
     self.* = .{
@@ -119,7 +127,10 @@ pub fn destroy(self: *ChunkManager) void {
     self.gen_out.close(io);
     self.mesh_out.close(io);
     while (popClosed(GenResult, &self.gen_out, io)) |r| gpa.destroy(r.chunk);
-    while (popClosed(MeshResult, &self.mesh_out, io)) |r| if (r.mesh) |m| gpa.free(m.quads);
+    while (popClosed(MeshResult, &self.mesh_out, io)) |r| if (r.mesh) |m| {
+        var mesh = m;
+        mesh.deinit(gpa);
+    };
 
     self.gen_in.deinit(gpa);
     self.gen_out.deinit(gpa);
@@ -248,6 +259,7 @@ fn handleGenResult(self: *ChunkManager, r: GenResult) void {
     const e = self.entries.getPtr(r.pos) orelse return self.gpa.destroy(r.chunk);
     if (e.id != r.id) return self.gpa.destroy(r.chunk);
     e.state = .{ .generated = r.chunk };
+    e.empty = r.empty;
 }
 
 fn handleMeshResult(self: *ChunkManager, r: MeshResult) !void {
@@ -264,6 +276,8 @@ fn handleMeshResult(self: *ChunkManager, r: MeshResult) !void {
     if (e.id != r.id) return mesh.deinit(self.gpa);
     e.mesh_in_flight = false;
     if (r.version <= e.displayed_version) return mesh.deinit(self.gpa);
+    // On failure the mesh is dropped: remesh later instead of keeping a stale one.
+    errdefer e.dirty = true;
     try self.uploads.append(self.gpa, .{ .pos = r.pos, .mesh = mesh });
     e.displayed_version = r.version;
     self.stats.uploads += 1;
@@ -272,6 +286,7 @@ fn handleMeshResult(self: *ChunkManager, r: MeshResult) !void {
 fn dispatchMeshes(self: *ChunkManager) !void {
     const center = self.last_center orelse return;
     const r: i64 = self.config.radius;
+    var budget = self.config.max_mesh_dispatch;
     var it = self.entries.iterator();
     // ponytail: scans every entry each frame; keep a dirty list if it shows in a profile.
     next: while (it.next()) |kv| {
@@ -283,9 +298,11 @@ fn dispatchMeshes(self: *ChunkManager) !void {
             .generated => |c| c,
         };
         if (horizontalDist2(pos, center) > r * r) continue;
+        // Edits bypass the budget: the player is waiting on them.
+        if (e.version == 1 and budget == 0) continue;
         // Never displayed and empty: nothing to draw. Once displayed, an
         // emptied chunk still sends a 0-quad mesh so the renderer frees it.
-        if (e.displayed_version == 0 and chunk.isEmpty()) {
+        if (e.displayed_version == 0 and e.empty) {
             e.dirty = false;
             continue;
         }
@@ -306,6 +323,7 @@ fn dispatchMeshes(self: *ChunkManager) !void {
             try self.mesh_in.push(self.gpa, self.io, job);
         e.dirty = false;
         e.mesh_in_flight = true;
+        if (e.version == 1) budget -= 1;
         self.stats.mesh_in_flight += 1;
         self.stats.mesh_dispatched += 1;
     }
@@ -357,7 +375,7 @@ fn enqueueMissing(self: *ChunkManager, center: ChunkPos) !void {
         if (self.edited.get(pos)) |saved| {
             const chunk = try gpa.create(Chunk);
             chunk.* = saved.*;
-            self.entries.putAssumeCapacityNoClobber(pos, .{ .id = id, .state = .{ .generated = chunk } });
+            self.entries.putAssumeCapacityNoClobber(pos, .{ .id = id, .state = .{ .generated = chunk }, .empty = chunk.isEmpty() });
             continue;
         }
         const chunk = try gpa.create(Chunk);
@@ -392,7 +410,7 @@ fn unloadFar(self: *ChunkManager, center: [2]i32) !void {
 fn genTask(_: Allocator, _: Io, job: GenJob) anyerror!?GenResult {
     const generator: world.Generator = .init(job.seed);
     world.generate(&generator, job.pos, job.chunk);
-    return .{ .id = job.id, .pos = job.pos, .chunk = job.chunk };
+    return .{ .id = job.id, .pos = job.pos, .chunk = job.chunk, .empty = job.chunk.isEmpty() };
 }
 
 /// Never returns an error: the pool would drop the result and leak what it owns.
@@ -587,9 +605,12 @@ test "edited chunk survives unload and reload" {
 test "destroy with jobs still queued leaks nothing" {
     const cm = try createTest(.{ .seed = 7, .radius = 6, .gen_workers = 1, .mesh_workers = 1 });
     defer cm.destroy();
-    for (0..3) |_| {
+    // Pump until meshing has started, so both pools hold jobs at destroy time.
+    var i: usize = 0;
+    while (cm.stats.mesh_dispatched == 0 and i < 5000) : (i += 1) {
         try cm.update(blockPos(0, 0));
-        try testing.io.sleep(.fromMilliseconds(5), .awake);
+        try testing.io.sleep(.fromMilliseconds(1), .awake);
     }
+    try testing.expect(cm.stats.mesh_dispatched > 0);
     try testing.expect(cm.stats.gen_queued > 0);
 }
diff --git a/src/main.zig b/src/main.zig
index 06bec79..34278fc 100644
--- a/src/main.zig
+++ b/src/main.zig
@@ -33,8 +33,10 @@ pub fn main(init: std.process.Init) !void {
     var renderer: Renderer = try .init(gpa, &ctx, framebufferExtent(window));
     defer renderer.deinit();
 
-    const workers = @max(1, (std.Thread.getCpuCount() catch 2) - 1);
-    const chunks = try ChunkManager.create(gpa, io, .{ .seed = 42, .gen_workers = workers, .mesh_workers = workers });
+    // One core stays free for the render thread: oversubscribing starves it while loading.
+    const workers = @max(2, (std.Thread.getCpuCount() catch 3) - 1);
+    const mesh_workers = @max(1, workers / 3);
+    const chunks = try ChunkManager.create(gpa, io, .{ .seed = 42, .gen_workers = workers - mesh_workers, .mesh_workers = mesh_workers });
     defer chunks.destroy();
 
     var camera: Camera = .{};
diff --git a/src/threading/worker_pool.zig b/src/threading/worker_pool.zig
index ba60e4c..802a373 100644
--- a/src/threading/worker_pool.zig
+++ b/src/threading/worker_pool.zig
@@ -181,10 +181,12 @@ pub fn WorkerPool(comptime InType: type, comptime OutType: type) type {
         fn spawnOne(self: *Self, allocator: Allocator, io: Io) !void {
             const id = self.workers.items.len;
             const worker = try allocator.create(Worker);
+            errdefer allocator.destroy(worker);
             worker.* = .empty;
             worker.id = id;
-            self.workers.appendAssumeCapacity(worker);
+            // Listed only once running: shutdown awaits every listed worker's future.
             worker.future = try io.concurrent(Worker.entry, .{ worker, allocator, io, self.in_queue, self.out_queue, self.task_fn });
+            self.workers.appendAssumeCapacity(worker);
         }
 
         fn cleanEnded(self: *Self, allocator: Allocator, io: Io) void {
diff --git a/src/world/mesher.zig b/src/world/mesher.zig
index 5438e9c..4daf22a 100644
--- a/src/world/mesher.zig
+++ b/src/world/mesher.zig
@@ -45,24 +45,25 @@ pub const Mesh = struct {
 
 /// Builds the padded volume of `center`. Missing neighbors (null) count as air.
 /// Neighbor order follows `Face`: +X, -X, +Y, -Y, +Z, -Z.
+/// Runs on the main thread for every mesh job, so rows of 32 blocks are copied
+/// with memcpy wherever both layouts keep x contiguous.
 pub fn buildVolume(center: *const Chunk, neighbors: [6]?*const Chunk, out: *Volume) void {
     out.* = @splat(.air);
-    for (0..cs) |y| for (0..cs) |z| for (0..cs) |x| {
-        out[volumeIndex(x + 1, y + 1, z + 1)] = center.blocks[x + cs * z + cs * cs * y];
+    for (0..cs) |y| for (0..cs) |z| {
+        @memcpy(out[volumeIndex(1, y + 1, z + 1)..][0..cs], center.blocks[cs * z + cs * cs * y ..][0..cs]);
     };
-    for (0..cs) |a| for (0..cs) |b| {
-        const l = struct {
-            fn at(c: ?*const Chunk, x: usize, y: usize, z: usize) Block {
-                return if (c) |n| n.blocks[x + cs * z + cs * cs * y] else .air;
-            }
-        }.at;
-        // a, b run over the two axes of the shared face.
-        out[volumeIndex(padded - 1, a + 1, b + 1)] = l(neighbors[@intFromEnum(Face.pos_x)], 0, a, b);
-        out[volumeIndex(0, a + 1, b + 1)] = l(neighbors[@intFromEnum(Face.neg_x)], cs - 1, a, b);
-        out[volumeIndex(a + 1, padded - 1, b + 1)] = l(neighbors[@intFromEnum(Face.pos_y)], a, 0, b);
-        out[volumeIndex(a + 1, 0, b + 1)] = l(neighbors[@intFromEnum(Face.neg_y)], a, cs - 1, b);
-        out[volumeIndex(a + 1, b + 1, padded - 1)] = l(neighbors[@intFromEnum(Face.pos_z)], a, b, 0);
-        out[volumeIndex(a + 1, b + 1, 0)] = l(neighbors[@intFromEnum(Face.neg_z)], a, b, cs - 1);
+    // ±Y and ±Z faces: rows along x.
+    for (0..cs) |a| {
+        if (neighbors[@intFromEnum(Face.pos_y)]) |n| @memcpy(out[volumeIndex(1, padded - 1, a + 1)..][0..cs], n.blocks[cs * a ..][0..cs]);
+        if (neighbors[@intFromEnum(Face.neg_y)]) |n| @memcpy(out[volumeIndex(1, 0, a + 1)..][0..cs], n.blocks[cs * a + cs * cs * (cs - 1) ..][0..cs]);
+        if (neighbors[@intFromEnum(Face.pos_z)]) |n| @memcpy(out[volumeIndex(1, a + 1, padded - 1)..][0..cs], n.blocks[cs * cs * a ..][0..cs]);
+        if (neighbors[@intFromEnum(Face.neg_z)]) |n| @memcpy(out[volumeIndex(1, a + 1, 0)..][0..cs], n.blocks[cs * (cs - 1) + cs * cs * a ..][0..cs]);
+    }
+    // ±X faces: one block per row.
+    for (0..cs) |y| for (0..cs) |z| {
+        const i = cs * z + cs * cs * y;
+        if (neighbors[@intFromEnum(Face.pos_x)]) |n| out[volumeIndex(padded - 1, y + 1, z + 1)] = n.blocks[i];
+        if (neighbors[@intFromEnum(Face.neg_x)]) |n| out[volumeIndex(0, y + 1, z + 1)] = n.blocks[i + cs - 1];
     };
 }
 
```

Run: `git apply --check /tmp/task12.patch` first if you want a dry run; it must report nothing.

- [ ] **Step 2: Run the tests three times (threads)**

Run: `for i in 1 2 3; do zig build test --summary all 2>&1 | grep "Build Summary"; done`
Expected: three lines with all tests passed (77 at this point). The destroy test now waits until meshing has started, so both pools hold jobs at `destroy` time.

- [ ] **Step 3: Run and check loading**

Run: `zig build && timeout 7 ./zig-out/bin/ft_vox 2>&1 | grep -v worker_pool`
Expected: exactly these two lines and no `error(vulkan)` / `warning(vulkan)` line:

```
info(vulkan): validation layers: on
info(vulkan): GPU: <your GPU name>
```

Loading check (fps while the ~3000 chunks stream in; needs `xdotool`):

```bash
(timeout 10 ./zig-out/bin/ft_vox >/dev/null 2>&1 &); sleep 1
for i in 1 2 3 4 5 6; do sleep 1; xdotool getwindowname "$(xdotool search --name '^ft_vox' | head -1)" | sed 's/ | pos.*//'; done
```

Expected: every line above ~60 fps on the dev GPU (it was 1–4 fps before this task) while the chunk count climbs to ~3060.

- [ ] **Step 4: Commit**

```bash
zig fmt --check src build.zig
git add src/world/mesher.zig src/ChunkManager.zig src/threading/worker_pool.zig src/Camera.zig src/main.zig
git commit -m "perf: mesh dispatch budget, row-copy snapshots, worker split; review fixes

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Renderer consolidation and cascaded shadow maps

**Files:**
- Create: `src/render/Shadows.zig`, `src/render/shaders/pull.glsl`, `src/render/shaders/shadow.vert`
- Replace: `src/render/Context.zig`, `src/render/Swapchain.zig`, `src/render/pipeline.zig`, `src/render/ChunkBuffers.zig`, `src/render/gpu.zig`, `src/render/Renderer.zig`, `src/render/shaders/gpu.glsl`, `src/render/shaders/cull.comp`, `src/render/shaders/chunk.vert`, `src/render/shaders/chunk.frag`, `build.zig`, `src/main.zig`

**Interfaces:**
- Consumes: `Camera` (`pos`, `forward()`, `fov_y`, `near`, `viewProj`), `Sun` (`lighting()`, `direction()`), `ChunkBuffers`, `pipeline`, `Image`, `Buffer` from the previous plan.
- Produces: `Shadows { resolution, image, layer_views, sampler }`, `Shadows.init(ctx, resolution)`, `Shadows.fitCascades(resolution, camera_pos, camera_forward, fov_y, aspect, near, sun_dir) [3]Cascade` (`Cascade = { view_proj, texel }`), `Shadows.cascades = 3`, `Shadows.splits = .{ 20, 64, 180 }`; `Renderer.init(gpa, ctx, extent, Options{ shadow_resolution = 2048 })`; `Renderer.FrameInput = { camera: Camera, light: Sun.Lighting, shadows: bool, fog_start, fog_end }` (the renderer now builds the view-projection itself from the swapchain aspect).

How a frame works now:
1. Chunk uploads are recorded first (slots created this frame are culled this frame), then `FrameData` is written: camera data, 3 cascade matrices + frustum planes + texel sizes, `shadow.x = 1` by day.
2. `cull.comp` runs once per view: view 0 = camera (faces that can face the camera), views 1–3 = cascades (faces turned towards the light), each into its own range of the indirect buffer (`view * max_draws`) and its own counter.
3. Each cascade layer is cleared and, by day, drawn depth-only with `shadow.vert` (depth clamp, negative depth bias for reverse-Z) through `vkCmdDrawIndirectCount` on its range. The layers are cleared even at night because the lighting pass samples them.
4. The main pass pushes the shadow array with `vkCmdPushDescriptorSet` (no pool, no sets), then draws the sky and the chunks; `chunk.frag` picks the cascade by distance, offsets along the normal by ~1.5 texels and filters 3×3 with a `GREATER_OR_EQUAL` comparison sampler.

Review fixes folded in: `recreate` builds the new swapchain and depth image before touching the current ones (a failure no longer leaves freed objects for `deinit`); the old swapchain is only retired by `Swapchain.init` and destroyed by the caller; a 0×0 surface returns `error.ZeroExtent` and is skipped; the size check compares against the last requested extent (no rebuild loop when the surface clamps); depth barriers cover early and late fragment tests; every init step has its `errdefer`; Vulkan wrappers are allocated before the handles they load; when the quad buffer is full the previous mesh of a chunk stays visible; the dead `pipeline.Stage` type is gone.

- [ ] **Step 1: Write the shadow module (tests included)**

Create `src/render/Shadows.zig`:

```zig
//! Cascaded shadow maps: a depth array (one layer per cascade), per-layer
//! views for rendering, an array view + comparison sampler for lighting, and
//! stable cascade fitting (bounding spheres, texel-snapped).
const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const Context = @import("Context.zig");
const Image = @import("Image.zig");

const Shadows = @This();

pub const cascades = 3;
pub const format: vk.Format = .d32_sfloat;
/// Far distance (blocks from the camera) covered by each cascade.
pub const splits = [cascades]f32{ 20, 64, 180 };
/// Extra depth range towards the sun so casters outside a cascade still cast.
const caster_margin = 256;

resolution: u32,
image: Image,
layer_views: [cascades]vk.ImageView,
sampler: vk.Sampler,

pub fn init(ctx: *const Context, resolution: u32) !Shadows {
    var image: Image = try .init(ctx, .{ .width = resolution, .height = resolution }, format, .{ .depth_stencil_attachment_bit = true, .sampled_bit = true }, .{ .depth_bit = true }, cascades);
    errdefer image.deinit(ctx);
    var layer_views: [cascades]vk.ImageView = undefined;
    for (&layer_views, 0..) |*v, i| {
        v.* = try ctx.device.createImageView(&.{
            .image = image.image,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{ .aspect_mask = .{ .depth_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = @intCast(i), .layer_count = 1 },
        }, null);
    }
    // Reverse-Z: a fragment is lit when its depth >= the stored occluder depth.
    // Outside the map the border (depth 0) leaves everything lit.
    const sampler = try ctx.device.createSampler(&.{
        .mag_filter = .linear,
        .min_filter = .linear,
        .mipmap_mode = .nearest,
        .address_mode_u = .clamp_to_border,
        .address_mode_v = .clamp_to_border,
        .address_mode_w = .clamp_to_border,
        .mip_lod_bias = 0,
        .anisotropy_enable = .false,
        .max_anisotropy = 1,
        .compare_enable = .true,
        .compare_op = .greater_or_equal,
        .min_lod = 0,
        .max_lod = 0,
        .border_color = .float_opaque_black,
        .unnormalized_coordinates = .false,
    }, null);
    return .{ .resolution = resolution, .image = image, .layer_views = layer_views, .sampler = sampler };
}

pub fn deinit(self: *Shadows, ctx: *const Context) void {
    ctx.device.destroySampler(self.sampler, null);
    for (self.layer_views) |v| ctx.device.destroyImageView(v, null);
    self.image.deinit(ctx);
    self.* = undefined;
}

pub const Cascade = struct {
    /// Light view-projection (row-vector zmath convention, reverse-Z orthographic, Vulkan clip Y down).
    view_proj: zm.Mat,
    /// World size of one shadow texel.
    texel: f32,
};

pub fn fitCascades(resolution: u32, camera_pos: [3]f32, camera_forward: [3]f32, fov_y: f32, aspect: f32, near: f32, sun_dir: [3]f32) [cascades]Cascade {
    const up = if (@abs(sun_dir[1]) > 0.99) zm.f32x4(0, 0, 1, 0) else zm.f32x4(0, 1, 0, 0);
    // Rotation only: the light "camera" sits at the origin looking down -sun_dir.
    const light_view = zm.lookToRh(zm.f32x4(0, 0, 0, 1), zm.f32x4(-sun_dir[0], -sun_dir[1], -sun_dir[2], 0), up);
    const tan_y = @tan(fov_y / 2);
    const tan_x = tan_y * aspect;
    var out: [cascades]Cascade = undefined;
    var d0: f32 = near;
    for (splits, 0..) |d1, i| {
        // Bounding sphere of the frustum slice [d0, d1]: its center lies on the
        // view axis; the radius depends only on the slice, so it is stable
        // under camera rotation (no shimmering).
        const k = tan_x * tan_x + tan_y * tan_y;
        const mid = @min(d1, 0.5 * (d0 + d1) * (1 + k));
        const r_far = @sqrt((d1 - mid) * (d1 - mid) + d1 * d1 * k);
        const r_near = @sqrt((mid - d0) * (mid - d0) + d0 * d0 * k);
        const radius = @ceil(@max(r_far, r_near));
        const center_ws = zm.f32x4(camera_pos[0] + camera_forward[0] * mid, camera_pos[1] + camera_forward[1] * mid, camera_pos[2] + camera_forward[2] * mid, 1);
        var c = zm.mul(center_ws, light_view);
        // Snap the center to the shadow texel grid.
        const texel = 2 * radius / @as(f32, @floatFromInt(resolution));
        c[0] = @floor(c[0] / texel) * texel;
        c[1] = @floor(c[1] / texel) * texel;
        const n = -c[2] - radius - caster_margin; // distances along -Z
        const f = -c[2] + radius;
        const s = 1 / radius;
        const proj: zm.Mat = .{
            zm.f32x4(s, 0, 0, 0),
            zm.f32x4(0, -s, 0, 0),
            zm.f32x4(0, 0, 1 / (f - n), 0),
            zm.f32x4(-c[0] * s, c[1] * s, f / (f - n), 1),
        };
        out[i] = .{ .view_proj = zm.mul(light_view, proj), .texel = texel };
        d0 = d1;
    }
    return out;
}

const testing = std.testing;

test "cascade covers the camera and maps it inside the depth range" {
    const cs = fitCascades(2048, .{ 10, 80, -5 }, .{ 0, 0, -1 }, std.math.degreesToRadians(70), 16.0 / 9.0, 0.1, .{ 0.3, 0.8, -0.2 });
    for (cs) |c| {
        const p = zm.mul(zm.f32x4(10, 80, -12, 1), c.view_proj); // a point just in front of the camera
        try testing.expect(@abs(p[0]) <= 1 and @abs(p[1]) <= 1);
        try testing.expect(p[2] >= 0 and p[2] <= 1);
    }
}

test "closer to the sun means larger depth (reverse-Z)" {
    const sun = [3]f32{ 0, 1, 0 };
    const cs = fitCascades(2048, .{ 0, 80, 0 }, .{ 0, 0, -1 }, 1.2, 1.5, 0.1, sun);
    const low = zm.mul(zm.f32x4(0, 70, -10, 1), cs[0].view_proj);
    const high = zm.mul(zm.f32x4(0, 90, -10, 1), cs[0].view_proj);
    try testing.expect(high[2] > low[2]);
}
```

- [ ] **Step 2: Update the GPU data definitions**

Replace `src/render/gpu.zig` with:

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
    cascade_vp: [3]zm.Mat,
    cascade_planes: [18][4]f32,
    cascade_splits: [4]f32,
    cascade_texel: [4]f32,
    shadow: [4]f32,
    chunk_capacity: u32,
    max_draws: u32,
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
    /// 0: camera, 1..3: shadow cascades.
    view: u32,
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

Replace `src/render/shaders/gpu.glsl` with:

```glsl
// GPU data shared by the chunk pipelines (cull, chunk, shadow).
#include "common.glsl"

#extension GL_EXT_buffer_reference : require

// Mirrors render/gpu.zig. All buffers are reached through device addresses.
layout(buffer_reference, scalar) readonly buffer FrameData {
    mat4 view_proj;
    vec4 planes[6];      // camera frustum, xyz normal pointing inside, w distance
    vec4 camera_pos;
    vec4 sun_dir;        // towards the light (sun by day, moon by night)
    vec4 sun_color;
    vec4 ambient;
    vec4 fog;            // x: start, y: end (blocks)
    vec4 palette[8];     // block albedo, indexed by Block
    mat4 cascade_vp[3];
    vec4 cascade_planes[18]; // 6 planes per cascade
    vec4 cascade_splits; // xyz: far distance of each cascade (blocks)
    vec4 cascade_texel;  // xyz: world size of one shadow texel per cascade
    vec4 shadow;         // x: 1 if shadows are on
    uint chunk_capacity;
    uint max_draws;      // indirect commands per view
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
layout(buffer_reference, scalar) buffer Count { uint n[4]; }; // one counter per view

// One push-constant block for every chunk pipeline.
layout(push_constant, scalar) uniform Push {
    FrameData frame;
    Metas metas;
    Quads quads;
    Draws draws;
    Count count;
    uint view;           // 0: camera, 1..3: shadow cascades
} pc;

const vec3 face_normals[6] = vec3[6](
    vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));
```

- [ ] **Step 3: Write the shaders**

Create `src/render/shaders/pull.glsl`:

```glsl
// Vertex pulling for the chunk vertex shaders (uses vertex-stage built-ins).
#include "gpu.glsl"

// Quad (u, v) corners of the two triangles.
const vec2 corners[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0), vec2(1, 1), vec2(0, 1));

// Vertex pulling: world position of this vertex, from gl_VertexIndex (quad and
// corner) and gl_InstanceIndex (slot * 8 + face).
vec3 pullVertex(out uint face, out uint block) {
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
    // (u, v) is counter-clockwise seen from the front only for +Z, -X and -Y
    // faces: swap the corner axes for the others.
    vec2 c = corners[gl_VertexIndex % 6];
    if ((axis == 2u) != positive) c = c.yx;

    vec3 base = vec3(p) + (positive ? face_normals[face] : vec3(0));
    return vec3(m.origin) + base + u * (c.x * size.x) + v * (c.y * size.y);
}
```

Replace `src/render/shaders/cull.comp` with:

```glsl
#version 460
#extension GL_GOOGLE_include_directive : require
#include "gpu.glsl"

// One invocation per chunk slot, for view pc.view: frustum test, then one
// indirect draw per face direction worth drawing for that view.
layout(local_size_x = 64) in;

void main() {
    uint slot = gl_GlobalInvocationID.x;
    if (slot >= pc.frame.chunk_capacity) return;
    ChunkMeta m = pc.metas.m[slot];
    if (m.enabled == 0) return;

    vec3 lo = vec3(m.origin);
    vec3 hi = lo + 32.0;
    for (uint i = 0; i < 6; i++) {
        vec4 p = pc.view == 0 ? pc.frame.planes[i] : pc.frame.cascade_planes[(pc.view - 1) * 6 + i];
        vec3 v = mix(lo, hi, greaterThan(p.xyz, vec3(0))); // corner furthest along the normal
        if (dot(p.xyz, v) + p.w < 0.0) return;
    }

    bool visible[6];
    if (pc.view == 0) {
        // A +X face lies at x >= lo.x + 1, so it can only be seen from x > lo.x (and so on).
        vec3 cam = pc.frame.camera_pos.xyz;
        visible = bool[6](cam.x > lo.x, cam.x < hi.x, cam.y > lo.y, cam.y < hi.y, cam.z > lo.z, cam.z < hi.z);
    } else {
        // Directional light: only faces turned towards it cast shadows.
        for (uint f = 0; f < 6; f++) visible[f] = dot(face_normals[f], pc.frame.sun_dir.xyz) > 0.0;
    }
    uint first = m.first_quad;
    uint base = pc.view * pc.frame.max_draws;
    for (uint f = 0; f < 6; f++) {
        uint n = m.counts[f];
        if (n != 0 && visible[f]) {
            uint i = atomicAdd(pc.count.n[pc.view], 1);
            pc.draws.d[base + i] = DrawCmd(n * 6, 1, first * 6, slot * 8 + f);
        }
        first += n;
    }
}
```

Replace `src/render/shaders/chunk.vert` with:

```glsl
#version 460
#extension GL_GOOGLE_include_directive : require
#include "pull.glsl"

layout(location = 0) out vec3 world_pos;
layout(location = 1) flat out uint face;
layout(location = 2) flat out uint block;

void main() {
    world_pos = pullVertex(face, block);
    gl_Position = pc.frame.view_proj * vec4(world_pos, 1.0);
}
```

Create `src/render/shaders/shadow.vert`:

```glsl
#version 460
#extension GL_GOOGLE_include_directive : require
#include "pull.glsl"

// Depth-only pass into one shadow cascade (pc.view = 1 + cascade index).
void main() {
    uint face, block;
    vec3 world_pos = pullVertex(face, block);
    gl_Position = pc.frame.cascade_vp[pc.view - 1] * vec4(world_pos, 1.0);
}
```

Replace `src/render/shaders/chunk.frag` with:

```glsl
#version 460
#extension GL_GOOGLE_include_directive : require
#include "gpu.glsl"

layout(set = 0, binding = 0) uniform sampler2DArrayShadow shadow_map;

layout(location = 0) in vec3 world_pos;
layout(location = 1) flat in uint face;
layout(location = 2) flat in uint block;
layout(location = 0) out vec4 out_color;

// 1 = fully lit, 0 = in shadow. PCF 3x3 on the cascade covering this fragment.
float sunVisibility(vec3 n, float dist) {
    if (pc.frame.shadow.x == 0.0) return 1.0;
    uint c = dist < pc.frame.cascade_splits.x ? 0u : dist < pc.frame.cascade_splits.y ? 1u : dist < pc.frame.cascade_splits.z ? 2u : 3u;
    if (c == 3u) return 1.0;
    // Normal offset: push the lookup out of the surface by ~1.5 texels.
    vec3 p = world_pos + n * pc.frame.cascade_texel[c] * 1.5;
    vec4 lp = pc.frame.cascade_vp[c] * vec4(p, 1.0);
    vec3 ndc = lp.xyz / lp.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0).xy);
    float lit = 0.0;
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++)
            lit += texture(shadow_map, vec4(uv + vec2(x, y) * texel, float(c), ndc.z));
    return lit / 9.0;
}

void main() {
    vec3 n = face_normals[face];
    vec3 sun = pc.frame.sun_dir.xyz;
    vec3 albedo = pc.frame.palette[block].rgb;
    vec3 to_frag = world_pos - pc.frame.camera_pos.xyz;
    float dist = length(to_frag);

    float ndl = max(dot(n, sun), 0.0);
    float lit = ndl > 0.0 ? sunVisibility(n, dist) : 0.0;
    // Hemispheric ambient: brighter from the sky than from the ground.
    vec3 ambient = pc.frame.ambient.rgb * mix(0.5, 1.0, n.y * 0.5 + 0.5);
    vec3 color = albedo * (pc.frame.sun_color.rgb * ndl * lit + ambient);

    float fog = smoothstep(pc.frame.fog.x, pc.frame.fog.y, dist);
    color = mix(color, skyColor(normalize(to_frag), sun), fog);
    out_color = vec4(tonemap(color), 1.0);
}
```

- [ ] **Step 4: Harden the Vulkan layer**

Replace `src/render/Context.zig` with:

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
    // Wrappers are allocated before the handles they load, so that no failure
    // can leave a live instance or device without the means to destroy it.
    self.vki = try gpa.create(vk.InstanceWrapper);
    errdefer gpa.destroy(self.vki);
    self.vkd = try gpa.create(vk.DeviceWrapper);
    errdefer gpa.destroy(self.vkd);

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

Replace `src/render/Swapchain.zig` with:

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
    if (actual.width == 0 or actual.height == 0) return error.ZeroExtent;
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

    const images = try ctx.device.getSwapchainImagesAllocKHR(handle, gpa);
    errdefer gpa.free(images);
    const views = try gpa.alloc(vk.ImageView, images.len);
    errdefer gpa.free(views);
    const render_done = try gpa.alloc(vk.Semaphore, images.len);
    errdefer gpa.free(render_done);
    var created: usize = 0;
    errdefer for (views[0..created], render_done[0..created]) |v, s| {
        ctx.device.destroyImageView(v, null);
        ctx.device.destroySemaphore(s, null);
    };
    for (images, views, render_done) |img, *view, *sem| {
        view.* = try ctx.device.createImageView(&.{
            .image = img,
            .view_type = .@"2d",
            .format = format.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{ .aspect_mask = .{ .color_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
        }, null);
        sem.* = ctx.device.createSemaphore(&.{}, null) catch |err| {
            ctx.device.destroyImageView(view.*, null);
            return err;
        };
        created += 1;
    }
    return .{ .handle = handle, .format = format.format, .extent = actual, .images = images, .views = views, .render_done = render_done };
}

/// `old` (the previous swapchain, or null) is only retired: the caller destroys
/// it once the new one exists, so a failure here leaves it usable.
pub fn deinit(self: *Swapchain, ctx: *const Context, gpa: Allocator) void {
    for (self.views, self.render_done) |v, s| {
        ctx.device.destroyImageView(v, null);
        ctx.device.destroySemaphore(s, null);
    }
    gpa.free(self.views);
    gpa.free(self.render_done);
    gpa.free(self.images);
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

Replace `src/render/pipeline.zig` with:

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

Replace `src/render/ChunkBuffers.zig` with:

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

// ponytail: linear scan per upload, O(n²) over an initial burst; a pos → index
// map if it ever shows in a profile (it does not next to meshing).
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
/// metadata updates. The caller issues the barrier that makes them visible to
/// culling and drawing.
pub fn record(self: *ChunkBuffers, cmd: vk.CommandBufferProxy, staging: *const Buffer) !void {
    barrier(cmd, .{ .compute_shader_bit = true, .vertex_shader_bit = true, .draw_indirect_bit = true }, .{ .shader_storage_read_bit = true, .indirect_command_read_bit = true }, .{ .copy_bit = true, .clear_bit = true }, .{ .transfer_write_bit = true });

    const staged = staging.slice(Quad);
    var used: usize = 0;
    var copies: std.ArrayList(vk.BufferCopy) = .empty;
    defer copies.deinit(self.gpa);
    var done: usize = 0;
    for (self.pending.items) |p| {
        if (used + p.quads.len > staged.len) break;
        done += 1;
        if (p.quads.len == 0) {
            try self.release(p.pos);
            continue;
        }
        // Allocate before releasing: when the buffer is full the previous mesh stays visible.
        const range = self.ranges.alloc(@intCast(p.quads.len)) orelse {
            std.log.scoped(.render).warn("quad buffer full, keeping the previous mesh of chunk {any}", .{p.pos});
            continue;
        };
        const index = if (self.slots.get(p.pos)) |old| blk: {
            try self.ranges.free(self.gpa, old.range);
            self.quad_count -= old.range.len;
            break :blk old.index;
        } else self.free_slots.pop() orelse blk: {
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

- [ ] **Step 5: Replace the renderer**

Replace `src/render/Renderer.zig` with:

```zig
//! Frame loop: frames in flight, chunk uploads, GPU culling for the camera and
//! each shadow cascade, shadow passes, then sky and chunks.
const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const world = @import("world");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");
const Swapchain = @import("Swapchain.zig");
const pipeline = @import("pipeline.zig");
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const ChunkBuffers = @import("ChunkBuffers.zig");
const Shadows = @import("Shadows.zig");
const gpu = @import("gpu.zig");
const Camera = @import("../Camera.zig");
const Sun = @import("../Sun.zig");

const Renderer = @This();

pub const frames_in_flight = 2;
const depth_format: vk.Format = .d32_sfloat;
/// Views culled each frame: the camera, then one per shadow cascade.
const views = 1 + Shadows.cascades;
const max_draws = ChunkBuffers.max_chunks * 6;
const chunk_stages: vk.ShaderStageFlags = .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true };

const Frame = struct {
    pool: vk.CommandPool,
    cmd: vk.CommandBuffer,
    image_acquired: vk.Semaphore,
    fence: vk.Fence,
    /// Written by the CPU while the frame is recorded, read by the GPU.
    frame_data: Buffer,
    staging: Buffer,
};

pub const Options = struct {
    shadow_resolution: u32 = 2048,
};

pub const FrameInput = struct {
    camera: Camera,
    light: Sun.Lighting,
    /// Shadows are skipped at night (the moon casts none).
    shadows: bool,
    fog_start: f32,
    fog_end: f32,
};

gpa: Allocator,
ctx: *const Context,
swapchain: Swapchain,
/// Framebuffer size the swapchain was last built for (not the clamped one).
requested_extent: vk.Extent2D,
depth: Image,
frames: [frames_in_flight]Frame,
frame_index: usize = 0,
sky_layout: vk.PipelineLayout,
sky_pipeline: vk.Pipeline,
chunks: ChunkBuffers,
shadows: Shadows,
draws: Buffer,
draw_count: Buffer,
set_layout: vk.DescriptorSetLayout,
chunk_layout: vk.PipelineLayout,
cull_pipeline: vk.Pipeline,
chunk_pipeline: vk.Pipeline,
shadow_pipeline: vk.Pipeline,

pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D, options: Options) !Renderer {
    const d = ctx.device;
    var swapchain: Swapchain = try .init(ctx, gpa, extent, .null_handle);
    errdefer swapchain.deinit(ctx, gpa);
    var depth: Image = try .initDepth(ctx, swapchain.extent, depth_format);
    errdefer depth.deinit(ctx);

    var frames: [frames_in_flight]Frame = undefined;
    var frames_done: usize = 0;
    errdefer for (frames[0..frames_done]) |*f| destroyFrame(ctx, f);
    for (&frames) |*f| {
        f.* = try createFrame(ctx);
        frames_done += 1;
    }

    var chunks: ChunkBuffers = try .init(ctx, gpa);
    errdefer chunks.deinit(ctx);
    var shadows: Shadows = try .init(ctx, options.shadow_resolution);
    errdefer shadows.deinit(ctx);
    var draws: Buffer = try .init(ctx, views * max_draws * @sizeOf(gpu.DrawCmd), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true }, false);
    errdefer draws.deinit(ctx);
    var draw_count: Buffer = try .init(ctx, views * @sizeOf(u32), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true, .transfer_dst_bit = true }, false);
    errdefer draw_count.deinit(ctx);

    // The shadow map is bound with a push descriptor: no pool, no sets.
    const binding: vk.DescriptorSetLayoutBinding = .{ .binding = 0, .descriptor_type = .combined_image_sampler, .descriptor_count = 1, .stage_flags = .{ .fragment_bit = true } };
    const set_layout = try d.createDescriptorSetLayout(&.{ .flags = .{ .push_descriptor_bit = true }, .binding_count = 1, .p_bindings = @ptrCast(&binding) }, null);
    errdefer d.destroyDescriptorSetLayout(set_layout, null);

    const sky_layout = try pipeline.createLayout(ctx, @sizeOf(SkyPush), .{ .fragment_bit = true }, &.{});
    errdefer d.destroyPipelineLayout(sky_layout, null);
    const chunk_layout = try pipeline.createLayout(ctx, @sizeOf(gpu.Push), chunk_stages, &.{set_layout});
    errdefer d.destroyPipelineLayout(chunk_layout, null);

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
    errdefer d.destroyPipeline(sky_pipeline, null);
    const cull_pipeline = try pipeline.createCompute(ctx, chunk_layout, pipeline.spirv("cull.comp"));
    errdefer d.destroyPipeline(cull_pipeline, null);
    const chunk_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = chunk_layout,
        .vertex = pipeline.spirv("chunk.vert"),
        .fragment = pipeline.spirv("chunk.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
    });
    errdefer d.destroyPipeline(chunk_pipeline, null);
    // Casters beyond the cascade's near plane are clamped instead of clipped.
    const shadow_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = chunk_layout,
        .vertex = pipeline.spirv("shadow.vert"),
        .fragment = null,
        .color_format = null,
        .depth_format = Shadows.format,
        .depth_clamp = true,
        .depth_bias = true,
    });

    return .{
        .gpa = gpa,
        .ctx = ctx,
        .swapchain = swapchain,
        .requested_extent = extent,
        .depth = depth,
        .frames = frames,
        .sky_layout = sky_layout,
        .sky_pipeline = sky_pipeline,
        .chunks = chunks,
        .shadows = shadows,
        .draws = draws,
        .draw_count = draw_count,
        .set_layout = set_layout,
        .chunk_layout = chunk_layout,
        .cull_pipeline = cull_pipeline,
        .chunk_pipeline = chunk_pipeline,
        .shadow_pipeline = shadow_pipeline,
    };
}

pub fn deinit(self: *Renderer) void {
    const d = self.ctx.device;
    d.deviceWaitIdle() catch {};
    d.destroyPipeline(self.shadow_pipeline, null);
    d.destroyPipeline(self.chunk_pipeline, null);
    d.destroyPipeline(self.cull_pipeline, null);
    d.destroyPipeline(self.sky_pipeline, null);
    d.destroyPipelineLayout(self.chunk_layout, null);
    d.destroyPipelineLayout(self.sky_layout, null);
    d.destroyDescriptorSetLayout(self.set_layout, null);
    self.draw_count.deinit(self.ctx);
    self.draws.deinit(self.ctx);
    self.shadows.deinit(self.ctx);
    self.chunks.deinit(self.ctx);
    for (&self.frames) |*f| destroyFrame(self.ctx, f);
    self.depth.deinit(self.ctx);
    self.swapchain.deinit(self.ctx, self.gpa);
}

fn createFrame(ctx: *const Context) !Frame {
    const d = ctx.device;
    const pool = try d.createCommandPool(&.{ .flags = .{ .reset_command_buffer_bit = true }, .queue_family_index = ctx.queue_family }, null);
    errdefer d.destroyCommandPool(pool, null);
    var cmd: vk.CommandBuffer = undefined;
    try d.allocateCommandBuffers(&.{ .command_pool = pool, .level = .primary, .command_buffer_count = 1 }, @ptrCast(&cmd));
    const image_acquired = try d.createSemaphore(&.{}, null);
    errdefer d.destroySemaphore(image_acquired, null);
    const fence = try d.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
    errdefer d.destroyFence(fence, null);
    var frame_data: Buffer = try .init(ctx, @sizeOf(gpu.FrameData), .{ .storage_buffer_bit = true }, true);
    errdefer frame_data.deinit(ctx);
    const staging: Buffer = try .init(ctx, ChunkBuffers.staging_size, .{ .transfer_src_bit = true }, true);
    return .{ .pool = pool, .cmd = cmd, .image_acquired = image_acquired, .fence = fence, .frame_data = frame_data, .staging = staging };
}

fn destroyFrame(ctx: *const Context, f: *Frame) void {
    f.staging.deinit(ctx);
    f.frame_data.deinit(ctx);
    ctx.device.destroyFence(f.fence, null);
    ctx.device.destroySemaphore(f.image_acquired, null);
    ctx.device.destroyCommandPool(f.pool, null);
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
    // Reset only once an image is acquired: an early return must leave the fence signalled.
    try d.resetFences(&.{frame.fence});
    const image_index = acquired.image_index;

    const cmd: vk.CommandBufferProxy = .init(frame.cmd, self.ctx.vkd);
    try cmd.resetCommandBuffer(.{});
    try cmd.beginCommandBuffer(&.{ .flags = .{ .one_time_submit_bit = true } });

    const ext = self.swapchain.extent;
    const aspect = @as(f32, @floatFromInt(ext.width)) / @as(f32, @floatFromInt(ext.height));
    const view_proj = in.camera.viewProj(aspect);
    // Uploads first: slots allocated this frame are then culled this frame.
    try self.chunks.record(cmd, &frame.staging);
    self.writeFrameData(frame, in, view_proj, aspect);

    // 1. Chunk uploads, then GPU culling for every view into the indirect buffer.
    var push: gpu.Push = .{
        .frame = frame.frame_data.address,
        .metas = self.chunks.metas.address,
        .quads = self.chunks.quads.address,
        .draws = self.draws.address,
        .count = self.draw_count.address,
        .view = 0,
    };
    cmd.fillBuffer(self.draw_count.handle, 0, views * @sizeOf(u32), 0);
    ChunkBuffers.barrier(cmd, .{ .copy_bit = true, .clear_bit = true }, .{ .transfer_write_bit = true }, .{ .compute_shader_bit = true, .vertex_shader_bit = true }, .{ .shader_storage_read_bit = true, .shader_storage_write_bit = true });
    cmd.bindPipeline(.compute, self.cull_pipeline);
    const groups = std.math.divCeil(u32, self.chunks.slot_high, 64) catch unreachable;
    const culled_views: u32 = if (in.shadows) views else 1;
    for (0..culled_views) |v| {
        push.view = @intCast(v);
        cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
        cmd.dispatch(groups, 1, 1);
    }
    ChunkBuffers.barrier(cmd, .{ .compute_shader_bit = true }, .{ .shader_storage_write_bit = true }, .{ .draw_indirect_bit = true }, .{ .indirect_command_read_bit = true });

    // 2. Shadow cascades, depth only. Cleared even when skipped: the lighting pass samples them.
    const shadow_image = self.shadows.image.image;
    imageBarrier(cmd, shadow_image, .{ .depth_bit = true }, .undefined, .depth_attachment_optimal, .{ .fragment_shader_bit = true }, .{}, .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true });
    const res = self.shadows.resolution;
    for (self.shadows.layer_views, 1..) |layer_view, v| {
        const att: vk.RenderingAttachmentInfo = .{
            .image_view = layer_view,
            .image_layout = .depth_attachment_optimal,
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .depth_stencil = .{ .depth = 0, .stencil = 0 } },
        };
        cmd.beginRendering(&.{
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = res, .height = res } },
            .layer_count = 1,
            .view_mask = 0,
            .color_attachment_count = 0,
            .p_depth_attachment = &att,
        });
        if (in.shadows) {
            setViewport(cmd, .{ .width = res, .height = res });
            // Reverse-Z: a negative bias pushes casters away from the light.
            cmd.setDepthBias(-1.5, 0, -2.0);
            cmd.bindPipeline(.graphics, self.shadow_pipeline);
            push.view = @intCast(v);
            cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
            cmd.drawIndirectCount(self.draws.handle, v * max_draws * @sizeOf(gpu.DrawCmd), self.draw_count.handle, v * @sizeOf(u32), max_draws, @sizeOf(gpu.DrawCmd));
        }
        cmd.endRendering();
    }
    imageBarrier(cmd, shadow_image, .{ .depth_bit = true }, .depth_attachment_optimal, .depth_read_only_optimal, .{ .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .fragment_shader_bit = true }, .{ .shader_sampled_read_bit = true });

    // 3. Main pass: sky, then chunks.
    const image = self.swapchain.images[image_index];
    imageBarrier(cmd, image, .{ .color_bit = true }, .undefined, .color_attachment_optimal, .{ .color_attachment_output_bit = true }, .{}, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true });
    imageBarrier(cmd, self.depth.image, .{ .depth_bit = true }, .undefined, .depth_attachment_optimal, .{ .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true });
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

    const d_sun = in.light.dir;
    const sky: SkyPush = .{ .inv_view_proj = zm.inverse(view_proj), .sun_dir = .{ d_sun[0], d_sun[1], d_sun[2], 0 } };
    cmd.bindPipeline(.graphics, self.sky_pipeline);
    cmd.pushConstants(self.sky_layout, .{ .fragment_bit = true }, 0, @sizeOf(SkyPush), &sky);
    cmd.draw(3, 1, 0, 0);

    const shadow_info: vk.DescriptorImageInfo = .{ .sampler = self.shadows.sampler, .image_view = self.shadows.image.view, .image_layout = .depth_read_only_optimal };
    cmd.pushDescriptorSet(.graphics, self.chunk_layout, 0, &.{.{
        .dst_set = .null_handle,
        .dst_binding = 0,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = .combined_image_sampler,
        .p_image_info = @ptrCast(&shadow_info),
        .p_buffer_info = undefined,
        .p_texel_buffer_view = undefined,
    }});
    cmd.bindPipeline(.graphics, self.chunk_pipeline);
    push.view = 0;
    cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
    cmd.drawIndirectCount(self.draws.handle, 0, self.draw_count.handle, 0, max_draws, @sizeOf(gpu.DrawCmd));

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
    if (present == .suboptimal_khr or extent.width != self.requested_extent.width or extent.height != self.requested_extent.height)
        try self.recreate(extent);
}

fn writeFrameData(self: *Renderer, frame: *Frame, in: FrameInput, view_proj: zm.Mat, aspect: f32) void {
    var palette: [8][4]f32 = undefined;
    for (&palette, 0..) |*c, i| {
        const rgb = @as(world.Block, @enumFromInt(i)).color();
        c.* = .{ rgb[0], rgb[1], rgb[2], 1 };
    }
    const cam = in.camera;
    const l = in.light;
    const cascades = Shadows.fitCascades(self.shadows.resolution, cam.pos, cam.forward(), cam.fov_y, aspect, cam.near, l.dir);
    var cascade_vp: [Shadows.cascades]zm.Mat = undefined;
    var cascade_planes: [Shadows.cascades * 6][4]f32 = undefined;
    var cascade_texel: [4]f32 = .{ 0, 0, 0, 0 };
    for (cascades, 0..) |c, i| {
        cascade_vp[i] = c.view_proj;
        @memcpy(cascade_planes[i * 6 ..][0..6], &gpu.frustumPlanes(c.view_proj));
        cascade_texel[i] = c.texel;
    }

    const data: *gpu.FrameData = @ptrCast(@alignCast(frame.frame_data.mapped.?));
    data.* = .{
        .view_proj = view_proj,
        .planes = gpu.frustumPlanes(view_proj),
        .camera_pos = .{ cam.pos[0], cam.pos[1], cam.pos[2], 1 },
        .sun_dir = .{ l.dir[0], l.dir[1], l.dir[2], 0 },
        .sun_color = .{ l.color[0], l.color[1], l.color[2], 0 },
        .ambient = .{ l.ambient[0], l.ambient[1], l.ambient[2], 0 },
        .fog = .{ in.fog_start, in.fog_end, 0, 0 },
        .palette = palette,
        .cascade_vp = cascade_vp,
        .cascade_planes = cascade_planes,
        .cascade_splits = .{ Shadows.splits[0], Shadows.splits[1], Shadows.splits[2], 0 },
        .cascade_texel = cascade_texel,
        .shadow = .{ if (in.shadows) 1 else 0, 0, 0, 0 },
        .chunk_capacity = self.chunks.slot_high,
        .max_draws = max_draws,
    };
}

/// Rebuilds swapchain and depth buffer for `extent`. Atomic: on failure the
/// current ones stay valid.
fn recreate(self: *Renderer, extent: vk.Extent2D) !void {
    if (extent.width == 0 or extent.height == 0) return; // minimized
    try self.ctx.device.deviceWaitIdle();
    var swapchain = Swapchain.init(self.ctx, self.gpa, extent, self.swapchain.handle) catch |err| switch (err) {
        error.ZeroExtent => return, // minimized between the size query and now
        else => return err,
    };
    errdefer swapchain.deinit(self.ctx, self.gpa);
    const depth: Image = try .initDepth(self.ctx, swapchain.extent, depth_format);
    self.swapchain.deinit(self.ctx, self.gpa); // the old handle is retired, destroying it is valid
    self.depth.deinit(self.ctx);
    self.swapchain = swapchain;
    self.depth = depth;
    self.requested_extent = extent;
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

- [ ] **Step 6: Build the new shaders and update the main loop**

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
    const shaders = [_][]const u8{ "fullscreen.vert", "sky.frag", "cull.comp", "chunk.vert", "chunk.frag", "shadow.vert" };
    for (shaders) |name| {
        const glslc = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.4", "-O", "-o" });
        const spv = glslc.addOutputFileArg(b.fmt("{s}.spv", .{name}));
        glslc.addFileArg(b.path(b.fmt("src/render/shaders/{s}", .{name})));
        glslc.addFileInput(b.path("src/render/shaders/common.glsl"));
        glslc.addFileInput(b.path("src/render/shaders/gpu.glsl"));
        glslc.addFileInput(b.path("src/render/shaders/pull.glsl"));
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
```

- [ ] **Step 7: Build, test, run**

Run: `zig build test --summary all`
Expected: all tests pass (79), including `cascade covers the camera and maps it inside the depth range` and `closer to the sun means larger depth (reverse-Z)`.

Run: `zig build && timeout 7 ./zig-out/bin/ft_vox 2>&1 | grep -v worker_pool`
Expected: exactly these two lines and no `error(vulkan)` / `warning(vulkan)` line:

```
info(vulkan): validation layers: on
info(vulkan): GPU: <your GPU name>
```

Visual check (optional, needs `xdotool` and `ffmpeg`): start the engine, wait ~6 s, click in the window, move the mouse down ~300 px with `xdotool mousemove_relative -- 0 30` in a loop, hold `ctrl` ~1 s, capture the window: trees cast shadows on the grass and terrain steps shade each other; no shadow acne (stripes) on lit faces.

- [ ] **Step 8: Commit**

```bash
zig fmt --check src build.zig
git add build.zig src/main.zig src/render
git commit -m "feat(render): cascaded shadow maps, per-view GPU culling; renderer hardening

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Command-line settings

**Files:**
- Create: `src/Settings.zig`
- Replace: `src/main.zig`

**Interfaces:**
- Produces: `Settings { seed: u64 = 42, radius: u16 = 16, shadow_resolution: u32 = 2048, day_length: f32 = 240 }`, `Settings.parse(args: []const []const u8) ParseError!Settings` (without the program name), `Settings.usage`, `ParseError = error{ UnknownOption, MissingValue, InvalidValue }`. Bounds: radius 4..32, shadow resolution a power of two in 512..4096, day length > 0.

- [ ] **Step 1: Write the settings module (tests included)**

Create `src/Settings.zig`:

```zig
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
            if (!(s.day_length > 0)) return error.InvalidValue;
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
}
```

- [ ] **Step 2: Wire the settings into the main loop**

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
const Settings = @import("Settings.zig");

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

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const settings = Settings.parse(args[1..]) catch |err| {
        std.debug.print("ft_vox: {t}\n{s}", .{ err, Settings.usage });
        std.process.exit(2);
    };

    try glfw.init();
    defer glfw.terminate();
    glfw.windowHint(.client_api, .no_api);
    const window = try glfw.Window.create(1280, 720, "ft_vox", null, null);
    defer window.destroy();
    try glfw.setInputMode(window, .cursor, .disabled);

    var ctx: Context = try .init(gpa, window);
    defer ctx.deinit(gpa);
    var renderer: Renderer = try .init(gpa, &ctx, framebufferExtent(window), .{ .shadow_resolution = settings.shadow_resolution });
    defer renderer.deinit();

    // One core stays free for the render thread: oversubscribing starves it while loading.
    const workers = @max(2, (std.Thread.getCpuCount() catch 3) - 1);
    const mesh_workers = @max(1, workers / 3);
    const chunks = try ChunkManager.create(gpa, io, .{ .seed = settings.seed, .radius = settings.radius, .gen_workers = workers - mesh_workers, .mesh_workers = mesh_workers });
    defer chunks.destroy();

    var camera: Camera = .{};
    var sun: Sun = .{ .day_length = settings.day_length };
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
        const far: f32 = @floatFromInt(@as(u32, settings.radius) * world.chunk_size);
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
    _ = @import("Settings.zig");
}
```

- [ ] **Step 3: Build, test, run**

Run: `zig build test --summary all`
Expected: all tests pass (82), including `defaults`, `all options` and `errors`.

Run: `zig build && timeout 7 ./zig-out/bin/ft_vox 2>&1 | grep -v worker_pool`
Expected: exactly these two lines and no `error(vulkan)` / `warning(vulkan)` line:

```
info(vulkan): validation layers: on
info(vulkan): GPU: <your GPU name>
```

Run: `./zig-out/bin/ft_vox --radius 99; echo "exit=$?"`
Expected:

```
ft_vox: InvalidValue
usage: ft_vox [--seed N] [--radius 4..32] [--shadow-res 512..4096] [--day-length SECONDS]
exit=2
```

Run: `timeout 6 ./zig-out/bin/ft_vox --radius 8 --shadow-res 1024 --seed 7 2>&1 | grep -v worker_pool` — same two info lines; the window title shows far fewer chunks (a radius of 8).

- [ ] **Step 4: Commit**

```bash
zig fmt --check src build.zig
git add src/Settings.zig src/main.zig
git commit -m "feat: command-line settings (seed, radius, shadow resolution, day length)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Block breaking with an outline

**Files:**
- Create: `src/render/shaders/outline.vert`, `src/render/shaders/outline.frag`
- Replace: `src/render/Renderer.zig`, `build.zig`, `src/main.zig`

**Interfaces:**
- Consumes: `world.raycast(origin, dir, max_dist, lookup) ?world.Hit`, `ChunkManager.blockAt` (the manager is the raycast `lookup`), `ChunkManager.breakBlock(pos) !bool`.
- Produces: `Renderer.FrameInput.target: ?world.BlockPos` (block to outline).

Behavior: every frame the camera ray is cast up to 8 blocks; the hit block gets a dark wireframe outline (24 vertices generated in the vertex shader, 0.4 % larger than the block, depth-tested without depth writes); a left-click press (rising edge, one block per click) breaks it, and the ChunkManager's coalesced remesh brings the change back within a frame or two.

- [ ] **Step 1: Write the outline shaders**

Create `src/render/shaders/outline.vert`:

```glsl
#version 460
#extension GL_EXT_scalar_block_layout : require

// The 12 edges of the targeted block, as a line list (24 vertices).
layout(push_constant, scalar) uniform Push {
    mat4 view_proj;
    vec3 block;
} pc;

const uint edges[24] = uint[24](0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7);

void main() {
    uint c = edges[gl_VertexIndex];
    vec3 corner = vec3(c & 1u, (c >> 2) & 1u, (c >> 1) & 1u);
    // Slightly larger than the block so the lines are not hidden by its faces.
    vec3 p = pc.block + 0.5 + (corner - 0.5) * 1.004;
    gl_Position = pc.view_proj * vec4(p, 1.0);
}
```

Create `src/render/shaders/outline.frag`:

```glsl
#version 460

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(0.05, 0.05, 0.05, 1.0);
}
```

- [ ] **Step 2: Draw the outline**

Replace `src/render/Renderer.zig` with:

```zig
//! Frame loop: frames in flight, chunk uploads, GPU culling for the camera and
//! each shadow cascade, shadow passes, then sky and chunks.
const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");
const world = @import("world");
const Allocator = std.mem.Allocator;
const Context = @import("Context.zig");
const Swapchain = @import("Swapchain.zig");
const pipeline = @import("pipeline.zig");
const Image = @import("Image.zig");
const Buffer = @import("Buffer.zig");
const ChunkBuffers = @import("ChunkBuffers.zig");
const Shadows = @import("Shadows.zig");
const gpu = @import("gpu.zig");
const Camera = @import("../Camera.zig");
const Sun = @import("../Sun.zig");

const Renderer = @This();

pub const frames_in_flight = 2;
const depth_format: vk.Format = .d32_sfloat;
/// Views culled each frame: the camera, then one per shadow cascade.
const views = 1 + Shadows.cascades;
const max_draws = ChunkBuffers.max_chunks * 6;
const chunk_stages: vk.ShaderStageFlags = .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true };

const Frame = struct {
    pool: vk.CommandPool,
    cmd: vk.CommandBuffer,
    image_acquired: vk.Semaphore,
    fence: vk.Fence,
    /// Written by the CPU while the frame is recorded, read by the GPU.
    frame_data: Buffer,
    staging: Buffer,
};

pub const Options = struct {
    shadow_resolution: u32 = 2048,
};

pub const FrameInput = struct {
    camera: Camera,
    light: Sun.Lighting,
    /// Shadows are skipped at night (the moon casts none).
    shadows: bool,
    /// Block to outline (the one the player aims at).
    target: ?world.BlockPos,
    fog_start: f32,
    fog_end: f32,
};

gpa: Allocator,
ctx: *const Context,
swapchain: Swapchain,
/// Framebuffer size the swapchain was last built for (not the clamped one).
requested_extent: vk.Extent2D,
depth: Image,
frames: [frames_in_flight]Frame,
frame_index: usize = 0,
sky_layout: vk.PipelineLayout,
sky_pipeline: vk.Pipeline,
chunks: ChunkBuffers,
shadows: Shadows,
draws: Buffer,
draw_count: Buffer,
set_layout: vk.DescriptorSetLayout,
chunk_layout: vk.PipelineLayout,
cull_pipeline: vk.Pipeline,
chunk_pipeline: vk.Pipeline,
shadow_pipeline: vk.Pipeline,
outline_layout: vk.PipelineLayout,
outline_pipeline: vk.Pipeline,

pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D, options: Options) !Renderer {
    const d = ctx.device;
    var swapchain: Swapchain = try .init(ctx, gpa, extent, .null_handle);
    errdefer swapchain.deinit(ctx, gpa);
    var depth: Image = try .initDepth(ctx, swapchain.extent, depth_format);
    errdefer depth.deinit(ctx);

    var frames: [frames_in_flight]Frame = undefined;
    var frames_done: usize = 0;
    errdefer for (frames[0..frames_done]) |*f| destroyFrame(ctx, f);
    for (&frames) |*f| {
        f.* = try createFrame(ctx);
        frames_done += 1;
    }

    var chunks: ChunkBuffers = try .init(ctx, gpa);
    errdefer chunks.deinit(ctx);
    var shadows: Shadows = try .init(ctx, options.shadow_resolution);
    errdefer shadows.deinit(ctx);
    var draws: Buffer = try .init(ctx, views * max_draws * @sizeOf(gpu.DrawCmd), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true }, false);
    errdefer draws.deinit(ctx);
    var draw_count: Buffer = try .init(ctx, views * @sizeOf(u32), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true, .transfer_dst_bit = true }, false);
    errdefer draw_count.deinit(ctx);

    // The shadow map is bound with a push descriptor: no pool, no sets.
    const binding: vk.DescriptorSetLayoutBinding = .{ .binding = 0, .descriptor_type = .combined_image_sampler, .descriptor_count = 1, .stage_flags = .{ .fragment_bit = true } };
    const set_layout = try d.createDescriptorSetLayout(&.{ .flags = .{ .push_descriptor_bit = true }, .binding_count = 1, .p_bindings = @ptrCast(&binding) }, null);
    errdefer d.destroyDescriptorSetLayout(set_layout, null);

    const sky_layout = try pipeline.createLayout(ctx, @sizeOf(SkyPush), .{ .fragment_bit = true }, &.{});
    errdefer d.destroyPipelineLayout(sky_layout, null);
    const chunk_layout = try pipeline.createLayout(ctx, @sizeOf(gpu.Push), chunk_stages, &.{set_layout});
    errdefer d.destroyPipelineLayout(chunk_layout, null);

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
    errdefer d.destroyPipeline(sky_pipeline, null);
    const cull_pipeline = try pipeline.createCompute(ctx, chunk_layout, pipeline.spirv("cull.comp"));
    errdefer d.destroyPipeline(cull_pipeline, null);
    const chunk_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = chunk_layout,
        .vertex = pipeline.spirv("chunk.vert"),
        .fragment = pipeline.spirv("chunk.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
    });
    errdefer d.destroyPipeline(chunk_pipeline, null);
    // Casters beyond the cascade's near plane are clamped instead of clipped.
    const shadow_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = chunk_layout,
        .vertex = pipeline.spirv("shadow.vert"),
        .fragment = null,
        .color_format = null,
        .depth_format = Shadows.format,
        .depth_clamp = true,
        .depth_bias = true,
    });
    errdefer d.destroyPipeline(shadow_pipeline, null);
    const outline_layout = try pipeline.createLayout(ctx, @sizeOf(OutlinePush), .{ .vertex_bit = true }, &.{});
    errdefer d.destroyPipelineLayout(outline_layout, null);
    const outline_pipeline = try pipeline.createGraphics(ctx, .{
        .layout = outline_layout,
        .vertex = pipeline.spirv("outline.vert"),
        .fragment = pipeline.spirv("outline.frag"),
        .color_format = swapchain.format,
        .depth_format = depth_format,
        .depth_write = false,
        .cull_back = false,
        .topology = .line_list,
    });

    return .{
        .gpa = gpa,
        .ctx = ctx,
        .swapchain = swapchain,
        .requested_extent = extent,
        .depth = depth,
        .frames = frames,
        .sky_layout = sky_layout,
        .sky_pipeline = sky_pipeline,
        .chunks = chunks,
        .shadows = shadows,
        .draws = draws,
        .draw_count = draw_count,
        .set_layout = set_layout,
        .chunk_layout = chunk_layout,
        .cull_pipeline = cull_pipeline,
        .chunk_pipeline = chunk_pipeline,
        .shadow_pipeline = shadow_pipeline,
        .outline_layout = outline_layout,
        .outline_pipeline = outline_pipeline,
    };
}

pub fn deinit(self: *Renderer) void {
    const d = self.ctx.device;
    d.deviceWaitIdle() catch {};
    d.destroyPipeline(self.outline_pipeline, null);
    d.destroyPipelineLayout(self.outline_layout, null);
    d.destroyPipeline(self.shadow_pipeline, null);
    d.destroyPipeline(self.chunk_pipeline, null);
    d.destroyPipeline(self.cull_pipeline, null);
    d.destroyPipeline(self.sky_pipeline, null);
    d.destroyPipelineLayout(self.chunk_layout, null);
    d.destroyPipelineLayout(self.sky_layout, null);
    d.destroyDescriptorSetLayout(self.set_layout, null);
    self.draw_count.deinit(self.ctx);
    self.draws.deinit(self.ctx);
    self.shadows.deinit(self.ctx);
    self.chunks.deinit(self.ctx);
    for (&self.frames) |*f| destroyFrame(self.ctx, f);
    self.depth.deinit(self.ctx);
    self.swapchain.deinit(self.ctx, self.gpa);
}

fn createFrame(ctx: *const Context) !Frame {
    const d = ctx.device;
    const pool = try d.createCommandPool(&.{ .flags = .{ .reset_command_buffer_bit = true }, .queue_family_index = ctx.queue_family }, null);
    errdefer d.destroyCommandPool(pool, null);
    var cmd: vk.CommandBuffer = undefined;
    try d.allocateCommandBuffers(&.{ .command_pool = pool, .level = .primary, .command_buffer_count = 1 }, @ptrCast(&cmd));
    const image_acquired = try d.createSemaphore(&.{}, null);
    errdefer d.destroySemaphore(image_acquired, null);
    const fence = try d.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
    errdefer d.destroyFence(fence, null);
    var frame_data: Buffer = try .init(ctx, @sizeOf(gpu.FrameData), .{ .storage_buffer_bit = true }, true);
    errdefer frame_data.deinit(ctx);
    const staging: Buffer = try .init(ctx, ChunkBuffers.staging_size, .{ .transfer_src_bit = true }, true);
    return .{ .pool = pool, .cmd = cmd, .image_acquired = image_acquired, .fence = fence, .frame_data = frame_data, .staging = staging };
}

fn destroyFrame(ctx: *const Context, f: *Frame) void {
    f.staging.deinit(ctx);
    f.frame_data.deinit(ctx);
    ctx.device.destroyFence(f.fence, null);
    ctx.device.destroySemaphore(f.image_acquired, null);
    ctx.device.destroyCommandPool(f.pool, null);
}

const SkyPush = extern struct {
    inv_view_proj: zm.Mat,
    sun_dir: [4]f32,
};

const OutlinePush = extern struct {
    view_proj: zm.Mat,
    block: [3]f32,
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
    // Reset only once an image is acquired: an early return must leave the fence signalled.
    try d.resetFences(&.{frame.fence});
    const image_index = acquired.image_index;

    const cmd: vk.CommandBufferProxy = .init(frame.cmd, self.ctx.vkd);
    try cmd.resetCommandBuffer(.{});
    try cmd.beginCommandBuffer(&.{ .flags = .{ .one_time_submit_bit = true } });

    const ext = self.swapchain.extent;
    const aspect = @as(f32, @floatFromInt(ext.width)) / @as(f32, @floatFromInt(ext.height));
    const view_proj = in.camera.viewProj(aspect);
    // Uploads first: slots allocated this frame are then culled this frame.
    try self.chunks.record(cmd, &frame.staging);
    self.writeFrameData(frame, in, view_proj, aspect);

    // 1. Chunk uploads, then GPU culling for every view into the indirect buffer.
    var push: gpu.Push = .{
        .frame = frame.frame_data.address,
        .metas = self.chunks.metas.address,
        .quads = self.chunks.quads.address,
        .draws = self.draws.address,
        .count = self.draw_count.address,
        .view = 0,
    };
    cmd.fillBuffer(self.draw_count.handle, 0, views * @sizeOf(u32), 0);
    ChunkBuffers.barrier(cmd, .{ .copy_bit = true, .clear_bit = true }, .{ .transfer_write_bit = true }, .{ .compute_shader_bit = true, .vertex_shader_bit = true }, .{ .shader_storage_read_bit = true, .shader_storage_write_bit = true });
    cmd.bindPipeline(.compute, self.cull_pipeline);
    const groups = std.math.divCeil(u32, self.chunks.slot_high, 64) catch unreachable;
    const culled_views: u32 = if (in.shadows) views else 1;
    for (0..culled_views) |v| {
        push.view = @intCast(v);
        cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
        cmd.dispatch(groups, 1, 1);
    }
    ChunkBuffers.barrier(cmd, .{ .compute_shader_bit = true }, .{ .shader_storage_write_bit = true }, .{ .draw_indirect_bit = true }, .{ .indirect_command_read_bit = true });

    // 2. Shadow cascades, depth only. Cleared even when skipped: the lighting pass samples them.
    const shadow_image = self.shadows.image.image;
    imageBarrier(cmd, shadow_image, .{ .depth_bit = true }, .undefined, .depth_attachment_optimal, .{ .fragment_shader_bit = true }, .{}, .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true });
    const res = self.shadows.resolution;
    for (self.shadows.layer_views, 1..) |layer_view, v| {
        const att: vk.RenderingAttachmentInfo = .{
            .image_view = layer_view,
            .image_layout = .depth_attachment_optimal,
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .depth_stencil = .{ .depth = 0, .stencil = 0 } },
        };
        cmd.beginRendering(&.{
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = res, .height = res } },
            .layer_count = 1,
            .view_mask = 0,
            .color_attachment_count = 0,
            .p_depth_attachment = &att,
        });
        if (in.shadows) {
            setViewport(cmd, .{ .width = res, .height = res });
            // Reverse-Z: a negative bias pushes casters away from the light.
            cmd.setDepthBias(-1.5, 0, -2.0);
            cmd.bindPipeline(.graphics, self.shadow_pipeline);
            push.view = @intCast(v);
            cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
            cmd.drawIndirectCount(self.draws.handle, v * max_draws * @sizeOf(gpu.DrawCmd), self.draw_count.handle, v * @sizeOf(u32), max_draws, @sizeOf(gpu.DrawCmd));
        }
        cmd.endRendering();
    }
    imageBarrier(cmd, shadow_image, .{ .depth_bit = true }, .depth_attachment_optimal, .depth_read_only_optimal, .{ .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .fragment_shader_bit = true }, .{ .shader_sampled_read_bit = true });

    // 3. Main pass: sky, then chunks.
    const image = self.swapchain.images[image_index];
    imageBarrier(cmd, image, .{ .color_bit = true }, .undefined, .color_attachment_optimal, .{ .color_attachment_output_bit = true }, .{}, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true });
    imageBarrier(cmd, self.depth.image, .{ .depth_bit = true }, .undefined, .depth_attachment_optimal, .{ .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true });
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

    const d_sun = in.light.dir;
    const sky: SkyPush = .{ .inv_view_proj = zm.inverse(view_proj), .sun_dir = .{ d_sun[0], d_sun[1], d_sun[2], 0 } };
    cmd.bindPipeline(.graphics, self.sky_pipeline);
    cmd.pushConstants(self.sky_layout, .{ .fragment_bit = true }, 0, @sizeOf(SkyPush), &sky);
    cmd.draw(3, 1, 0, 0);

    const shadow_info: vk.DescriptorImageInfo = .{ .sampler = self.shadows.sampler, .image_view = self.shadows.image.view, .image_layout = .depth_read_only_optimal };
    cmd.pushDescriptorSet(.graphics, self.chunk_layout, 0, &.{.{
        .dst_set = .null_handle,
        .dst_binding = 0,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = .combined_image_sampler,
        .p_image_info = @ptrCast(&shadow_info),
        .p_buffer_info = undefined,
        .p_texel_buffer_view = undefined,
    }});
    cmd.bindPipeline(.graphics, self.chunk_pipeline);
    push.view = 0;
    cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
    cmd.drawIndirectCount(self.draws.handle, 0, self.draw_count.handle, 0, max_draws, @sizeOf(gpu.DrawCmd));

    if (in.target) |t| {
        const outline: OutlinePush = .{ .view_proj = view_proj, .block = .{ @floatFromInt(t.x), @floatFromInt(t.y), @floatFromInt(t.z) } };
        cmd.bindPipeline(.graphics, self.outline_pipeline);
        cmd.pushConstants(self.outline_layout, .{ .vertex_bit = true }, 0, @sizeOf(OutlinePush), &outline);
        cmd.draw(24, 1, 0, 0);
    }

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
    if (present == .suboptimal_khr or extent.width != self.requested_extent.width or extent.height != self.requested_extent.height)
        try self.recreate(extent);
}

fn writeFrameData(self: *Renderer, frame: *Frame, in: FrameInput, view_proj: zm.Mat, aspect: f32) void {
    var palette: [8][4]f32 = undefined;
    for (&palette, 0..) |*c, i| {
        const rgb = @as(world.Block, @enumFromInt(i)).color();
        c.* = .{ rgb[0], rgb[1], rgb[2], 1 };
    }
    const cam = in.camera;
    const l = in.light;
    const cascades = Shadows.fitCascades(self.shadows.resolution, cam.pos, cam.forward(), cam.fov_y, aspect, cam.near, l.dir);
    var cascade_vp: [Shadows.cascades]zm.Mat = undefined;
    var cascade_planes: [Shadows.cascades * 6][4]f32 = undefined;
    var cascade_texel: [4]f32 = .{ 0, 0, 0, 0 };
    for (cascades, 0..) |c, i| {
        cascade_vp[i] = c.view_proj;
        @memcpy(cascade_planes[i * 6 ..][0..6], &gpu.frustumPlanes(c.view_proj));
        cascade_texel[i] = c.texel;
    }

    const data: *gpu.FrameData = @ptrCast(@alignCast(frame.frame_data.mapped.?));
    data.* = .{
        .view_proj = view_proj,
        .planes = gpu.frustumPlanes(view_proj),
        .camera_pos = .{ cam.pos[0], cam.pos[1], cam.pos[2], 1 },
        .sun_dir = .{ l.dir[0], l.dir[1], l.dir[2], 0 },
        .sun_color = .{ l.color[0], l.color[1], l.color[2], 0 },
        .ambient = .{ l.ambient[0], l.ambient[1], l.ambient[2], 0 },
        .fog = .{ in.fog_start, in.fog_end, 0, 0 },
        .palette = palette,
        .cascade_vp = cascade_vp,
        .cascade_planes = cascade_planes,
        .cascade_splits = .{ Shadows.splits[0], Shadows.splits[1], Shadows.splits[2], 0 },
        .cascade_texel = cascade_texel,
        .shadow = .{ if (in.shadows) 1 else 0, 0, 0, 0 },
        .chunk_capacity = self.chunks.slot_high,
        .max_draws = max_draws,
    };
}

/// Rebuilds swapchain and depth buffer for `extent`. Atomic: on failure the
/// current ones stay valid.
fn recreate(self: *Renderer, extent: vk.Extent2D) !void {
    if (extent.width == 0 or extent.height == 0) return; // minimized
    try self.ctx.device.deviceWaitIdle();
    var swapchain = Swapchain.init(self.ctx, self.gpa, extent, self.swapchain.handle) catch |err| switch (err) {
        error.ZeroExtent => return, // minimized between the size query and now
        else => return err,
    };
    errdefer swapchain.deinit(self.ctx, self.gpa);
    const depth: Image = try .initDepth(self.ctx, swapchain.extent, depth_format);
    self.swapchain.deinit(self.ctx, self.gpa); // the old handle is retired, destroying it is valid
    self.depth.deinit(self.ctx);
    self.swapchain = swapchain;
    self.depth = depth;
    self.requested_extent = extent;
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
    const shaders = [_][]const u8{ "fullscreen.vert", "sky.frag", "cull.comp", "chunk.vert", "chunk.frag", "shadow.vert", "outline.vert", "outline.frag" };
    for (shaders) |name| {
        const glslc = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.4", "-O", "-o" });
        const spv = glslc.addOutputFileArg(b.fmt("{s}.spv", .{name}));
        glslc.addFileArg(b.path(b.fmt("src/render/shaders/{s}", .{name})));
        glslc.addFileInput(b.path("src/render/shaders/common.glsl"));
        glslc.addFileInput(b.path("src/render/shaders/gpu.glsl"));
        glslc.addFileInput(b.path("src/render/shaders/pull.glsl"));
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

- [ ] **Step 3: Aim and break in the main loop**

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
const Settings = @import("Settings.zig");

pub const std_options: std.Options = .{
    // Worker pool state changes are too chatty at debug level.
    .log_scope_levels = &.{.{ .scope = .worker_pool, .level = .info }},
};

const walk_speed: f32 = 12; // blocks per second
const sprint_factor: f32 = 6;
const mouse_sensitivity: f32 = 0.0025;
/// How far the player can reach to break a block.
const reach: f32 = 8;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const settings = Settings.parse(args[1..]) catch |err| {
        std.debug.print("ft_vox: {t}\n{s}", .{ err, Settings.usage });
        std.process.exit(2);
    };

    try glfw.init();
    defer glfw.terminate();
    glfw.windowHint(.client_api, .no_api);
    const window = try glfw.Window.create(1280, 720, "ft_vox", null, null);
    defer window.destroy();
    try glfw.setInputMode(window, .cursor, .disabled);

    var ctx: Context = try .init(gpa, window);
    defer ctx.deinit(gpa);
    var renderer: Renderer = try .init(gpa, &ctx, framebufferExtent(window), .{ .shadow_resolution = settings.shadow_resolution });
    defer renderer.deinit();

    // One core stays free for the render thread: oversubscribing starves it while loading.
    const workers = @max(2, (std.Thread.getCpuCount() catch 3) - 1);
    const mesh_workers = @max(1, workers / 3);
    const chunks = try ChunkManager.create(gpa, io, .{ .seed = settings.seed, .radius = settings.radius, .gen_workers = workers - mesh_workers, .mesh_workers = mesh_workers });
    defer chunks.destroy();

    var camera: Camera = .{};
    var sun: Sun = .{ .day_length = settings.day_length };
    var last_cursor = window.getCursorPos();
    var last_time = glfw.getTime();
    var title_timer: f64 = 0;
    var was_clicking = false;
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

        // Aim and break: left click edge, one block per click.
        const target = world.raycast(camera.pos, camera.forward(), reach, chunks);
        const clicking = window.getMouseButton(.left) == .press;
        if (clicking and !was_clicking) {
            if (target) |hit| _ = try chunks.breakBlock(hit.pos);
        }
        was_clicking = clicking;

        // Streaming: hand finished meshes to the GPU, uploads before unloads.
        try chunks.update(.{ .x = @intFromFloat(@floor(camera.pos[0])), .y = @intFromFloat(@floor(camera.pos[1])), .z = @intFromFloat(@floor(camera.pos[2])) });
        for (chunks.takeUploads()) |u| try renderer.chunks.upload(u.pos, u.mesh.quads, u.mesh.counts);
        for (chunks.takeUnloads()) |p| try renderer.chunks.remove(p);

        const extent = framebufferExtent(window);
        if (extent.width == 0 or extent.height == 0) continue;
        const far: f32 = @floatFromInt(@as(u32, settings.radius) * world.chunk_size);
        try renderer.drawFrame(extent, .{
            .camera = camera,
            .light = sun.lighting(),
            .shadows = sun.direction()[1] > 0.02,
            .target = if (target) |hit| hit.pos else null,
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
    _ = @import("Settings.zig");
}
```

- [ ] **Step 4: Build, test, run**

Run: `zig build test --summary all`
Expected: all tests pass (82).

Run: `zig build && timeout 7 ./zig-out/bin/ft_vox 2>&1 | grep -v worker_pool`
Expected: exactly these two lines and no `error(vulkan)` / `warning(vulkan)` line:

```
info(vulkan): validation layers: on
info(vulkan): GPU: <your GPU name>
```

Visual check (optional, `xdotool` + `ffmpeg`): start the engine, wait ~6 s, click in the window, look down (`xdotool mousemove_relative -- 0 30` × 14), hold `ctrl` ~1.5 s to get close to the ground, capture the window (the aimed block shows a dark outline), click 6 times (`xdotool click 1`, 0.35 s apart), capture again: a hole six blocks deep appears where the outline was.

- [ ] **Step 5: Commit**

```bash
zig fmt --check src build.zig
git add build.zig src/main.zig src/render
git commit -m "feat: block breaking with raycast, outline and coalesced remeshing

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## After this plan

The spec's milestones are complete. Candidates for a next spec (each needs its own brainstorming and spec revision before any code): block placing (same path as breaking), transparent water, per-vertex ambient occlusion (merge only faces with equal AO), saving edited chunks to disk, collisions.
