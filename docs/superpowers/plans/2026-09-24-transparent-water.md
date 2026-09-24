# ft_vox — transparent water (jalons W1–W3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transparent water with sky reflection (Fresnel), a lowered surface, depth-dependent opacity through Vulkan 1.4 dynamic rendering local read, received shadows, and an under-water view.

**Architecture:** The mesher puts water quads into their own 6 groups (12 groups per mesh) and flags surface water (bit 41 of the quad). The camera culling pass writes water draws into a fifth indirect range. Inside the single main render pass, after the opaque chunks, a by-region barrier makes the depth visible to the water pipeline, which reads it in place as an input attachment (`subpassLoad`) to compute the water thickness, and blends over the scene. Under water, fog and sky switch to deep blue.

**Tech Stack:** Zig 0.16.0, Vulkan 1.4 (`dynamicRenderingLocalRead` + `dynamicRenderingLocalReadDepthStencilAttachments`), GLSL/glslc. Spec: `docs/superpowers/specs/2026-09-23-voxel-engine-design.md` revision r5, section « Eau transparente (r5) ».

**Provenance:** the three patches were produced from a prototype built, tested (85/85) and run on the dev GPU (AMD Radeon Renoir, RADV) with validation layers on and no validation message; each state was replayed from a clean checkout of `ai`. Measured: 99 fps on the reference view (106 before water; target ≥ 95). Screenshots confirmed the sand floor visible through shallow water, darker deep water, terrain shadows on the water, and the blue under-water view.

## Global Constraints

- Zig 0.16.0; run every command from the repo root `/home/lailouezzz/Documents/git/ft_vox`.
- Vulkan 1.4; the device must expose `dynamicRenderingLocalRead` and `dynamicRenderingLocalReadDepthStencilAttachments` (checked at device selection).
- Reverse-Z everywhere (depth 1 near, `GREATER_OR_EQUAL`); the water pipeline tests depth but never writes it.
- `gpu.glsl` and `gpu.zig` stay byte-compatible; the `comptime` checks in `gpu.zig` must pass (`ChunkMeta` is 68 bytes).
- Quad groups: 0–5 opaque faces, 6–11 water faces, each in `Face` order; `firstInstance = slot * 16 + group`.
- `world` must not import `vulkan`, `zglfw` or `threading`.
- `zig fmt --check src build.zig` must pass before every commit; test logs at level `err` fail the test runner.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` (or the model that actually wrote the commit).

Capture helper (only the game window):

```bash
shot() { W=$(xdotool search --name '^ft_vox' | head -1); eval "$(xdotool getwindowgeometry --shell "$W")"; ffmpeg -loglevel error -f x11grab -video_size "${WIDTH}x${HEIGHT}" -i ":0+${X},${Y}" -frames:v 1 -y "$1"; }
```

---

### Task 16: Water groups, surface flag, fifth indirect range (W1)

**Files:**
- Modify: `src/world/mesher.zig` (12 groups, surface water type, `Quad.surface`, tests)
- Modify: `src/render/gpu.zig`, `src/render/shaders/gpu.glsl` (`ChunkMeta.counts[12]`, `Count.n[5]`, docs)
- Modify: `src/render/shaders/cull.comp` (12 groups; water → view 4), `src/render/shaders/pull.glsl` (`slot * 16`, surface drop)
- Modify: `src/render/ChunkBuffers.zig` (counts type), `src/render/Renderer.zig` (5 ranges, water drawn opaque for now, `fog.z = near`)

**Interfaces:**
- Produces: `mesher.groups = 12`, `mesher.water_group = 6`, `Quad.surface: bool`, `Mesh.counts: [12]u32`; `ChunkMeta.counts: [12]u32`; `FrameData.fog.z` = camera near plane; Renderer constants `water_view = 4`, `regions = 5`.

Surface water (a water block whose block above is not water) is a separate mesher-internal type, so it never merges with deep water: its quads carry `surface = true`, and the vertex shader lowers the top face and the top edge of side faces by 1/8 block. The coverage-based property test now also compares the surface flag and checks each quad sits in its group. After this task the water is still drawn opaque (with the chunk pipeline, from the new range), but its surface is visibly lowered.

- [ ] **Step 1: Apply the patch**

Save the patch below as `/tmp/task16.patch` (the lines between the fences, exactly), then run `git apply --check /tmp/task16.patch` (must print nothing) and `git apply /tmp/task16.patch`.

````diff
diff --git a/src/render/ChunkBuffers.zig b/src/render/ChunkBuffers.zig
index 1dbfdd0..54f6297 100644
--- a/src/render/ChunkBuffers.zig
+++ b/src/render/ChunkBuffers.zig
@@ -22,7 +22,7 @@ pub const max_chunks = 16 * 1024;
 pub const staging_size = 16 * 1024 * 1024;
 
 const Slot = struct { index: u32, range: world.FreeList.Range };
-const Pending = struct { pos: world.ChunkPos, quads: []Quad, counts: [6]u32 };
+const Pending = struct { pos: world.ChunkPos, quads: []Quad, counts: [world.mesher.groups]u32 };
 
 gpa: Allocator,
 quads: Buffer,
@@ -58,7 +58,7 @@ pub fn deinit(self: *ChunkBuffers, ctx: *const Context) void {
 }
 
 /// Queues `quads` (copied) as the new mesh of `pos`, replacing any older one.
-pub fn upload(self: *ChunkBuffers, pos: world.ChunkPos, quads: []const Quad, counts: [6]u32) !void {
+pub fn upload(self: *ChunkBuffers, pos: world.ChunkPos, quads: []const Quad, counts: [world.mesher.groups]u32) !void {
     self.dropPending(pos);
     const copy = try self.gpa.dupe(Quad, quads);
     errdefer self.gpa.free(copy);
diff --git a/src/render/Renderer.zig b/src/render/Renderer.zig
index beebc42..5267229 100644
--- a/src/render/Renderer.zig
+++ b/src/render/Renderer.zig
@@ -22,6 +22,10 @@ pub const frames_in_flight = 2;
 const depth_format: vk.Format = .d32_sfloat;
 /// Views culled each frame: the camera, then one per shadow cascade.
 const views = 1 + Shadows.cascades;
+/// Indirect range filled with the camera's water groups (after the culled views).
+const water_view = views;
+/// Indirect ranges and counters: one per culled view, plus water.
+const regions = views + 1;
 const max_draws = ChunkBuffers.max_chunks * 6;
 const chunk_stages: vk.ShaderStageFlags = .{ .compute_bit = true, .vertex_bit = true, .fragment_bit = true };
 
@@ -91,9 +95,9 @@ pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D, options: O
     errdefer chunks.deinit(ctx);
     var shadows: Shadows = try .init(ctx, options.shadow_resolution);
     errdefer shadows.deinit(ctx);
-    var draws: Buffer = try .init(ctx, views * max_draws * @sizeOf(gpu.DrawCmd), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true }, false);
+    var draws: Buffer = try .init(ctx, regions * max_draws * @sizeOf(gpu.DrawCmd), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true }, false);
     errdefer draws.deinit(ctx);
-    var draw_count: Buffer = try .init(ctx, views * @sizeOf(u32), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true, .transfer_dst_bit = true }, false);
+    var draw_count: Buffer = try .init(ctx, regions * @sizeOf(u32), .{ .storage_buffer_bit = true, .indirect_buffer_bit = true, .transfer_dst_bit = true }, false);
     errdefer draw_count.deinit(ctx);
 
     // The shadow map is bound with a push descriptor: no pool, no sets.
@@ -263,7 +267,7 @@ pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
         .count = self.draw_count.address,
         .view = 0,
     };
-    cmd.fillBuffer(self.draw_count.handle, 0, views * @sizeOf(u32), 0);
+    cmd.fillBuffer(self.draw_count.handle, 0, regions * @sizeOf(u32), 0);
     ChunkBuffers.barrier(cmd, .{ .copy_bit = true, .clear_bit = true }, .{ .transfer_write_bit = true }, .{ .compute_shader_bit = true, .vertex_shader_bit = true }, .{ .shader_storage_read_bit = true, .shader_storage_write_bit = true });
     cmd.bindPipeline(.compute, self.cull_pipeline);
     const groups = std.math.divCeil(u32, self.chunks.slot_high, 64) catch unreachable;
@@ -362,6 +366,10 @@ pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
     push.view = 0;
     cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
     cmd.drawIndirectCount(self.draws.handle, 0, self.draw_count.handle, 0, max_draws, @sizeOf(gpu.DrawCmd));
+    // Water groups, still drawn opaque with the chunk pipeline (transparency comes next).
+    push.view = water_view;
+    cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
+    cmd.drawIndirectCount(self.draws.handle, water_view * max_draws * @sizeOf(gpu.DrawCmd), self.draw_count.handle, water_view * @sizeOf(u32), max_draws, @sizeOf(gpu.DrawCmd));
 
     if (in.target) |t| {
         const outline: OutlinePush = .{ .view_proj = view_proj, .block = .{ @floatFromInt(t.x), @floatFromInt(t.y), @floatFromInt(t.z) } };
@@ -425,7 +433,7 @@ fn writeFrameData(self: *Renderer, frame: *Frame, in: FrameInput, view_proj: zm.
         .sun_dir = .{ l.dir[0], l.dir[1], l.dir[2], 0 },
         .sun_color = .{ l.color[0], l.color[1], l.color[2], 0 },
         .ambient = .{ l.ambient[0], l.ambient[1], l.ambient[2], 0 },
-        .fog = .{ in.fog_start, in.fog_end, 0, 0 },
+        .fog = .{ in.fog_start, in.fog_end, cam.near, 0 },
         .palette = palette,
         .cascade_vp = cascade_vp,
         .cascade_planes = cascade_planes,
diff --git a/src/render/gpu.zig b/src/render/gpu.zig
index 788edf5..66056a7 100644
--- a/src/render/gpu.zig
+++ b/src/render/gpu.zig
@@ -12,6 +12,7 @@ pub const FrameData = extern struct {
     sun_dir: [4]f32,
     sun_color: [4]f32,
     ambient: [4]f32,
+    /// x: fog start, y: fog end (blocks), z: camera near plane, w: 1 when the camera is under water.
     fog: [4]f32,
     palette: [8][4]f32,
     cascade_vp: [3]zm.Mat,
@@ -26,7 +27,8 @@ pub const FrameData = extern struct {
 pub const ChunkMeta = extern struct {
     origin: [3]i32,
     first_quad: u32,
-    counts: [6]u32,
+    /// Quads per group: 6 opaque faces, then 6 water faces (`world.mesher.groups`).
+    counts: [world.mesher.groups]u32,
     enabled: u32,
 };
 
@@ -44,14 +46,14 @@ pub const Push = extern struct {
     quads: u64,
     draws: u64,
     count: u64,
-    /// 0: camera, 1..3: shadow cascades.
+    /// Culling/drawing view: 0 camera (opaque), 1..3 shadow cascades, 4 camera (water).
     view: u32,
 };
 
 // Comptime layout checks: fail the build if these CPU mirrors drift from
 // src/render/shaders/gpu.glsl.
 comptime {
-    if (@sizeOf(ChunkMeta) != 44) @compileError("ChunkMeta size drifted from gpu.glsl's ChunkMeta");
+    if (@sizeOf(ChunkMeta) != 68) @compileError("ChunkMeta size drifted from gpu.glsl's ChunkMeta");
     if (@offsetOf(Push, "view") != 40) @compileError("Push.view offset drifted from gpu.glsl's Push");
     if (@offsetOf(FrameData, "chunk_capacity") != 896) @compileError("FrameData.chunk_capacity offset drifted from gpu.glsl's FrameData");
     if (@offsetOf(FrameData, "palette") != 240) @compileError("FrameData.palette offset drifted from gpu.glsl's FrameData");
diff --git a/src/render/shaders/cull.comp b/src/render/shaders/cull.comp
index 2a276fb..b6e68ea 100644
--- a/src/render/shaders/cull.comp
+++ b/src/render/shaders/cull.comp
@@ -2,10 +2,18 @@
 #extension GL_GOOGLE_include_directive : require
 #include "gpu.glsl"
 
-// One invocation per chunk slot, for view pc.view: frustum test, then one
-// indirect draw per face direction worth drawing for that view.
+// One invocation per chunk slot, for view pc.view (0: camera, 1..3: cascades):
+// frustum test, then one indirect draw per face group worth drawing. The camera
+// pass also writes the water groups into the water view's range (view 4).
 layout(local_size_x = 64) in;
 
+const uint water_view = 4u;
+
+void emit(uint view, uint n, uint first, uint instance) {
+    uint i = atomicAdd(pc.count.n[view], 1);
+    pc.draws.d[view * pc.frame.max_draws + i] = DrawCmd(n * 6, 1, first * 6, instance);
+}
+
 void main() {
     uint slot = gl_GlobalInvocationID.x;
     if (slot >= pc.frame.chunk_capacity) return;
@@ -15,8 +23,8 @@ void main() {
     vec3 lo = vec3(m.origin);
     vec3 hi = lo + 32.0;
     for (uint i = 0; i < 6; i++) {
-        // Skip the cascade's near plane (index 4): depth clamp keeps casters in front of
-        // it, but culling them here would drop casters between the sun and the cascade.
+        // Cascades keep casters behind their near plane: the shadow pipeline
+        // clamps their depth instead of clipping them.
         if (pc.view != 0 && i == 4) continue;
         vec4 p = pc.view == 0 ? pc.frame.planes[i] : pc.frame.cascade_planes[(pc.view - 1) * 6 + i];
         vec3 v = mix(lo, hi, greaterThan(p.xyz, vec3(0))); // corner furthest along the normal
@@ -33,12 +41,15 @@ void main() {
         for (uint f = 0; f < 6; f++) visible[f] = dot(face_normals[f], pc.frame.sun_dir.xyz) > 0.0;
     }
     uint first = m.first_quad;
-    uint base = pc.view * pc.frame.max_draws;
-    for (uint f = 0; f < 6; f++) {
-        uint n = m.counts[f];
-        if (n != 0 && visible[f]) {
-            uint i = atomicAdd(pc.count.n[pc.view], 1);
-            pc.draws.d[base + i] = DrawCmd(n * 6, 1, first * 6, slot * 8 + f);
+    for (uint g = 0; g < 12; g++) {
+        uint n = m.counts[g];
+        if (n != 0) {
+            if (g < 6) {
+                if (visible[g]) emit(pc.view, n, first, slot * 16 + g);
+            } else if (pc.view == 0) {
+                // Water: seen from both sides (from under water too), casts no shadow.
+                emit(water_view, n, first, slot * 16 + g);
+            }
         }
         first += n;
     }
diff --git a/src/render/shaders/gpu.glsl b/src/render/shaders/gpu.glsl
index 66f5a5f..b0233d9 100644
--- a/src/render/shaders/gpu.glsl
+++ b/src/render/shaders/gpu.glsl
@@ -11,7 +11,7 @@ layout(buffer_reference, scalar) readonly buffer FrameData {
     vec4 sun_dir;        // towards the light (sun by day, moon by night)
     vec4 sun_color;
     vec4 ambient;
-    vec4 fog;            // x: start, y: end (blocks)
+    vec4 fog;            // x: start, y: end (blocks), z: camera near plane, w: 1 under water
     vec4 palette[8];     // block albedo, indexed by Block
     mat4 cascade_vp[3];
     vec4 cascade_planes[18]; // 6 planes per cascade
@@ -25,7 +25,7 @@ layout(buffer_reference, scalar) readonly buffer FrameData {
 struct ChunkMeta {
     ivec3 origin;        // world block coordinates of the chunk's min corner
     uint first_quad;
-    uint counts[6];      // quads per face, in Face order
+    uint counts[12];     // quads per group: 6 opaque faces, then 6 water faces, in Face order
     uint enabled;
 };
 layout(buffer_reference, scalar) readonly buffer Metas { ChunkMeta m[]; };
@@ -33,7 +33,7 @@ layout(buffer_reference, scalar) readonly buffer Quads { uvec2 q[]; };
 
 struct DrawCmd { uint vertex_count; uint instance_count; uint first_vertex; uint first_instance; };
 layout(buffer_reference, scalar) writeonly buffer Draws { DrawCmd d[]; };
-layout(buffer_reference, scalar) buffer Count { uint n[4]; }; // one counter per view
+layout(buffer_reference, scalar) buffer Count { uint n[5]; }; // one counter per view
 
 // One push-constant block for every chunk pipeline.
 layout(push_constant, scalar) uniform Push {
@@ -42,7 +42,7 @@ layout(push_constant, scalar) uniform Push {
     Quads quads;
     Draws draws;
     Count count;
-    uint view;           // 0: camera, 1..3: shadow cascades
+    uint view;           // 0: camera (opaque), 1..3: shadow cascades, 4: camera (water)
 } pc;
 
 const vec3 face_normals[6] = vec3[6](
diff --git a/src/render/shaders/pull.glsl b/src/render/shaders/pull.glsl
index 934d7b3..b167b20 100644
--- a/src/render/shaders/pull.glsl
+++ b/src/render/shaders/pull.glsl
@@ -11,15 +11,19 @@ const vec2 corners[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 0),
 // in-plane axes so neighbouring quads overlap instead of meeting edge to edge.
 const float quad_inflate = 0.001;
 
+// How far the top of surface water sits below the block's top.
+const float water_drop = 0.125;
+
 // Vertex pulling: world position of this vertex, from gl_VertexIndex (quad and
-// corner) and gl_InstanceIndex (slot * 8 + face).
+// corner) and gl_InstanceIndex (slot * 16 + group).
 vec3 pullVertex(out uint face, out uint block) {
     uvec2 q = pc.quads.q[gl_VertexIndex / 6];
     uvec3 p = uvec3(q.x & 63u, (q.x >> 6) & 63u, (q.x >> 12) & 63u);
     vec2 size = vec2((q.x >> 18) & 63u, (q.x >> 24) & 63u);
     face = (q.x >> 30) | ((q.y & 1u) << 2);
     block = (q.y >> 1) & 255u;
-    ChunkMeta m = pc.metas.m[gl_InstanceIndex >> 3];
+    bool surface = ((q.y >> 9) & 1u) != 0u; // bit 41: water with no water above
+    ChunkMeta m = pc.metas.m[gl_InstanceIndex >> 4];
 
     // Width/height axes: ±X -> (z, y), ±Y -> (x, z), ±Z -> (x, y).
     uint axis = face >> 1;
@@ -32,6 +36,10 @@ vec3 pullVertex(out uint face, out uint block) {
     if ((axis == 2u) != positive) c = c.yx;
 
     vec3 base = vec3(p) + (positive ? face_normals[face] : vec3(0));
-    return vec3(m.origin) + base + u * (c.x * (size.x + 2 * quad_inflate) - quad_inflate) +
+    vec3 pos = vec3(m.origin) + base + u * (c.x * (size.x + 2 * quad_inflate) - quad_inflate) +
         v * (c.y * (size.y + 2 * quad_inflate) - quad_inflate);
+    // Surface water sits 1/8 block lower: its whole top face, and the top edge
+    // of its side faces (v runs along y for ±X and ±Z faces).
+    if (surface && (face == 2u || (axis != 1u && c.y == 1.0))) pos.y -= water_drop;
+    return pos;
 }
diff --git a/src/world/mesher.zig b/src/world/mesher.zig
index 400ddc1..5b4f841 100644
--- a/src/world/mesher.zig
+++ b/src/world/mesher.zig
@@ -28,14 +28,20 @@ pub const Quad = packed struct(u64) {
     h: u6, // 1..32
     face: Face,
     block: Block,
-    _pad: u23 = 0,
+    /// Water block with no water above it: the renderer lowers its top edge.
+    surface: bool = false,
+    _pad: u22 = 0,
 };
 
+/// Quad groups per mesh: the 6 opaque faces, then the 6 water faces, in `Face` order.
+pub const groups = 12;
+pub const water_group = 6;
+
 pub const Mesh = struct {
-    /// Quads sorted by face, in `Face` order.
+    /// Quads sorted by group: opaque faces then water faces, each in `Face` order.
     quads: []Quad,
-    /// Number of quads per face; face f starts at sum(counts[0..f]).
-    counts: [6]u32,
+    /// Number of quads per group; group g starts at sum(counts[0..g]).
+    counts: [groups]u32,
 
     pub fn deinit(m: *Mesh, gpa: Allocator) void {
         gpa.free(m.quads);
@@ -77,12 +83,17 @@ fn toXyz(face: Face, d: usize, u: usize, v: usize) [3]usize {
 }
 
 /// Scratch memory: per block type, per axis, a 34×34 grid of 34-bit columns.
+/// Mesher-internal types: every Block, plus "surface water" (water with no water
+/// above), kept apart from deep water so the two never merge into one quad.
+const types = block_count + 1;
+const surface_water = block_count;
+
 const Scratch = struct {
     /// cols[type][axis][v * 34 + u], bit i = block at depth i along the axis.
-    cols: [block_count][3][padded * padded]u64,
+    cols: [types][3][padded * padded]u64,
     solid: [3][padded * padded]u64,
     non_air: [3][padded * padded]u64,
-    present: [block_count]bool,
+    present: [types]bool,
 };
 
 threadlocal var scratch: Scratch = undefined;
@@ -90,8 +101,8 @@ threadlocal var scratch: Scratch = undefined;
 /// Concatenates the six per-face lists into one slice sorted by face, and takes
 /// ownership of `lists`: they are freed before this returns, on both the success
 /// and the allocation-failure path. Callers must not deinit/free `lists` themselves.
-fn fromLists(gpa: Allocator, lists: *[6]std.ArrayList(Quad)) Allocator.Error!Mesh {
-    var counts: [6]u32 = undefined;
+fn fromLists(gpa: Allocator, lists: *[groups]std.ArrayList(Quad)) Allocator.Error!Mesh {
+    var counts: [groups]u32 = undefined;
     var total: usize = 0;
     for (lists, 0..) |l, f| {
         counts[f] = @intCast(l.items.len);
@@ -120,7 +131,8 @@ pub fn mesh(gpa: Allocator, vol: *const Volume) Allocator.Error!Mesh {
     for (0..padded) |y| for (0..padded) |z| for (0..padded) |x| {
         const b = vol[volumeIndex(x, y, z)];
         if (b == .air) continue;
-        const t = @intFromEnum(b);
+        const is_surface = b == .water and y + 1 < padded and vol[volumeIndex(x, y + 1, z)] != .water;
+        const t: usize = if (is_surface) surface_water else @intFromEnum(b);
         if (!s.present[t]) {
             s.present[t] = true;
             @memset(std.mem.asBytes(&s.cols[t]), 0);
@@ -134,23 +146,25 @@ pub fn mesh(gpa: Allocator, vol: *const Volume) Allocator.Error!Mesh {
     // 2. Occluder masks: OR of the present types' columns.
     @memset(std.mem.asBytes(&s.solid), 0);
     @memset(std.mem.asBytes(&s.non_air), 0);
-    for (0..block_count) |t| {
+    for (0..types) |t| {
         if (!s.present[t]) continue;
-        const solid = (@as(Block, @enumFromInt(t))).isSolid();
+        const solid = t != surface_water and (@as(Block, @enumFromInt(t))).isSolid();
         for (0..3) |axis| for (0..padded * padded) |i| {
             s.non_air[axis][i] |= s.cols[t][axis][i];
             if (solid) s.solid[axis][i] |= s.cols[t][axis][i];
         };
     }
 
-    var lists: [6]std.ArrayList(Quad) = @splat(.empty);
+    var lists: [groups]std.ArrayList(Quad) = @splat(.empty);
     {
         errdefer for (&lists) |*l| l.deinit(gpa);
 
         const inner: u64 = ((@as(u64, 1) << cs) - 1) << 1; // bits 1..32
-        for (0..block_count) |t| {
-            if (!s.present[t] or t == @intFromEnum(Block.air)) continue;
-            const block: Block = @enumFromInt(t);
+        for (0..types) |t| {
+            if (!s.present[t]) continue;
+            const surface = t == surface_water;
+            const block: Block = if (surface) .water else @enumFromInt(t);
+            const group_base: usize = if (block == .water) water_group else 0;
             for (0..6) |f| {
                 const face: Face = @enumFromInt(f);
                 const axis = f / 2;
@@ -172,7 +186,7 @@ pub fn mesh(gpa: Allocator, vol: *const Volume) Allocator.Error!Mesh {
                     }
                 };
                 if (!any) continue;
-                for (&planes, 0..) |*plane, d| try greedyPlane(gpa, &lists[f], plane, face, block, d);
+                for (&planes, 0..) |*plane, d| try greedyPlane(gpa, &lists[group_base + f], plane, face, block, surface, d);
             }
         }
     }
@@ -181,7 +195,7 @@ pub fn mesh(gpa: Allocator, vol: *const Volume) Allocator.Error!Mesh {
 }
 
 /// Greedy-merges one 32×32 binary plane (rows = v, bits = u) into rectangles.
-fn greedyPlane(gpa: Allocator, out: *std.ArrayList(Quad), plane: *[cs]u32, face: Face, block: Block, d: usize) Allocator.Error!void {
+fn greedyPlane(gpa: Allocator, out: *std.ArrayList(Quad), plane: *[cs]u32, face: Face, block: Block, surface: bool, d: usize) Allocator.Error!void {
     for (0..cs) |v| {
         while (plane[v] != 0) {
             const row: u64 = plane[v];
@@ -200,6 +214,7 @@ fn greedyPlane(gpa: Allocator, out: *std.ArrayList(Quad), plane: *[cs]u32, face:
                 .h = @intCast(h),
                 .face = face,
                 .block = block,
+                .surface = surface,
             });
         }
     }
@@ -207,7 +222,7 @@ fn greedyPlane(gpa: Allocator, out: *std.ArrayList(Quad), plane: *[cs]u32, face:
 
 /// Reference mesher: one quad per visible face. Slow, obviously correct; used by tests.
 pub fn meshNaive(gpa: Allocator, vol: *const Volume) Allocator.Error!Mesh {
-    var lists: [6]std.ArrayList(Quad) = @splat(.empty);
+    var lists: [groups]std.ArrayList(Quad) = @splat(.empty);
     {
         errdefer for (&lists) |*l| l.deinit(gpa);
         for (0..cs) |y| for (0..cs) |z| for (0..cs) |x| {
@@ -223,7 +238,9 @@ pub fn meshNaive(gpa: Allocator, vol: *const Volume) Allocator.Error!Mesh {
                     )
                 ];
                 if (!b.faceVisible(nb)) continue;
-                try lists[f].append(gpa, .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z), .w = 1, .h = 1, .face = face, .block = b });
+                const surface = b == .water and vol[volumeIndex(x + 1, y + 2, z + 1)] != .water;
+                const group = if (b == .water) water_group + f else f;
+                try lists[group].append(gpa, .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z), .w = 1, .h = 1, .face = face, .block = b, .surface = surface });
             }
         };
     }
@@ -243,9 +260,22 @@ fn setInner(vol: *Volume, x: usize, y: usize, z: usize, b: Block) void {
 
 /// Expands quads to unit faces: covered[face][block index] = block type, and
 /// fails if any unit face is covered twice.
-fn coverage(gpa: Allocator, m: Mesh) ![]Block {
-    const covered = try gpa.alloc(Block, 6 * Chunk.volume);
-    @memset(covered, .air);
+/// Expands quads to unit faces: covered[face][block index] = block type in the low
+/// byte, 0x100 when flagged as water surface. Fails if a unit face is covered twice
+/// or a quad sits in the wrong group.
+fn coverage(gpa: Allocator, m: Mesh) ![]u16 {
+    const covered = try gpa.alloc(u16, 6 * Chunk.volume);
+    errdefer gpa.free(covered);
+    @memset(covered, 0);
+    var start: usize = 0;
+    for (m.counts, 0..) |n, g| {
+        for (m.quads[start..][0..n]) |q| {
+            const f: usize = @intFromEnum(q.face);
+            const expected = if (q.block == .water) water_group + f else f;
+            if (g != expected) return error.WrongGroup;
+        }
+        start += n;
+    }
     for (m.quads) |q| {
         for (0..q.h) |dv| for (0..q.w) |du| {
             const base = toXyz(q.face, 0, du, dv);
@@ -253,8 +283,8 @@ fn coverage(gpa: Allocator, m: Mesh) ![]Block {
             const y = q.y + base[1];
             const z = q.z + base[2];
             const i = @as(usize, @intFromEnum(q.face)) * Chunk.volume + x + cs * z + cs * cs * y;
-            if (covered[i] != .air) return error.Overlap;
-            covered[i] = q.block;
+            if (covered[i] != 0) return error.Overlap;
+            covered[i] = @intFromEnum(q.block) | (@as(u16, @intFromBool(q.surface)) << 8);
         };
     }
     return covered;
@@ -272,7 +302,7 @@ test "single block gives 6 quads" {
     var m = try mesh(testing.allocator, &vol);
     defer m.deinit(testing.allocator);
     try testing.expectEqual(6, m.quads.len);
-    try testing.expectEqual([6]u32{ 1, 1, 1, 1, 1, 1 }, m.counts);
+    try testing.expectEqual([groups]u32{ 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 }, m.counts);
 }
 
 test "full chunk gives 6 quads of 32x32" {
@@ -409,14 +439,17 @@ test "greedy covers exactly the naive faces on random volumes" {
         defer gpa.free(a);
         const b = try coverage(gpa, naive);
         defer gpa.free(b);
-        try testing.expectEqualSlices(Block, b, a);
+        try testing.expectEqualSlices(u16, b, a);
         try testing.expect(greedy.quads.len <= naive.quads.len);
     }
 }
 
-/// Quads for `face`, as sorted into `m.quads` (grouped by `Face` order).
+/// Opaque quads for `face`, as sorted into `m.quads`.
 fn quadsForFace(m: Mesh, face: Face) []Quad {
-    const f = @intFromEnum(face);
+    return quadsInGroup(m, @intFromEnum(face));
+}
+
+fn quadsInGroup(m: Mesh, f: usize) []Quad {
     var start: usize = 0;
     for (0..f) |i| start += m.counts[i];
     return m.quads[start..][0..m.counts[f]];
@@ -496,6 +529,48 @@ test "greedy merges on real terrain" {
     defer gpa.free(a);
     const b = try coverage(gpa, naive);
     defer gpa.free(b);
-    try testing.expectEqualSlices(Block, b, a);
+    try testing.expectEqualSlices(u16, b, a);
     try testing.expect(greedy.quads.len * 2 < naive.quads.len);
 }
+
+test "water goes to its own groups; surface water is flagged and never merged with deep water" {
+    var vol: Volume = @splat(.air);
+    // A 3-block water column: y = 0 and 1 are deep, y = 2 has air above (surface).
+    for (0..3) |y| setInner(&vol, 5, y, 5, .water);
+    var m = try mesh(testing.allocator, &vol);
+    defer m.deinit(testing.allocator);
+    for (0..water_group) |g| try testing.expectEqual(0, m.counts[g]);
+
+    const top = quadsInGroup(m, water_group + @as(usize, @intFromEnum(Face.pos_y)));
+    try testing.expectEqual(1, top.len);
+    try testing.expect(top[0].surface);
+    try testing.expectEqual(2, top[0].y);
+
+    const bottom = quadsInGroup(m, water_group + @as(usize, @intFromEnum(Face.neg_y)));
+    try testing.expectEqual(1, bottom.len);
+    try testing.expect(!bottom[0].surface);
+
+    // Side: one deep quad two blocks tall, one surface quad on top.
+    const side = quadsInGroup(m, water_group + @as(usize, @intFromEnum(Face.pos_x)));
+    try testing.expectEqual(2, side.len);
+    for (side) |q| {
+        if (q.surface) {
+            try testing.expectEqual(2, q.y);
+            try testing.expectEqual(1, q.h);
+        } else {
+            try testing.expectEqual(0, q.y);
+            try testing.expectEqual(2, q.h);
+        }
+    }
+}
+
+test "water under a solid block counts as surface water" {
+    var vol: Volume = @splat(.air);
+    setInner(&vol, 5, 5, 5, .water);
+    setInner(&vol, 5, 6, 5, .stone);
+    var m = try mesh(testing.allocator, &vol);
+    defer m.deinit(testing.allocator);
+    for (quadsInGroup(m, water_group + @as(usize, @intFromEnum(Face.pos_x)))) |q| try testing.expect(q.surface);
+    // Its top face touches stone, which hides it.
+    try testing.expectEqual(0, quadsInGroup(m, water_group + @as(usize, @intFromEnum(Face.pos_y))).len);
+}
````

- [ ] **Step 2: Test**

Run: `zig build test --summary all` — all tests pass (85 at this point).

- [ ] **Step 3: Run and check**

Run: `zig build && timeout 7 ./zig-out/bin/ft_vox 2>&1 | grep -v worker_pool`
Expected: exactly these two lines and no `error(vulkan)` / `warning(vulkan)` line:

```
info(vulkan): validation layers: on
info(vulkan): GPU: <your GPU name>
```

Visual (optional): water surfaces sit slightly below the surrounding sand; everything else is unchanged.

- [ ] **Step 4: Commit**

```bash
zig fmt --check src build.zig
git add -A src build.zig
git commit -m "feat(water): separate water quad groups, surface flag, fifth indirect range

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 17: Transparent water with dynamic rendering local read (W2)

**Files:**
- Modify: `src/render/Context.zig` (enable `dynamicRenderingLocalRead`, check the depth local-read property)
- Modify: `src/render/pipeline.zig` (`blend`, `reads_depth`, input attachment mappings)
- Create: `src/render/shaders/lighting.glsl`, `src/render/shaders/water.frag`
- Modify: `src/render/shaders/chunk.frag` (uses `lighting.glsl`), `src/render/Renderer.zig` (water pipeline, local-read pass), `build.zig` (`water.frag`, `lighting.glsl` input)

**Interfaces:**
- Consumes: Task 16's water range and groups.
- Produces: `pipeline.GraphicsDesc.blend`, `.reads_depth`; `pipeline.depth_input_mapping`, `pipeline.default_input_mapping`; `lighting.glsl` (`shadow_map`, `sunVisibility(world_pos, n, dist)`, `applyFog(color, to_frag, dist)`); push-descriptor binding 1 = scene depth (input attachment).

How the pass works: the depth image is created with `input_attachment` usage and stays in `RENDERING_LOCAL_READ` for the whole main pass. After the opaque chunks, `vkCmdPipelineBarrier2` with `BY_REGION` (depth writes → fragment input-attachment reads, old = new layout) runs inside the rendering; `vkCmdSetRenderingInputAttachmentIndices` maps the depth to input attachment 0 (colour unused), matching the water pipeline's `RenderingInputAttachmentIndexInfo`; the water draws with blending, depth test without writes and no culling; the default mapping is restored before the outline. `water.frag` linearises both depths (reverse-Z infinite: distance = near / depth) to get the water thickness, which drives opacity and colour; Schlick Fresnel (F0 = 0.02) blends in the sky colour; the sun adds a specular highlight; shadows are sampled as for chunks.

- [ ] **Step 1: Apply the patch**

Save the patch below as `/tmp/task17.patch` (the lines between the fences, exactly), then run `git apply --check /tmp/task17.patch` (must print nothing) and `git apply /tmp/task17.patch`.

````diff
diff --git a/build.zig b/build.zig
index e2775fa..956c914 100644
--- a/build.zig
+++ b/build.zig
@@ -49,7 +49,7 @@ pub fn build(b: *std.Build) void {
     exe_mod.linkLibrary(zglfw.artifact("glfw"));
 
     // Shaders: GLSL -> SPIR-V with glslc, embedded with @embedFile("<name>").
-    const shaders = [_][]const u8{ "fullscreen.vert", "sky.frag", "cull.comp", "chunk.vert", "chunk.frag", "shadow.vert", "outline.vert", "outline.frag" };
+    const shaders = [_][]const u8{ "fullscreen.vert", "sky.frag", "cull.comp", "chunk.vert", "chunk.frag", "shadow.vert", "outline.vert", "outline.frag", "water.frag" };
     for (shaders) |name| {
         const glslc = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.4", "-O", "-o" });
         const spv = glslc.addOutputFileArg(b.fmt("{s}.spv", .{name}));
@@ -57,6 +57,7 @@ pub fn build(b: *std.Build) void {
         glslc.addFileInput(b.path("src/render/shaders/common.glsl"));
         glslc.addFileInput(b.path("src/render/shaders/gpu.glsl"));
         glslc.addFileInput(b.path("src/render/shaders/pull.glsl"));
+        glslc.addFileInput(b.path("src/render/shaders/lighting.glsl"));
         exe_mod.addAnonymousImport(name, .{ .root_source_file = spv });
     }
 
diff --git a/src/render/Context.zig b/src/render/Context.zig
index 5ae3838..3e87cbf 100644
--- a/src/render/Context.zig
+++ b/src/render/Context.zig
@@ -81,7 +81,7 @@ pub fn init(gpa: Allocator, window: *glfw.Window) !Context {
     log.info("GPU: {s}", .{std.mem.sliceTo(&self.props.device_name, 0)});
 
     // Device: one graphics+present queue, the features the renderer relies on.
-    var f14: vk.PhysicalDeviceVulkan14Features = .{ .push_descriptor = .true, .maintenance_5 = .true };
+    var f14: vk.PhysicalDeviceVulkan14Features = .{ .push_descriptor = .true, .maintenance_5 = .true, .dynamic_rendering_local_read = .true };
     var f13: vk.PhysicalDeviceVulkan13Features = .{ .p_next = &f14, .dynamic_rendering = .true, .synchronization_2 = .true, .maintenance_4 = .true };
     var f12: vk.PhysicalDeviceVulkan12Features = .{ .p_next = &f13, .buffer_device_address = .true, .draw_indirect_count = .true, .timeline_semaphore = .true, .scalar_block_layout = .true };
     const f10: vk.PhysicalDeviceFeatures2 = .{ .p_next = &f12, .features = .{ .shader_int_64 = .true, .multi_draw_indirect = .true, .draw_indirect_first_instance = .true, .depth_clamp = .true } };
@@ -171,6 +171,7 @@ fn missingFeature(instance: vk.InstanceProxy, pdev: vk.PhysicalDevice) ?[]const
     const checks = .{
         .{ f14.push_descriptor, "pushDescriptor" },
         .{ f14.maintenance_5, "maintenance5" },
+        .{ f14.dynamic_rendering_local_read, "dynamicRenderingLocalRead" },
         .{ f13.dynamic_rendering, "dynamicRendering" },
         .{ f13.synchronization_2, "synchronization2" },
         .{ f13.maintenance_4, "maintenance4" },
@@ -184,6 +185,35 @@ fn missingFeature(instance: vk.InstanceProxy, pdev: vk.PhysicalDevice) ?[]const
         .{ f10.features.depth_clamp, "depthClamp" },
     };
     inline for (checks) |c| if (c[0] != .true) return c[1];
+    // Transparent water reads the depth attachment in place.
+    var p14: vk.PhysicalDeviceVulkan14Properties = .{
+        .line_sub_pixel_precision_bits = 0,
+        .max_vertex_attrib_divisor = 0,
+        .supports_non_zero_first_instance = .false,
+        .max_push_descriptors = 0,
+        .dynamic_rendering_local_read_depth_stencil_attachments = .false,
+        .dynamic_rendering_local_read_multisampled_attachments = .false,
+        .early_fragment_multisample_coverage_after_sample_counting = .false,
+        .early_fragment_sample_mask_test_before_sample_counting = .false,
+        .depth_stencil_swizzle_one_support = .false,
+        .polygon_mode_point_size = .false,
+        .non_strict_single_pixel_wide_lines_use_parallelogram = .false,
+        .non_strict_wide_lines_use_parallelogram = .false,
+        .block_texel_view_compatible_multiple_layers = .false,
+        .max_combined_image_sampler_descriptor_count = 0,
+        .fragment_shading_rate_clamp_combiner_inputs = .false,
+        .default_robustness_storage_buffers = .device_default,
+        .default_robustness_uniform_buffers = .device_default,
+        .default_robustness_vertex_inputs = .device_default,
+        .default_robustness_images = .device_default,
+        .copy_src_layout_count = 0,
+        .copy_dst_layout_count = 0,
+        .optimal_tiling_layout_uuid = @splat(0),
+        .identical_memory_type_requirements = .false,
+    };
+    var p2: vk.PhysicalDeviceProperties2 = .{ .p_next = &p14, .properties = undefined };
+    instance.getPhysicalDeviceProperties2(pdev, &p2);
+    if (p14.dynamic_rendering_local_read_depth_stencil_attachments != .true) return "dynamicRenderingLocalReadDepthStencilAttachments";
     return null;
 }
 
diff --git a/src/render/Renderer.zig b/src/render/Renderer.zig
index 5267229..7676439 100644
--- a/src/render/Renderer.zig
+++ b/src/render/Renderer.zig
@@ -73,6 +73,7 @@ chunk_layout: vk.PipelineLayout,
 cull_pipeline: vk.Pipeline,
 chunk_pipeline: vk.Pipeline,
 shadow_pipeline: vk.Pipeline,
+water_pipeline: vk.Pipeline,
 outline_layout: vk.PipelineLayout,
 outline_pipeline: vk.Pipeline,
 
@@ -80,7 +81,7 @@ pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D, options: O
     const d = ctx.device;
     var swapchain: Swapchain = try .init(ctx, gpa, extent, .null_handle);
     errdefer swapchain.deinit(ctx, gpa);
-    var depth: Image = try .initDepth(ctx, swapchain.extent, depth_format);
+    var depth: Image = try createDepth(ctx, swapchain.extent);
     errdefer depth.deinit(ctx);
 
     var frames: [frames_in_flight]Frame = undefined;
@@ -101,8 +102,12 @@ pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D, options: O
     errdefer draw_count.deinit(ctx);
 
     // The shadow map is bound with a push descriptor: no pool, no sets.
-    const binding: vk.DescriptorSetLayoutBinding = .{ .binding = 0, .descriptor_type = .combined_image_sampler, .descriptor_count = 1, .stage_flags = .{ .fragment_bit = true } };
-    const set_layout = try d.createDescriptorSetLayout(&.{ .flags = .{ .push_descriptor_bit = true }, .binding_count = 1, .p_bindings = @ptrCast(&binding) }, null);
+    // Binding 1: the scene depth, read in place by the water pass (input attachment).
+    const bindings = [_]vk.DescriptorSetLayoutBinding{
+        .{ .binding = 0, .descriptor_type = .combined_image_sampler, .descriptor_count = 1, .stage_flags = .{ .fragment_bit = true } },
+        .{ .binding = 1, .descriptor_type = .input_attachment, .descriptor_count = 1, .stage_flags = .{ .fragment_bit = true } },
+    };
+    const set_layout = try d.createDescriptorSetLayout(&.{ .flags = .{ .push_descriptor_bit = true }, .binding_count = bindings.len, .p_bindings = &bindings }, null);
     errdefer d.destroyDescriptorSetLayout(set_layout, null);
 
     const sky_layout = try pipeline.createLayout(ctx, @sizeOf(SkyPush), .{ .fragment_bit = true }, &.{});
@@ -142,6 +147,20 @@ pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D, options: O
         .depth_bias = true,
     });
     errdefer d.destroyPipeline(shadow_pipeline, null);
+    // Transparent water: blended over the opaque scene, depth-tested but not
+    // written, both sides visible (from under water too).
+    const water_pipeline = try pipeline.createGraphics(ctx, .{
+        .layout = chunk_layout,
+        .vertex = pipeline.spirv("chunk.vert"),
+        .fragment = pipeline.spirv("water.frag"),
+        .color_format = swapchain.format,
+        .depth_format = depth_format,
+        .depth_write = false,
+        .cull_back = false,
+        .blend = true,
+        .reads_depth = true,
+    });
+    errdefer d.destroyPipeline(water_pipeline, null);
     const outline_layout = try pipeline.createLayout(ctx, @sizeOf(OutlinePush), .{ .vertex_bit = true }, &.{});
     errdefer d.destroyPipelineLayout(outline_layout, null);
     const outline_pipeline = try pipeline.createGraphics(ctx, .{
@@ -173,6 +192,7 @@ pub fn init(gpa: Allocator, ctx: *const Context, extent: vk.Extent2D, options: O
         .cull_pipeline = cull_pipeline,
         .chunk_pipeline = chunk_pipeline,
         .shadow_pipeline = shadow_pipeline,
+        .water_pipeline = water_pipeline,
         .outline_layout = outline_layout,
         .outline_pipeline = outline_pipeline,
     };
@@ -183,6 +203,7 @@ pub fn deinit(self: *Renderer) void {
     d.deviceWaitIdle() catch {};
     d.destroyPipeline(self.outline_pipeline, null);
     d.destroyPipelineLayout(self.outline_layout, null);
+    d.destroyPipeline(self.water_pipeline, null);
     d.destroyPipeline(self.shadow_pipeline, null);
     d.destroyPipeline(self.chunk_pipeline, null);
     d.destroyPipeline(self.cull_pipeline, null);
@@ -316,7 +337,8 @@ pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
     // 3. Main pass: sky, then chunks.
     const image = self.swapchain.images[image_index];
     imageBarrier(cmd, image, .{ .color_bit = true }, .undefined, .color_attachment_optimal, .{ .color_attachment_output_bit = true }, .{}, .{ .color_attachment_output_bit = true }, .{ .color_attachment_write_bit = true });
-    imageBarrier(cmd, self.depth.image, .{ .depth_bit = true }, .undefined, .depth_attachment_optimal, .{ .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true });
+    // The depth stays in RENDERING_LOCAL_READ for the whole pass: the water reads it in place.
+    imageBarrier(cmd, self.depth.image, .{ .depth_bit = true }, .undefined, .rendering_local_read, .{ .late_fragment_tests_bit = true, .fragment_shader_bit = true }, .{ .depth_stencil_attachment_write_bit = true }, .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true, .fragment_shader_bit = true }, .{ .depth_stencil_attachment_write_bit = true, .depth_stencil_attachment_read_bit = true, .input_attachment_read_bit = true });
     const color_att: vk.RenderingAttachmentInfo = .{
         .image_view = self.swapchain.views[image_index],
         .image_layout = .color_attachment_optimal,
@@ -328,7 +350,7 @@ pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
     };
     const depth_att: vk.RenderingAttachmentInfo = .{
         .image_view = self.depth.view,
-        .image_layout = .depth_attachment_optimal,
+        .image_layout = .rendering_local_read,
         .resolve_mode = .{},
         .resolve_image_layout = .undefined,
         .load_op = .clear,
@@ -352,7 +374,8 @@ pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
     cmd.draw(3, 1, 0, 0);
 
     const shadow_info: vk.DescriptorImageInfo = .{ .sampler = self.shadows.sampler, .image_view = self.shadows.image.view, .image_layout = .depth_read_only_optimal };
-    cmd.pushDescriptorSet(.graphics, self.chunk_layout, 0, &.{.{
+    const depth_info: vk.DescriptorImageInfo = .{ .sampler = .null_handle, .image_view = self.depth.view, .image_layout = .rendering_local_read };
+    cmd.pushDescriptorSet(.graphics, self.chunk_layout, 0, &.{ .{
         .dst_set = .null_handle,
         .dst_binding = 0,
         .dst_array_element = 0,
@@ -361,15 +384,42 @@ pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
         .p_image_info = @ptrCast(&shadow_info),
         .p_buffer_info = undefined,
         .p_texel_buffer_view = undefined,
-    }});
+    }, .{
+        .dst_set = .null_handle,
+        .dst_binding = 1,
+        .dst_array_element = 0,
+        .descriptor_count = 1,
+        .descriptor_type = .input_attachment,
+        .p_image_info = @ptrCast(&depth_info),
+        .p_buffer_info = undefined,
+        .p_texel_buffer_view = undefined,
+    } });
     cmd.bindPipeline(.graphics, self.chunk_pipeline);
     push.view = 0;
     cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
     cmd.drawIndirectCount(self.draws.handle, 0, self.draw_count.handle, 0, max_draws, @sizeOf(gpu.DrawCmd));
-    // Water groups, still drawn opaque with the chunk pipeline (transparency comes next).
+
+    // Water, inside the same pass: make this pass's depth writes visible to the
+    // water's in-place depth reads (by region: each pixel only reads itself).
+    const local_read: vk.ImageMemoryBarrier2 = .{
+        .src_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
+        .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },
+        .dst_stage_mask = .{ .fragment_shader_bit = true },
+        .dst_access_mask = .{ .input_attachment_read_bit = true },
+        .old_layout = .rendering_local_read,
+        .new_layout = .rendering_local_read,
+        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
+        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
+        .image = self.depth.image,
+        .subresource_range = .{ .aspect_mask = .{ .depth_bit = true }, .base_mip_level = 0, .level_count = 1, .base_array_layer = 0, .layer_count = 1 },
+    };
+    cmd.pipelineBarrier2(&.{ .dependency_flags = .{ .by_region_bit = true }, .image_memory_barrier_count = 1, .p_image_memory_barriers = @ptrCast(&local_read) });
+    cmd.setRenderingInputAttachmentIndices(&pipeline.depth_input_mapping);
+    cmd.bindPipeline(.graphics, self.water_pipeline);
     push.view = water_view;
     cmd.pushConstants(self.chunk_layout, chunk_stages, 0, @sizeOf(gpu.Push), &push);
     cmd.drawIndirectCount(self.draws.handle, water_view * max_draws * @sizeOf(gpu.DrawCmd), self.draw_count.handle, water_view * @sizeOf(u32), max_draws, @sizeOf(gpu.DrawCmd));
+    cmd.setRenderingInputAttachmentIndices(&pipeline.default_input_mapping);
 
     if (in.target) |t| {
         const outline: OutlinePush = .{ .view_proj = view_proj, .block = .{ @floatFromInt(t.x), @floatFromInt(t.y), @floatFromInt(t.z) } };
@@ -460,7 +510,7 @@ fn recreate(self: *Renderer, extent: vk.Extent2D) !void {
         else => return err,
     };
     errdefer swapchain.deinit(self.ctx, self.gpa);
-    const depth: Image = try .initDepth(self.ctx, swapchain.extent, depth_format);
+    const depth: Image = try createDepth(self.ctx, swapchain.extent);
     self.swapchain.deinit(self.ctx, self.gpa); // the old handle is retired, destroying it is valid
     self.depth.deinit(self.ctx);
     self.swapchain = swapchain;
@@ -468,6 +518,11 @@ fn recreate(self: *Renderer, extent: vk.Extent2D) !void {
     self.requested_extent = extent;
 }
 
+/// Scene depth: a depth attachment the water pass also reads as input attachment.
+fn createDepth(ctx: *const Context, extent: vk.Extent2D) !Image {
+    return .init(ctx, extent, depth_format, .{ .depth_stencil_attachment_bit = true, .input_attachment_bit = true }, .{ .depth_bit = true }, 1);
+}
+
 fn setViewport(cmd: vk.CommandBufferProxy, ext: vk.Extent2D) void {
     cmd.setViewport(0, &.{.{ .x = 0, .y = 0, .width = @floatFromInt(ext.width), .height = @floatFromInt(ext.height), .min_depth = 0, .max_depth = 1 }});
     cmd.setScissor(0, &.{.{ .offset = .{ .x = 0, .y = 0 }, .extent = ext }});
diff --git a/src/render/pipeline.zig b/src/render/pipeline.zig
index bfd8716..06b30de 100644
--- a/src/render/pipeline.zig
+++ b/src/render/pipeline.zig
@@ -22,8 +22,25 @@ pub const GraphicsDesc = struct {
     cull_back: bool = true,
     topology: vk.PrimitiveTopology = .triangle_list,
     depth_bias: bool = false,
+    /// Standard alpha blending (src alpha, 1 - src alpha) on the color attachment.
+    blend: bool = false,
+    /// Reads the depth attachment as input attachment 0 (dynamic rendering local read).
+    reads_depth: bool = false,
 };
 
+/// Input attachment mapping of a pipeline that reads the depth attachment:
+/// no color attachment is readable, depth is input attachment 0. Must match
+/// what `vkCmdSetRenderingInputAttachmentIndices` sets before drawing with it.
+pub const depth_input_indices = [_]u32{vk.ATTACHMENT_UNUSED};
+pub const depth_input_index: u32 = 0;
+pub const depth_input_mapping: vk.RenderingInputAttachmentIndexInfo = .{
+    .color_attachment_count = depth_input_indices.len,
+    .p_color_attachment_input_indices = &depth_input_indices,
+    .p_depth_input_attachment_index = &depth_input_index,
+};
+/// The default mapping (color i -> input i, depth not readable), restored after such draws.
+pub const default_input_mapping: vk.RenderingInputAttachmentIndexInfo = .{ .color_attachment_count = 1 };
+
 pub fn createGraphics(ctx: *const Context, d: GraphicsDesc) !vk.Pipeline {
     var modules: [2]vk.ShaderModuleCreateInfo = undefined;
     var stages: [2]vk.PipelineShaderStageCreateInfo = undefined;
@@ -36,6 +53,7 @@ pub fn createGraphics(ctx: *const Context, d: GraphicsDesc) !vk.Pipeline {
     }
     const color_formats: []const vk.Format = if (d.color_format) |*f| f[0..1] else &.{};
     const rendering: vk.PipelineRenderingCreateInfo = .{
+        .p_next = if (d.reads_depth) &depth_input_mapping else null,
         .view_mask = 0,
         .color_attachment_count = @intCast(color_formats.len),
         .p_color_attachment_formats = color_formats.ptr,
@@ -44,12 +62,12 @@ pub fn createGraphics(ctx: *const Context, d: GraphicsDesc) !vk.Pipeline {
     };
     const dynamic = [_]vk.DynamicState{ .viewport, .scissor, .depth_bias };
     const blend = vk.PipelineColorBlendAttachmentState{
-        .blend_enable = .false,
-        .src_color_blend_factor = .one,
-        .dst_color_blend_factor = .zero,
+        .blend_enable = if (d.blend) .true else .false,
+        .src_color_blend_factor = if (d.blend) .src_alpha else .one,
+        .dst_color_blend_factor = if (d.blend) .one_minus_src_alpha else .zero,
         .color_blend_op = .add,
         .src_alpha_blend_factor = .one,
-        .dst_alpha_blend_factor = .zero,
+        .dst_alpha_blend_factor = if (d.blend) .one_minus_src_alpha else .zero,
         .alpha_blend_op = .add,
         .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
     };
diff --git a/src/render/shaders/chunk.frag b/src/render/shaders/chunk.frag
index ad429a2..2ffc693 100644
--- a/src/render/shaders/chunk.frag
+++ b/src/render/shaders/chunk.frag
@@ -1,32 +1,12 @@
 #version 460
 #extension GL_GOOGLE_include_directive : require
-#include "gpu.glsl"
-
-layout(set = 0, binding = 0) uniform sampler2DArrayShadow shadow_map;
+#include "lighting.glsl"
 
 layout(location = 0) in vec3 world_pos;
 layout(location = 1) flat in uint face;
 layout(location = 2) flat in uint block;
 layout(location = 0) out vec4 out_color;
 
-// 1 = fully lit, 0 = in shadow. PCF 3x3 on the cascade covering this fragment.
-float sunVisibility(vec3 n, float dist) {
-    if (pc.frame.shadow.x == 0.0) return 1.0;
-    uint c = dist < pc.frame.cascade_splits.x ? 0u : dist < pc.frame.cascade_splits.y ? 1u : dist < pc.frame.cascade_splits.z ? 2u : 3u;
-    if (c == 3u) return 1.0;
-    // Normal offset: push the lookup out of the surface by ~1.5 texels.
-    vec3 p = world_pos + n * pc.frame.cascade_texel[c] * 1.5;
-    vec4 lp = pc.frame.cascade_vp[c] * vec4(p, 1.0);
-    vec3 ndc = lp.xyz / lp.w;
-    vec2 uv = ndc.xy * 0.5 + 0.5;
-    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0).xy);
-    float lit = 0.0;
-    for (int y = -1; y <= 1; y++)
-        for (int x = -1; x <= 1; x++)
-            lit += texture(shadow_map, vec4(uv + vec2(x, y) * texel, float(c), ndc.z));
-    return lit / 9.0;
-}
-
 void main() {
     vec3 n = face_normals[face];
     vec3 sun = pc.frame.sun_dir.xyz;
@@ -35,12 +15,9 @@ void main() {
     float dist = length(to_frag);
 
     float ndl = max(dot(n, sun), 0.0);
-    float lit = ndl > 0.0 ? sunVisibility(n, dist) : 0.0;
+    float lit = ndl > 0.0 ? sunVisibility(world_pos, n, dist) : 0.0;
     // Hemispheric ambient: brighter from the sky than from the ground.
     vec3 ambient = pc.frame.ambient.rgb * mix(0.5, 1.0, n.y * 0.5 + 0.5);
     vec3 color = albedo * (pc.frame.sun_color.rgb * ndl * lit + ambient);
-
-    float fog = smoothstep(pc.frame.fog.x, pc.frame.fog.y, dist);
-    color = mix(color, skyColor(normalize(to_frag), sun), fog);
-    out_color = vec4(tonemap(color), 1.0);
+    out_color = vec4(tonemap(applyFog(color, to_frag, dist)), 1.0);
 }
diff --git a/src/render/shaders/lighting.glsl b/src/render/shaders/lighting.glsl
new file mode 100644
index 0000000..8ee02dd
--- /dev/null
+++ b/src/render/shaders/lighting.glsl
@@ -0,0 +1,29 @@
+// Lighting shared by the chunk and water fragment shaders.
+#include "gpu.glsl"
+
+layout(set = 0, binding = 0) uniform sampler2DArrayShadow shadow_map;
+
+// 1 = fully lit, 0 = in shadow. PCF 3x3 on the cascade covering this fragment.
+float sunVisibility(vec3 world_pos, vec3 n, float dist) {
+    if (pc.frame.shadow.x == 0.0) return 1.0;
+    uint c = dist < pc.frame.cascade_splits.x ? 0u : dist < pc.frame.cascade_splits.y ? 1u : dist < pc.frame.cascade_splits.z ? 2u : 3u;
+    if (c == 3u) return 1.0;
+    // Normal offset: push the lookup out of the surface by ~1.5 texels.
+    vec3 p = world_pos + n * pc.frame.cascade_texel[c] * 1.5;
+    vec4 lp = pc.frame.cascade_vp[c] * vec4(p, 1.0);
+    vec3 ndc = lp.xyz / lp.w;
+    vec2 uv = ndc.xy * 0.5 + 0.5;
+    vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0).xy);
+    float lit = 0.0;
+    for (int y = -1; y <= 1; y++)
+        for (int x = -1; x <= 1; x++)
+            lit += texture(shadow_map, vec4(uv + vec2(x, y) * texel, float(c), ndc.z));
+    return lit / 9.0;
+}
+
+
+// Fog towards the sky.
+vec3 applyFog(vec3 color, vec3 to_frag, float dist) {
+    float fog = smoothstep(pc.frame.fog.x, pc.frame.fog.y, dist);
+    return mix(color, skyColor(normalize(to_frag), pc.frame.sun_dir.xyz), fog);
+}
diff --git a/src/render/shaders/water.frag b/src/render/shaders/water.frag
new file mode 100644
index 0000000..b57c4e3
--- /dev/null
+++ b/src/render/shaders/water.frag
@@ -0,0 +1,44 @@
+#version 460
+#extension GL_GOOGLE_include_directive : require
+#include "lighting.glsl"
+
+// Depth of the opaque scene at this pixel, read in place (dynamic rendering local read).
+layout(input_attachment_index = 0, set = 0, binding = 1) uniform subpassInput scene_depth;
+
+layout(location = 0) in vec3 world_pos;
+layout(location = 1) flat in uint face;
+layout(location = 2) flat in uint block;
+layout(location = 0) out vec4 out_color;
+
+const vec3 shallow = vec3(0.10, 0.45, 0.55);
+const vec3 deep = vec3(0.01, 0.08, 0.20);
+// How fast water gets opaque with thickness (per block).
+const float absorption = 0.18;
+
+void main() {
+    // Seen from below (under water) the face's back side faces the camera.
+    vec3 n = face_normals[face] * (gl_FrontFacing ? 1.0 : -1.0);
+    vec3 sun = pc.frame.sun_dir.xyz;
+    vec3 to_frag = world_pos - pc.frame.camera_pos.xyz;
+    float dist = length(to_frag);
+    vec3 view = -to_frag / dist;
+
+    // Reverse-Z infinite perspective: depth = near / view distance.
+    float near = pc.frame.fog.z;
+    float floor_depth = max(subpassLoad(scene_depth).r, 1e-7);
+    float thickness = max(near / floor_depth - near / gl_FragCoord.z, 0.0);
+    float murk = 1.0 - exp(-thickness * absorption);
+
+    float ndl = max(dot(n, sun), 0.0);
+    float lit = ndl > 0.0 ? sunVisibility(world_pos, n, dist) : 0.0;
+    vec3 light = pc.frame.sun_color.rgb * ndl * lit + pc.frame.ambient.rgb;
+    vec3 body = mix(shallow, deep, murk) * light;
+
+    // Schlick Fresnel: clear when looked at from above, a sky mirror at grazing angles.
+    float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, view), 0.0), 5.0);
+    vec3 sky = skyColor(reflect(-view, n), sun);
+    vec3 spec = pc.frame.sun_color.rgb * lit * pow(max(dot(reflect(-sun, n), view), 0.0), 256.0);
+    vec3 color = mix(body, sky, fresnel) + spec;
+    float alpha = clamp(mix(0.25, 0.95, murk) + fresnel, 0.0, 1.0);
+    out_color = vec4(tonemap(applyFog(color, to_frag, dist)), alpha);
+}
````

- [ ] **Step 2: Test**

Run: `zig build test --summary all` — all tests pass (85 at this point).

- [ ] **Step 3: Run and check**

Run: `zig build && timeout 7 ./zig-out/bin/ft_vox 2>&1 | grep -v worker_pool`
Expected: exactly these two lines and no `error(vulkan)` / `warning(vulkan)` line:

```
info(vulkan): validation layers: on
info(vulkan): GPU: <your GPU name>
```

Performance: the window title of the default view must show at least 95 fps on the dev GPU.

Visual (optional, xdotool + `shot`): from the spawn, click in the window, look down (`xdotool mousemove_relative -- 0 25` × 10), hold `w` ~1.4 s then `ctrl` ~1.6 s and capture: the sand floor shows through shallow water near the shore, deeper water is darker blue, terrain shadows fall on the water.

- [ ] **Step 4: Commit**

```bash
zig fmt --check src build.zig
git add -A src build.zig
git commit -m "feat(water): transparent water via dynamic rendering local read, Fresnel sky reflection

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 18: Under-water view (W3)

**Files:**
- Modify: `src/render/shaders/common.glsl` (`underwater_fog`), `src/render/shaders/lighting.glsl` (under-water fog branch), `src/render/shaders/sky.frag` (blue sky under water)
- Modify: `src/render/Renderer.zig` (`FrameInput.underwater`, `fog.w`, sky push `sun_dir.w`), `src/main.zig` (asks the ChunkManager if the camera block is water)

**Interfaces:**
- Consumes: `ChunkManager.blockAt`, Task 17's `applyFog`.
- Produces: `Renderer.FrameInput.underwater: bool`; `FrameData.fog.w` = 1 under water; sky push `sun_dir.w` = 1 under water.

Under water, chunks and water fade into a deep blue within 24 blocks and the sky is replaced by that blue; the water surface seen from below uses the flipped normal (`gl_FrontFacing`, already in `water.frag`).

- [ ] **Step 1: Apply the patch**

Save the patch below as `/tmp/task18.patch` (the lines between the fences, exactly), then run `git apply --check /tmp/task18.patch` (must print nothing) and `git apply /tmp/task18.patch`.

````diff
diff --git a/src/main.zig b/src/main.zig
index 0c1171a..48472f2 100644
--- a/src/main.zig
+++ b/src/main.zig
@@ -114,6 +114,7 @@ pub fn main(init: std.process.Init) !void {
             .light = sun.lighting(),
             .shadows = sun.direction()[1] > 0.02,
             .target = if (target) |hit| hit.pos else null,
+            .underwater = chunks.blockAt(.{ .x = @intFromFloat(@floor(camera.pos[0])), .y = @intFromFloat(@floor(camera.pos[1])), .z = @intFromFloat(@floor(camera.pos[2])) }) == .water,
             .fog_start = far * 0.6,
             .fog_end = far * 0.95,
         });
diff --git a/src/render/Renderer.zig b/src/render/Renderer.zig
index 7676439..7fe180b 100644
--- a/src/render/Renderer.zig
+++ b/src/render/Renderer.zig
@@ -50,6 +50,8 @@ pub const FrameInput = struct {
     shadows: bool,
     /// Block to outline (the one the player aims at).
     target: ?world.BlockPos,
+    /// The camera is inside a water block: blue fog and tint.
+    underwater: bool,
     fog_start: f32,
     fog_end: f32,
 };
@@ -368,7 +370,7 @@ pub fn drawFrame(self: *Renderer, extent: vk.Extent2D, in: FrameInput) !void {
     setViewport(cmd, ext);
 
     const d_sun = in.light.dir;
-    const sky: SkyPush = .{ .inv_view_proj = zm.inverse(view_proj), .sun_dir = .{ d_sun[0], d_sun[1], d_sun[2], 0 } };
+    const sky: SkyPush = .{ .inv_view_proj = zm.inverse(view_proj), .sun_dir = .{ d_sun[0], d_sun[1], d_sun[2], if (in.underwater) 1 else 0 } };
     cmd.bindPipeline(.graphics, self.sky_pipeline);
     cmd.pushConstants(self.sky_layout, .{ .fragment_bit = true }, 0, @sizeOf(SkyPush), &sky);
     cmd.draw(3, 1, 0, 0);
@@ -483,7 +485,7 @@ fn writeFrameData(self: *Renderer, frame: *Frame, in: FrameInput, view_proj: zm.
         .sun_dir = .{ l.dir[0], l.dir[1], l.dir[2], 0 },
         .sun_color = .{ l.color[0], l.color[1], l.color[2], 0 },
         .ambient = .{ l.ambient[0], l.ambient[1], l.ambient[2], 0 },
-        .fog = .{ in.fog_start, in.fog_end, cam.near, 0 },
+        .fog = .{ in.fog_start, in.fog_end, cam.near, if (in.underwater) 1 else 0 },
         .palette = palette,
         .cascade_vp = cascade_vp,
         .cascade_planes = cascade_planes,
diff --git a/src/render/shaders/common.glsl b/src/render/shaders/common.glsl
index c5ccdbd..b1931ff 100644
--- a/src/render/shaders/common.glsl
+++ b/src/render/shaders/common.glsl
@@ -20,3 +20,6 @@ vec3 skyColor(vec3 dir, vec3 sun) {
 vec3 tonemap(vec3 x) {
     return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
 }
+
+// Fog colour when the camera is under water.
+const vec3 underwater_fog = vec3(0.04, 0.18, 0.28);
diff --git a/src/render/shaders/lighting.glsl b/src/render/shaders/lighting.glsl
index 8ee02dd..86de42c 100644
--- a/src/render/shaders/lighting.glsl
+++ b/src/render/shaders/lighting.glsl
@@ -21,9 +21,11 @@ float sunVisibility(vec3 world_pos, vec3 n, float dist) {
     return lit / 9.0;
 }
 
+const float underwater_fog_end = 24.0;
 
-// Fog towards the sky.
+// Fog towards the sky, or towards deep blue when the camera is under water.
 vec3 applyFog(vec3 color, vec3 to_frag, float dist) {
+    if (pc.frame.fog.w != 0.0) return mix(color, underwater_fog, smoothstep(0.0, underwater_fog_end, dist));
     float fog = smoothstep(pc.frame.fog.x, pc.frame.fog.y, dist);
     return mix(color, skyColor(normalize(to_frag), pc.frame.sun_dir.xyz), fog);
 }
diff --git a/src/render/shaders/sky.frag b/src/render/shaders/sky.frag
index 93c5bfd..f2e7e74 100644
--- a/src/render/shaders/sky.frag
+++ b/src/render/shaders/sky.frag
@@ -7,7 +7,7 @@ layout(location = 0) out vec4 out_color;
 
 layout(push_constant, scalar) uniform Push {
     mat4 inv_view_proj;
-    vec4 sun_dir; // xyz: direction towards the sun
+    vec4 sun_dir; // xyz: direction towards the sun, w: 1 under water
 } pc;
 
 void main() {
@@ -15,5 +15,7 @@ void main() {
     vec4 far = pc.inv_view_proj * vec4(ndc, 1e-6, 1.0);
     vec4 near = pc.inv_view_proj * vec4(ndc, 1.0, 1.0);
     vec3 dir = normalize(far.xyz / far.w - near.xyz / near.w);
-    out_color = vec4(tonemap(skyColor(dir, pc.sun_dir.xyz)), 1.0);
+    // sun_dir.w = 1 when the camera is under water: the sky is lost in blue fog.
+    vec3 sky = pc.sun_dir.w != 0.0 ? underwater_fog : skyColor(dir, pc.sun_dir.xyz);
+    out_color = vec4(tonemap(sky), 1.0);
 }
````

- [ ] **Step 2: Test**

Run: `zig build test --summary all` — all tests pass (85 at this point).

- [ ] **Step 3: Run and check**

Run: `zig build && timeout 7 ./zig-out/bin/ft_vox 2>&1 | grep -v worker_pool`
Expected: exactly these two lines and no `error(vulkan)` / `warning(vulkan)` line:

```
info(vulkan): validation layers: on
info(vulkan): GPU: <your GPU name>
```

Visual (optional): fly into a lake (for example move towards low terrain and hold `ctrl` until the title shows a y below 62 inside water) and capture: blue fog, the lake floor with shadows, the surface seen from below as a lighter band.

- [ ] **Step 4: Commit**

```bash
zig fmt --check src build.zig
git add -A src build.zig
git commit -m "feat(water): under-water fog and sky

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## After this plan

Candidates for later specs: waves (animated normals), screen-space reflections, block placing, per-vertex ambient occlusion, saving edited chunks.
