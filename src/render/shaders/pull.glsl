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
