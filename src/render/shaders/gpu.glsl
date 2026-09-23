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
