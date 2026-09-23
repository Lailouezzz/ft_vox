#version 460
#extension GL_GOOGLE_include_directive : require
#include "pull.glsl"

// Depth-only pass into one shadow cascade (pc.view = 1 + cascade index).
void main() {
    uint face, block;
    vec3 world_pos = pullVertex(face, block);
    gl_Position = pc.frame.cascade_vp[pc.view - 1] * vec4(world_pos, 1.0);
}
