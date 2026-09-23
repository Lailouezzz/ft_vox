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
