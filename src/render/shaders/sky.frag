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
