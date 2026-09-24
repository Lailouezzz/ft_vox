#version 460
#extension GL_GOOGLE_include_directive : require
#include "lighting.glsl"

layout(location = 0) in vec3 world_pos;
layout(location = 1) flat in uint face;
layout(location = 2) flat in uint block;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 n = face_normals[face];
    vec3 sun = pc.frame.sun_dir.xyz;
    vec3 albedo = pc.frame.palette[block].rgb;
    vec3 to_frag = world_pos - pc.frame.camera_pos.xyz;
    float dist = length(to_frag);

    float ndl = max(dot(n, sun), 0.0);
    float lit = ndl > 0.0 ? sunVisibility(world_pos, n, dist) : 0.0;
    // Hemispheric ambient: brighter from the sky than from the ground.
    vec3 ambient = pc.frame.ambient.rgb * mix(0.5, 1.0, n.y * 0.5 + 0.5);
    vec3 color = albedo * (pc.frame.sun_color.rgb * ndl * lit + ambient);
    out_color = vec4(tonemap(applyFog(color, to_frag, dist)), 1.0);
}
