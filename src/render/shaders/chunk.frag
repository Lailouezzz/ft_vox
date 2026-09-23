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
