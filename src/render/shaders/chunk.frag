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
