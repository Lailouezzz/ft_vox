// Lighting shared by the chunk and water fragment shaders.
#include "gpu.glsl"

layout(set = 0, binding = 0) uniform sampler2DArrayShadow shadow_map;

// 1 = fully lit, 0 = in shadow. PCF 3x3 on the cascade covering this fragment.
float sunVisibility(vec3 world_pos, vec3 n, float dist) {
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

const float underwater_fog_end = 24.0;

// Fog towards the sky, or towards deep blue when the camera is under water.
vec3 applyFog(vec3 color, vec3 to_frag, float dist) {
    if (pc.frame.fog.w != 0.0) return mix(color, underwater_fog, smoothstep(0.0, underwater_fog_end, dist));
    float fog = smoothstep(pc.frame.fog.x, pc.frame.fog.y, dist);
    return mix(color, skyColor(normalize(to_frag), pc.frame.sun_dir.xyz), fog);
}
