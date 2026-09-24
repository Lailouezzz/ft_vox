#version 460
#extension GL_GOOGLE_include_directive : require
#include "lighting.glsl"

// Depth of the opaque scene at this pixel, read in place (dynamic rendering local read).
layout(input_attachment_index = 0, set = 0, binding = 1) uniform subpassInput scene_depth;

layout(location = 0) in vec3 world_pos;
layout(location = 1) flat in uint face;
layout(location = 2) flat in uint block;
layout(location = 0) out vec4 out_color;

const vec3 shallow = vec3(0.10, 0.45, 0.55);
const vec3 deep = vec3(0.01, 0.08, 0.20);
// How fast water gets opaque with thickness (per block).
const float absorption = 0.18;

void main() {
    // Seen from below (under water) the face's back side faces the camera.
    vec3 n = face_normals[face] * (gl_FrontFacing ? 1.0 : -1.0);
    vec3 sun = pc.frame.sun_dir.xyz;
    vec3 to_frag = world_pos - pc.frame.camera_pos.xyz;
    float dist = length(to_frag);
    vec3 view = -to_frag / dist;

    // Reverse-Z infinite perspective: depth = near / forward view distance.
    float near = pc.frame.fog.z;
    float thickness;
    if (pc.frame.fog.w != 0.0) {
        // Camera under water: this face is the surface seen from below, so
        // the water between the camera and it is just the view-ray distance
        // -- there is no further (opaque) depth sample behind it to diff.
        thickness = dist;
    } else {
        float floor_depth = max(subpassLoad(scene_depth).r, 1e-7);
        thickness = max(near / floor_depth - near / gl_FragCoord.z, 0.0);
        thickness *= dist * gl_FragCoord.z / near;
    }
    float murk = 1.0 - exp(-thickness * absorption);

    float ndl = max(dot(n, sun), 0.0);
    float lit = ndl > 0.0 ? sunVisibility(world_pos, n, dist) : 0.0;
    vec3 light = pc.frame.sun_color.rgb * ndl * lit + pc.frame.ambient.rgb;
    vec3 body = mix(shallow, deep, murk) * light;

    // Schlick Fresnel: clear when looked at from above, a sky mirror at grazing angles.
    float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, view), 0.0), 5.0);
    vec3 sky = skyColor(reflect(-view, n), sun);
    vec3 spec = pc.frame.sun_color.rgb * lit * pow(max(dot(reflect(-sun, n), view), 0.0), 256.0);
    vec3 color = mix(body, sky, fresnel) + spec;
    float alpha = clamp(mix(0.25, 0.95, murk) + fresnel, 0.0, 1.0);
    alpha = clamp(alpha + dot(spec, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    out_color = vec4(tonemap(applyFog(color, to_frag, dist)), alpha);
}
