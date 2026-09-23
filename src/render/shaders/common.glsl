// Shared lighting helpers.
#extension GL_EXT_scalar_block_layout : require

vec3 skyColor(vec3 dir, vec3 sun) {
    float day = clamp(sun.y * 4.0 + 0.5, 0.0, 1.0);
    vec3 zenith = mix(vec3(0.01, 0.01, 0.04), vec3(0.20, 0.45, 0.90), day);
    vec3 horizon = mix(vec3(0.05, 0.05, 0.10), vec3(0.70, 0.80, 0.95), day);
    // Warm horizon at sunrise/sunset.
    float dusk = clamp(1.0 - abs(sun.y) * 5.0, 0.0, 1.0);
    horizon = mix(horizon, vec3(1.0, 0.45, 0.20), dusk * 0.6);
    float t = pow(clamp(dir.y, 0.0, 1.0), 0.5);
    vec3 col = mix(horizon, zenith, t);
    if (dir.y < 0.0) col = mix(horizon, horizon * 0.3, clamp(-dir.y * 3.0, 0.0, 1.0));
    float sd = max(dot(dir, sun), 0.0);
    col += vec3(1.0, 0.9, 0.7) * (smoothstep(0.9995, 0.9998, sd) * 20.0 + pow(sd, 64.0) * 0.5 * day);
    return col;
}

// ACES filmic approximation (Narkowicz).
vec3 tonemap(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
