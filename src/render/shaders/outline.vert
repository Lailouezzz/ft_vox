#version 460
#extension GL_EXT_scalar_block_layout : require

// The 12 edges of the targeted block, as a line list (24 vertices).
layout(push_constant, scalar) uniform Push {
    mat4 view_proj;
    vec3 block;
} pc;

const uint edges[24] = uint[24](0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7);

void main() {
    uint c = edges[gl_VertexIndex];
    vec3 corner = vec3(c & 1u, (c >> 2) & 1u, (c >> 1) & 1u);
    // Slightly larger than the block so the lines are not hidden by its faces.
    vec3 p = pc.block + 0.5 + (corner - 0.5) * 1.004;
    gl_Position = pc.view_proj * vec4(p, 1.0);
}
