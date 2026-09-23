#version 460
// Fullscreen triangle, no vertex buffer.
layout(location = 0) out vec2 ndc;

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    ndc = uv * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
