#version 430 core

// buffer objects
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_texcoord;
// out
out vec2 uv_0;
// in
uniform mat4 m_projection;
uniform mat4 m_view;
uniform mat4 m_model;

void main() {
    // forward to fragment shader
    uv_0 = in_texcoord;
    gl_Position = m_projection * m_view * m_model * vec4(in_position, 1.0);
}