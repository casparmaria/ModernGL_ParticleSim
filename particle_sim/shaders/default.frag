#version 430 core

// out
out vec4 fragColor;
// in
in vec2 uv_0;
uniform sampler2D texture0;

void main() {
    // get color from texture
    vec3 original_color = texture(texture0, uv_0).rgb;
    // color to fragment
    fragColor = vec4(original_color, 1.0);
}