#version 330 core
layout (location = 0) in vec3 input_pos;
layout (location = 1) in vec3 input_color;

out VS_OUT
{
    vec3 pos;
    vec3 color;
} vs_out;
 
uniform mat4 model;

void main()
{
    vs_out.pos = vec3(model * vec4(input_pos, 1.0));
    vs_out.color = input_color;
}

