#version 330 core
layout (location = 0) in vec3 input_pos;

out VS_OUT
{
    vec3 pos;
} vs_out;
 
uniform mat4 model;

void main()
{
    vs_out.pos = vec3(model * vec4(input_pos, 1.0));
}

