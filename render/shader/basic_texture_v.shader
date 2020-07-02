#version 330 core
layout (location = 0) in vec3 input_pos;
layout (location = 1) in vec2 input_texcoord;

out VS_OUT
{
    vec3 pos;
    vec2 texcoord;
} vs_out;
 
uniform mat4 model;

void main()
{
    vs_out.pos = vec3(model * vec4(input_pos, 1.0));
    vs_out.texcoord = input_texcoord;
}

