#version 330 core
layout (location = 0) in vec3 input_pos;
layout (location = 1) in vec2 input_texcoord;
layout (location = 2) in vec3 input_normal; 

out VS_OUT
{
    vec3 pos;
    vec2 texcoord;
    vec3 normal; 
} vs_out;
 
uniform mat4 model;
uniform mat4 projection; 
uniform mat4 view; 

void main()
{
    vs_out.pos = vec3(model * vec4(input_pos, 1.0));
    vs_out.texcoord = input_texcoord;
    vs_out.normal = input_normal; 

    gl_Position = projection * view * model * vec4(input_pos, 1.0);
}

