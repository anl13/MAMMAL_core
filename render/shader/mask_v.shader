#version 330 core
layout (location = 0) in vec3 input_pos;
layout (location = 1) in vec3 input_normal;

out VS_OUT
{
    vec3 pos;
    vec3 normal; 
} vs_out;
 
uniform mat4 model;
uniform mat4 projection; 
uniform mat4 view; 

void main()
{
    vs_out.pos = vec3(model * vec4(input_pos, 1.0));
    vs_out.normal = vec3(vec4(input_normal,1.0)); 
    gl_Position = projection * view * model * vec4(input_pos, 1.0); 
}

