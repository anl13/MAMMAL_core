#version 330 core

in GS_OUT
{
    vec3 pos;
    vec3 color;
    vec3 normal;
} fs_in;

out vec4 out_color;

void main()
{        
    out_color =  vec4(fs_in.color, 1.0);
}

