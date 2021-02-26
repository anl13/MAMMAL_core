#version 330 core

in VS_OUT
{
    vec3 pos;
    vec2 texcoord;
    vec3 normal;
} fs_in;

out vec4 out_color;

uniform samplerCube depth_cube;       
uniform sampler2D object_texture;

uniform vec3 light_pos;
uniform float far_plane;
uniform vec3 view_pos;

vec4 get_max(vec4 color1, vec4 color2)
{
    vec4 result; 
    result[0] = max(color1[0], color2[0]);
    result[1] = max(color1[1], color2[1]);
    result[2] = max(color1[2], color2[2]);
    result[3] = max(color1[3], color2[3]); 
    return result; 
}

void main()
{        
    out_color = texture(object_texture, fs_in.texcoord);
}

