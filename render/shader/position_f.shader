#version 330 core

in VS_OUT
{
    vec3 pos;
    vec3 normal; 
    vec3 color; 
} fs_in;

out vec4 out_color;

uniform samplerCube depth_cube;       
uniform sampler2D object_texture;

uniform vec3 light_pos;
uniform float far_plane;
uniform vec3 view_pos;

uniform float material_ambient;
uniform float material_diffuse;
uniform float material_specular;
uniform float material_shininess;


void main()
{        
    out_color = vec4(fs_in.pos, 1); 
}

