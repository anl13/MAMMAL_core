#version 330 core

in GS_OUT
{
    vec3 pos;
    vec3 normal;
} fs_in;

out vec4 out_color;

uniform samplerCube depth_cube;       
uniform vec3 object_color;

uniform vec3 light_pos;
uniform float far_plane;
uniform vec3 view_pos;

void main()
{            
    /// render normal 
    out_color = vec4(fs_in.normal, 1.0); 
}
