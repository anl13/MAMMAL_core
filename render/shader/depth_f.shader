#version 330 core
in vec4 gs_out;

uniform vec3 light_pos;
uniform float far_plane;

void main()
{
    gl_FragDepth = length(gs_out.xyz - light_pos)/far_plane;
}