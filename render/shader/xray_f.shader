#version 330 core

in VS_OUT
{
    vec3 pos;
    vec3 normal;
    vec3 I; 
} fs_in;

out vec4 out_color;
   
uniform vec3 object_color;
uniform vec3 light_pos;
uniform float far_plane;
uniform vec3 view_pos;

void main()
{
    float edgefalloff = 0.5; 
    float intensity = 0.5; 
    float ambient = 0.6;
    float opac = dot(normalize(-fs_in.normal), normalize(-fs_in.I));
    opac = abs(opac);
    opac= ambient + intensity * (1.0 - pow(opac, edgefalloff));

    out_color = opac * vec4(object_color, 1);  
}