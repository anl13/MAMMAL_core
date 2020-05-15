#version 330 core

in GS_OUT
{
    vec3 pos;
    vec3 normal;
} fs_in;

out vec4 out_color;

uniform float far_plane;
uniform float near_plane;

float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));	
}

void main()
{            
    float depth = LinearizeDepth(gl_FragCoord.z) / far_plane;
    out_color = vec4(vec3(depth), 1.0);
}
