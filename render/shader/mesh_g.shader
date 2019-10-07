#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in VS_OUT
{
    vec3 pos;
    vec3 color;
} gs_in[];

out GS_OUT
{
    vec3 pos;
    vec3 color;
    vec3 normal;
} gs_out;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    vec3 v0 = gs_in[0].pos - gs_in[1].pos;
    vec3 v1 = gs_in[2].pos - gs_in[1].pos;
    vec3 normal = cross(v1, v0);
    normal = normalize(normal);

    for(int i = 0; i < 3; i++)
    {
        gs_out.pos = gs_in[i].pos;
        gs_out.color = gs_in[i].color;
        gs_out.normal = normal;
        gl_Position = projection*view*vec4(gs_in[i].pos,1.0);
        EmitVertex();
    }
    EndPrimitive();
}



