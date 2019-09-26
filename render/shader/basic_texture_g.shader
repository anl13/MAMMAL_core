#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in VS_OUT
{
    vec3 pos;
    vec2 texcoord;
} gs_in[];

out GS_OUT
{
    vec3 pos;
    vec2 texcoord;
    vec3 normal;
} gs_out;

uniform mat4 perspective;
uniform mat4 proj;

void main()
{
    vec3 v0 = gs_in[0].pos - gs_in[1].pos;
    vec3 v1 = gs_in[2].pos - gs_in[1].pos;
    vec3 normal = cross(v1, v0);
    normal = normalize(normal);

    for(int i = 0; i < 3; i++)
    {
        gs_out.pos = gs_in[i].pos;
        gs_out.texcoord = gs_in[i].texcoord;
        gs_out.normal = normal;
        gl_Position = perspective*proj*vec4(gs_in[i].pos,1.0);
        EmitVertex();
    }
    EndPrimitive();
}



