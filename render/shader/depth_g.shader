#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

out vec4 gs_out;

uniform mat4 shadow_matrices[6];

void main()
{
    for(int face = 0; face < 6; ++face)
    {
        gl_Layer = face; 
        for(int i = 0; i < 3; i++) 
        {
            gs_out = gl_in[i].gl_Position;
            gl_Position = shadow_matrices[face] * gs_out;
            EmitVertex();
        }
        EndPrimitive();
    }
} 