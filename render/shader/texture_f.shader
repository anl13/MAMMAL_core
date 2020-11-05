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
    float shadow = 0.0;
    // ambient
    float material_ambient = 0.3;
    float material_diffuse = 0.6;
    float material_specular = 0.002;
    float material_shininess = 1;
    // diffuse 
    vec3 lights[5];
    lights[0] = vec3(2,0,2.5);
    lights[1] = vec3(-2,0,2.5);
    lights[2] = vec3(0,-2,2.5);
    lights[3] = vec3(0,2,2.5);
    lights[4] = vec3(0,0,3);
    vec4 result = vec4(0,0,0,0);
    for(int i = 0; i < 5; i++)
    {
        vec3 light_dir = normalize(lights[i] - fs_in.pos);
        float diff = max(dot(fs_in.normal, light_dir), 0.0);
        float diffuse = diff * material_diffuse;  
        
        // specular
        vec3 view_dir = normalize(view_pos - fs_in.pos);
        vec3 reflect_dir = reflect(-light_dir, fs_in.normal);  
        // vec3 halfway_dir = normalize(light_dir + view_dir);
        // float spec = pow(max(dot(fs_in.normal, halfway_dir), 0.0), 32.0/(32));
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 1);
        float specular = spec * material_specular;  

        // sum
        vec4 local = (material_ambient + (1.0 - shadow) * (diffuse + specular)) *  texture(object_texture, fs_in.texcoord);
        result = get_max(result, local); 
    }
    out_color = result ;
}

