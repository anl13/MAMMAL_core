#version 330 core

in GS_OUT
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

uniform float material_ambient;
uniform float material_diffuse;
uniform float material_specular;
uniform float material_shininess;

// array of offset direction for sampling
vec3 grid_sampling_disk[20] = vec3[]
(
   vec3(1, 1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, 1,  1), 
   vec3(1, 1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
   vec3(1, 1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1, 1,  0),
   vec3(1, 0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1, 0, -1),
   vec3(0, 1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0, 1, -1)
);


void main()
{        


    // ambient
    float ambient = material_ambient;
  	
    // diffuse 
    vec3 light_dir = normalize(light_pos - fs_in.pos);
    float diff = max(dot(fs_in.normal, light_dir), 0.0);
    float diffuse = diff * material_diffuse;  
    
    // specular
    vec3 view_dir = normalize(view_pos - fs_in.pos);
    vec3 reflect_dir = reflect(-light_dir, fs_in.normal);  
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(fs_in.normal, halfway_dir), 0.0), 32.0/(material_shininess));
    float specular = spec * material_specular;  

    // sum
    out_color =  (ambient + (1.0 - shadow) * (diffuse + specular)) *  texture(object_texture, fs_in.texcoord);
    //out_color = vec4(vec3(texture(depth_cube, vec3(fs_in.pos - light_pos)).r),1.0);
    //out_color = vec4(vec3(shadow),1.0);
}

