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
    // calculate shadow
    float shadow = 0.0;

    vec3 light_distance = fs_in.pos - light_pos;
    float current_depth = (length(light_distance))/far_plane;      


    int samples = 20;
    float view_distance = length(view_pos - fs_in.pos);
    float disk_radius = (1.0+view_distance / far_plane) * 0.01;
    float bias = disk_radius;

    for(int i = 0; i < samples; ++i)
    {
        if(current_depth > texture(depth_cube, light_distance + grid_sampling_disk[i] * disk_radius).r + bias)
        {
            shadow += 1.0;
        }
    }
    shadow /= float(samples);

    // ambient
    float ambient = 0.5 * material_ambient;
  	
    // diffuse 
    vec3 light_dir = normalize(light_pos - fs_in.pos);
    float diff = max(dot(fs_in.normal, light_dir), 0.0);
    float diffuse = 0.5 * diff * material_diffuse;  
    
    // specular
    vec3 view_dir = normalize(view_pos - fs_in.pos);
    vec3 reflect_dir = reflect(-light_dir, fs_in.normal);  
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(fs_in.normal, halfway_dir), 0.0), 100.0/(material_shininess));
    float specular = 0.01 * spec * material_specular;  

    // sum
    out_color =  (ambient + (1.0 - shadow) * (diffuse + specular)) *  texture(object_texture, fs_in.texcoord);
    //out_color = vec4(vec3(texture(depth_cube, vec3(fs_in.pos - light_pos)).r),1.0);
    //out_color = vec4(vec3(shadow),1.0);
}

