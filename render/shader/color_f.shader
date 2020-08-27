#version 330 core

in VS_OUT
{
    vec3 pos;
    vec3 normal;
} fs_in;

out vec4 out_color;

uniform samplerCube depth_cube;  
uniform sampler2D object_texture;      
uniform vec3 object_color;
uniform vec3 light_pos;
uniform float far_plane;
uniform vec3 view_pos;


void main()
{            
    // calculate shadow
    float shadow = 0.0;
    float material_ambient = 0.5;
    float material_diffuse = 0.6;
    float material_specular = 0.01;
    float material_shininess = 1;
    // vec3 light_distance = fs_in.pos - light_pos;
    // float current_depth = (length(light_distance))/far_plane;      
    // int samples = 20;
    // float view_distance = length(view_pos - fs_in.pos);
    // float disk_radius = (1.0+view_distance / far_plane) * 0.01;
    // float bias = disk_radius;
    // for(int i = 0; i < samples; ++i)
    // {
    //     if(current_depth > texture(depth_cube, light_distance + grid_sampling_disk[i] * disk_radius).r + bias)
    //     {
    //         shadow += 1.0;
    //     }
    // }
    // shadow /= float(samples);

    // ambient
    float ambient = material_ambient;
  	
    // diffuse 
    vec3 light_dir = normalize(light_pos - fs_in.pos);
    float diff = max(dot(fs_in.normal, light_dir), 0.0);
    float diffuse = diff * material_diffuse;  
    
    // specular
    vec3 view_dir = normalize(view_pos - fs_in.pos);
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(fs_in.normal, halfway_dir), 0.0), 32.0/(material_shininess));
    float specular = spec * material_specular;  

    
    out_color =  (ambient + (1.0 - shadow) * (diffuse + specular)) * vec4(object_color,1.0);

    /// render normal 
    // out_color = vec4(fs_in.normal, 1.0); 

    // out_color = vec4(vec3(texture(depth_cube, vec3(fs_in.pos - light_pos)).r),1.0);
    //out_color = vec4(vec3(shadow),1.0);
}
