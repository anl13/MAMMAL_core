#version 330 core

in VS_OUT
{
    vec3 pos;
    vec3 normal; 
    vec3 color; 
} fs_in;

out vec4 out_color;

uniform samplerCube depth_cube;       
uniform sampler2D object_texture;

uniform vec3 light_pos;
uniform float far_plane;
uniform vec3 view_pos;

void main()
{        

    float shadow = 0.0;
    // ambient
    float material_ambient = 0.5;
    float material_diffuse = 0.6;
    float material_specular = 0.01;
    float material_shininess = 1;
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

    out_color =  (ambient + (1.0 - shadow) * (diffuse + specular)) * vec4(fs_in.color,1.0);
    // out_color = vec4(1,1,1,1); 
}

