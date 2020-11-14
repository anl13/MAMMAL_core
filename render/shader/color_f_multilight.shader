#version 330 core
out vec4 FragColor;

struct Material {
    float ambient; 
    float diffuse;
    float specular;
    float shininess;
}; 

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight {
    vec3 position;
    
    float constant;
    float linear;
    float quadratic;
	
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
  
    float constant;
    float linear;
    float quadratic;
  
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;       
};

#define NR_POINT_LIGHTS 6

in VS_OUT
{
    vec3 pos;
    vec3 normal;
} fs_in;

uniform vec3 object_color;
uniform vec3 view_pos;
uniform DirLight dirLight;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform SpotLight spotLight;
uniform Material material;

// function prototypes
// vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir);
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
// vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

void main()
{    
    // properties
    vec3 norm = normalize(fs_in.normal);
    vec3 viewDir = normalize(view_pos - fs_in.pos);
    
    // == =====================================================
    // Our lighting is set up in 3 phases: directional, point lights and an optional flashlight
    // For each phase, a calculate function is defined that calculates the corresponding color
    // per lamp. In the main() function we take all the calculated colors and sum them up for
    // this fragment's final color.
    vec3 result = vec3(0,0,0);
    // == =====================================================
    // phase 1: directional lighting
    // result += CalcDirLight(dirLight, norm, viewDir);
    // // phase 2: point lights
    for(int i = 0; i < NR_POINT_LIGHTS; i++)
        result += CalcPointLight(pointLights[i], norm, fs_in.pos, viewDir);    
    // // phase 3: spot light
    // result += CalcSpotLight(spotLight, norm, fs_in.pos, viewDir);    

    FragColor = vec4(result, 1.0);
}

// // calculates the color when using a directional light.
// vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir)
// {
//     vec3 lightDir = normalize(-light.direction);
//     // diffuse shading
//     float diff = max(dot(normal, lightDir), 0.0);
//     // specular shading
//     vec3 reflectDir = reflect(-lightDir, normal);
//     float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
//     // combine results
//     // vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords));
//     // vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, TexCoords));
//     // vec3 specular = light.specular * spec * vec3(texture(material.specular, TexCoords));
//     vec3 color = vec3(texture(object_texture, fs_in.texcoord));
//     // vec3 ambient = light.ambient * color;
//     // vec3 ambient = material.ambient * color;
//     // vec3 diffuse = light.diffuse * diff * vec3(material.diffuse, material.diffuse,material.diffuse);
//     // vec3 specular = light.specular * spec * vec3(material.specular, material.specular, material.specular);
//     // return (ambient + diffuse + specular);
//     return color; 
// }

// calculates the color when using a point light.
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fragPos);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    // attenuation
    float distance = dot((light.position - fragPos),lightDir);
    float total_dist = (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    
    if(total_dist > 20) total_dist = 20;
    if(total_dist < 1) total_dist = 1; 
    float attenuation = 
     1.0 / total_dist;    
    
    // combine results
    // vec3 color = vec3(texture(object_texture, fs_in.texcoord));
    vec3 color = object_color;
    vec3 ambient = light.ambient * material.ambient * color;
    // vec3 diffuse = light.diffuse * diff * vec3(material.diffuse, material.diffuse, material.diffuse);
    // vec3 specular = light.specular * spec * vec3(material.specular, material.specular, material.specular);
    vec3 diffuse = light.diffuse * material.diffuse * diff * color; 
    vec3 specular = light.specular * material.specular * spec * color; 
    // if(attenuation > 50) attenuation = 1;
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    return (ambient + diffuse + specular);
}

// // calculates the color when using a spot light.
// vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
// {
//     vec3 lightDir = normalize(light.position - fragPos);
//     // diffuse shading
//     float diff = max(dot(normal, lightDir), 0.0);
//     // specular shading
//     vec3 reflectDir = reflect(-lightDir, normal);
//     float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
//     // attenuation
//     float distance = length(light.position - fragPos);
//     float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));    
//     // spotlight intensity
//     float theta = dot(lightDir, normalize(-light.direction)); 
//     float epsilon = light.cutOff - light.outerCutOff;
//     float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
//     // combine results
//     vec3 ambient = light.ambient * vec3(texture(object_texture, fs_in.texcoord));
//     vec3 diffuse = light.diffuse * diff * vec3(material.diffuse, material.diffuse,material.diffuse);
//     vec3 specular = light.specular * spec * vec3(material.specular, material.specular, material.specular);
   
//     ambient *= attenuation * intensity;
//     diffuse *= attenuation * intensity;
//     specular *= attenuation * intensity;
//     return (ambient + diffuse + specular);
// }