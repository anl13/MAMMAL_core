#include "shader.h"
#include <vector>
#include <Eigen/Eigen>

SimpleShader::SimpleShader() {}


SimpleShader::~SimpleShader() {}


SimpleShader::SimpleShader(const std::string& vertexPath, const std::string& fragmentPath)
	:SimpleShader(vertexPath, fragmentPath, "")
{}


SimpleShader::SimpleShader(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath)
{
	// 1. retrieve the vertex/fragment source code from filePath
	std::string vertexCode;
	std::string fragmentCode;
	std::string geometryCode;
	std::ifstream vShaderFile;
	std::ifstream fShaderFile;
	std::ifstream gShaderFile;
	// ensure ifstream objects can throw exceptions:
	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		// open files
		vShaderFile.open(vertexPath.c_str());
		fShaderFile.open(fragmentPath.c_str());
		std::stringstream vShaderStream, fShaderStream;
		// read file's buffer contents into streams
		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();
		// close file handlers
		vShaderFile.close();
		fShaderFile.close();
		// convert stream into string
		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
		// if geometry shader path is present, also load a geometry shader
		if (geometryPath != "")
		{
			gShaderFile.open(geometryPath.c_str());
			std::stringstream gShaderStream;
			gShaderStream << gShaderFile.rdbuf();
			gShaderFile.close();
			geometryCode = gShaderStream.str();
		}
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	}
	const char* vShaderCode = vertexCode.c_str();
	const char * fShaderCode = fragmentCode.c_str();
	// 2. compile shaders
	unsigned int vertex, fragment;
	// vertex shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	CheckCompileErrors(vertex, "VERTEX");
	// fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	CheckCompileErrors(fragment, "FRAGMENT");
	// if geometry shader is given, compile geometry shader
	unsigned int geometry;
	if (geometryPath != "")
	{
		const char * gShaderCode = geometryCode.c_str();
		geometry = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometry, 1, &gShaderCode, NULL);
		glCompileShader(geometry);
		CheckCompileErrors(geometry, "GEOMETRY");
	}
	// shader Program
	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	if (geometryPath != "")
		glAttachShader(ID, geometry);
	glLinkProgram(ID);
	CheckCompileErrors(ID, "PROGRAM");
	// delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertex);
	glDeleteShader(fragment);
	if (geometryPath != "")
		glDeleteShader(geometry);
}


void SimpleShader::CheckCompileErrors(GLuint shader, std::string type)
{
	GLint success;
	GLchar infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

void SimpleShader::configMultiLight()
{
	SetFloat("material.shininess", 1);
	SetFloat("material.diffuse", 0.6f);
	SetFloat("material.specular", 0.01f);
	SetFloat("material.ambient", 0.5);

	/*
	   Here we set all the uniforms for the 5/6 types of lights we have. We have to set them manually and index
	   the proper PointLight struct in the array to set each uniform variable. This can be done more code-friendly
	   by defining light types as classes and set their values in there, or by using a more efficient uniform approach
	   by using 'Uniform buffer objects', but that is something we'll discuss in the 'Advanced GLSL' tutorial.
	*/
	std::vector<Eigen::Vector3f> pointLightPositions = {
		Eigen::Vector3f(0,0,4),
		Eigen::Vector3f(3, 0, 1.2),
		Eigen::Vector3f(-3, 0, 1.2),
		Eigen::Vector3f(0,2, 1.2),
		Eigen::Vector3f(0, -2, 1.2),
		Eigen::Vector3f(0,0,-6)
	};
	// directional light
	SetVec3("dirLight.direction", 0, 0, 3.f);
	SetVec3("dirLight.ambient", 0.1f, 0.1f, 0.1f);
	SetVec3("dirLight.diffuse", 0.4f, 0.4f, 0.4f);
	SetVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);
	// point light 1
	SetVec3("pointLights[0].position", pointLightPositions[0]);
	SetVec3("pointLights[0].ambient", 0.05, 0.05, 0.05);
	SetVec3("pointLights[0].diffuse", 0.8f, 0.8f, 0.8f);
	SetVec3("pointLights[0].specular", 1.f, 1.f, 1.f);
	SetFloat("pointLights[0].constant", 1);
	SetFloat("pointLights[0].linear", 0.09);
	SetFloat("pointLights[0].quadratic", 0.032);
	// point light 2
	SetVec3("pointLights[1].position", pointLightPositions[1]);
	SetVec3("pointLights[1].ambient", 0.05f, 0.05f, 0.05f);
	SetVec3("pointLights[1].diffuse", 0.8f, 0.8f, 0.8f);
	SetVec3("pointLights[1].specular", 0.4f, 0.4f, 0.4f);
	SetFloat("pointLights[1].constant", 1.0f);
	SetFloat("pointLights[1].linear", 0.09);
	SetFloat("pointLights[1].quadratic", 0.032);
	// point light 3
	SetVec3("pointLights[2].position", pointLightPositions[2]);
	SetVec3("pointLights[2].ambient", 0.05f, 0.05f, 0.05f);
	SetVec3("pointLights[2].diffuse", 0.8f, 0.8f, 0.8f);
	SetVec3("pointLights[2].specular", 0.4f, 0.4f, 0.4f);
	SetFloat("pointLights[2].constant", 1.0f);
	SetFloat("pointLights[2].linear", 0.09);
	SetFloat("pointLights[2].quadratic", 0.032);
	// point light 4
	SetVec3("pointLights[3].position", pointLightPositions[3]);
	SetVec3("pointLights[3].ambient", 0.05f, 0.05f, 0.05f);
	SetVec3("pointLights[3].diffuse", 0.8f, 0.8f, 0.8f);
	SetVec3("pointLights[3].specular", 0.4f, 0.4f, 0.4f);
	SetFloat("pointLights[3].constant", 1.0f);
	SetFloat("pointLights[3].linear", 0.09);
	SetFloat("pointLights[3].quadratic", 0.032);
	// point light 5
	SetVec3("pointLights[4].position", pointLightPositions[4]);
	SetVec3("pointLights[4].ambient", 0.05f, 0.05f, 0.05f);
	SetVec3("pointLights[4].diffuse", 0.8f, 0.8f, 0.8f);
	SetVec3("pointLights[4].specular", 0.4f, 0.4f, 0.4f);
	SetFloat("pointLights[4].constant", 1.0f);
	SetFloat("pointLights[4].linear", 0.09);
	SetFloat("pointLights[4].quadratic", 0.032);
	// point light 6
	SetVec3("pointLights[5].position", pointLightPositions[5]);
	SetVec3("pointLights[5].ambient", 0.05f, 0.05f, 0.05f);
	SetVec3("pointLights[5].diffuse", 0.8f, 0.8f, 0.8f);
	SetVec3("pointLights[5].specular", 0.4f, 0.4f, 0.4f);
	SetFloat("pointLights[5].constant", 1.0f);
	SetFloat("pointLights[5].linear", 0.09);
	SetFloat("pointLights[5].quadratic", 0.032);

	// spotLight

	SetVec3("spotLight.ambient", 0.0f, 0.0f, 0.0f);
	SetVec3("spotLight.diffuse", 1.0f, 1.0f, 1.0f);
	SetVec3("spotLight.specular", 1.0f, 1.0f, 1.0f);
	SetFloat("spotLight.constant", 1.0f);
	SetFloat("spotLight.linear", 0.09);
	SetFloat("spotLight.quadratic", 0.032);
	SetFloat("spotLight.cutOff", cosf(12.5f / 180 * 3.1415926));
	SetFloat("spotLight.outerCutOff", cosf(15.f / 180 * 3.1415926));
}