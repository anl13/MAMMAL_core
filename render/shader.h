#pragma once

#include <glad/glad.h>
#include <Eigen/Core>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class SimpleShader
{
public:
	SimpleShader();
	~SimpleShader();
	SimpleShader(const std::string& vertexPath, const std::string& fragmentPath, const std::string& geometryPath);
	SimpleShader(const std::string& vertexPath, const std::string& fragmentPath);

	unsigned int return_ID() { return ID;  }
	void Use() const { glUseProgram(ID); }
	void SetBool(const std::string &name, bool value) { glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value); }
	void SetInt(const std::string &name, int value) { glUniform1i(glGetUniformLocation(ID, name.c_str()), value); }
	void SetFloat(const std::string &name, float value) { glUniform1f(glGetUniformLocation(ID, name.c_str()), value); }
	void SetVec2(const std::string &name, const Eigen::Vector2f& value) { glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, value.data()); }
	void SetVec2(const std::string &name, float x, float y) { glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y); }
	void SetVec3(const std::string &name, const Eigen::Vector3f& value) { glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, value.data()); }
	void SetVec3(const std::string &name, float x, float y, float z) { glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z); }
	void SetVec4(const std::string &name, const Eigen::Vector4f& value) { glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, value.data()); }
	void SetVec4(const std::string &name, float x, float y, float z, float w) { glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w); }
	void SetMat2(const std::string &name, const Eigen::Matrix2f& mat) { glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, mat.data()); }
	void SetMat3(const std::string &name, const Eigen::Matrix3f& mat) { glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, mat.data()); }
	void SetMat4(const std::string &name, const Eigen::Matrix4f& mat) { glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, mat.data()); }

	void configMultiLight(); 
private:
	unsigned int ID;
	void CheckCompileErrors(GLuint shader, std::string type);
};

