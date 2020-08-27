#pragma once

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <boost/algorithm/string.hpp>
#include <cuda_runtime_api.h>
#include <vector_functions.hpp> 

#include "math_utils.h"



struct Mesh
{
public: 
	Mesh() {}
	Mesh(const std::string& filename) { Load(filename); }

	int vertex_num; 
	int texture_num; 
	int face_num; 

	std::vector<Eigen::Vector3f> vertices_vec; 
	std::vector<Eigen::Vector3f> normals_vec; 
	std::vector<Eigen::Vector3u> faces_v_vec; 
	std::vector<Eigen::Vector3u> faces_t_vec; 
	std::vector<Eigen::Vector2f> textures_vec; 
	
	void CalcNormal();
	void Load(const std::string& filename);
	void Save(const std::string& filename) const; 


private: 
	void SplitFaceStr(std::string &str, int &i1, int &i2, int &i3); 
};

class MeshEigen
{
public:
	MeshEigen() {}
	MeshEigen(const Mesh& _mesh);
	Eigen::Matrix3Xf vertices;
	Eigen::Matrix3Xf normals;
	Eigen::Matrix3Xu faces;
	Eigen::Matrix2Xf textures;

	void Deform(const Eigen::Vector3f &xyzScale);
};

class MeshFloat4
{
public:
	MeshFloat4() {}
	MeshFloat4(const Mesh& _mesh);
	std::vector<float4> vertices;
	std::vector<float4> normals;
	std::vector<unsigned int> indices;
};