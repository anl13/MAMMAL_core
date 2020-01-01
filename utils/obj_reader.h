#ifndef OBJIO_H
#define OBJIO_H

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
//#include <unordered_map>

#include <string>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>

class OBJReader
{
public:
	OBJReader(){};
	virtual ~OBJReader(){};

	void read(std::string filename);
    void write(std::string filename); 
	void set_face_buffer(std::vector<Eigen::Vector3i> &face_buffer);
    // variales 
	std::vector<Eigen::Vector3f> vertices;
	std::vector<Eigen::Vector3i> faces_v; // vertices of faces
	std::vector<Eigen::Vector3i> faces_t; // textures of faces
	std::vector<Eigen::Vector2f> textures;
    Eigen::MatrixXf vertices_eigen; 
    Eigen::MatrixXi faces_v_eigen; 

private:
	void split_face_str(std::string str, int& i1, int &i2, int &i3);
	// core function to get right face buffer
	float calcTriangleArea(float u1, float v1,
		float u2, float v2,
		float u3, float v3);
	//std::unordered_map<int, std::vector<int>> f_map;
	//std::unordered_map<int, int> v_map;
};
#endif 