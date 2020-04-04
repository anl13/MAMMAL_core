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
#include "math_utils.h"

class OBJReader
{
public:
	OBJReader(){};
	virtual ~OBJReader(){};

	void read(std::string filename);
    void write(std::string filename); 
	std::vector<Eigen::Vector3d> vertices;
	std::vector<int> tex_to_vert; 
	std::vector<Eigen::Vector3u> faces_v; // vertices of faces
	std::vector<Eigen::Vector3u> faces_t; // textures of faces
	std::vector<Eigen::Vector2d> textures;
    Eigen::MatrixXd vertices_eigen; 
    Eigen::Matrix<unsigned int,-1,-1,Eigen::ColMajor> faces_v_eigen; 
	Eigen::MatrixXd textures_eigen; 

private:
	void split_face_str(std::string str, int& i1, int &i2, int &i3);
	float calcTriangleArea(double u1, double v1,
		double u2, double v2,
		double u3, double v3);
};
#endif 