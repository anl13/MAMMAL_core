#pragma once
#include "math_utils.h"
#include "image_utils.h"

struct Volume
{
	Volume(); 
	~Volume();
	void initVolume();
	void computeVolumeFromRoi(
		std::vector<ROIdescripter>& det
	);
	void getSurface();
	Eigen::Vector3f computeNormal(int x,int y,int z);
	void get3DBox(std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector2i>& edges); // return a cube
	void saveXYZFileWithNormal(std::string filename);

	int resX, resY, resZ; // default 128
	Eigen::Vector3f center; 
	float dx, dy, dz;
	float* data; 
	bool* surface; 
	int xyz2index(const int& x, const int& y, const int& z);
	Eigen::Vector3f index2point(const int& x, const int& y, const int& z);
	std::vector<Eigen::Vector3f> point_cloud;
	std::vector<Eigen::Vector3f> normals; 
	Eigen::MatrixXf point_cloud_eigen;

	

};