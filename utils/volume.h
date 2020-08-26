#pragma once
#include "math_utils.h"
#include "image_utils.h"

/*
usage: 
Eigen::MatrixXd joints = models[pid]->getZ();
std::cout << joints.transpose() << std::endl;
V.center = joints.col(20).cast<float>();
V.computeVolumeFromRoi(m_rois);
std::cout << "compute volume now. " << std::endl;
V.getSurface();
V.saveXYZFileWithNormal(ss.str());
std::stringstream cmd;
cmd << "D:/Projects/animal_calib/PoissonRecon.x64.exe --in " << ss.str() << " --out " << ss.str() << ".ply";
const std::string cmd_str = cmd.str();
const char* cmd_cstr = cmd_str.c_str();
system(cmd_cstr);
*/
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
	void get3DBox(std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector2u>& edges); // return a cube
	void saveXYZFileWithNormal(std::string filename);
	void readXYZFileWithNormal(std::string filename); 

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