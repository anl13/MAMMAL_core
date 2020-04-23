#pragma once
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include "../utils/math_utils.h"

struct Model
{
	Model() {}
	Model(const std::string& filename) { Load(filename); }
	Eigen::Matrix3Xd vertices;
	Eigen::Matrix3Xd normals;
	Eigen::Matrix3Xu faces;

	void CalcNormal();
	void Load(const std::string& filename);
	void Save(const std::string& filename) const; 
};

