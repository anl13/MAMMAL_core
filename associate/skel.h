#pragma once 

#include <vector> 
#include <Eigen/Eigen> 

typedef Eigen::Matrix<double, 4, 20, Eigen::ColMajor> PIG_SKEL; 
typedef Eigen::Matrix<double, 3, 20, Eigen::ColMajor> PIG_SKEL_2D; 

extern std::vector<std::string> LABEL_NAMES;
extern std::vector<Eigen::Vector2i> BONES; 
extern std::vector<std::vector<int> > AFF; 
extern std::vector<int> PA; 
extern std::vector<float> BONES_LEN; 