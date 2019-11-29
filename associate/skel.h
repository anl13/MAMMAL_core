#pragma once 

#include <vector> 
#include <Eigen/Eigen> 

class SkelTopology
{
public: 
    int joint_num; 
    int bone_num; 
    std::vector<std::string> label_names; 
    std::vector<Eigen::Vector2i> bones; 
    std::vector<int> kpt_color_ids; 
    std::vector<double> kpt_conf_thresh; 
};
SkelTopology getSkelTopoByType(std::string type); 
typedef Eigen::MatrixXd PIG_SKEL; 
typedef Eigen::MatrixXd PIG_SKEL_2D; 