#pragma once 

#include <vector> 
#include <Eigen/Eigen> 

using std::vector; 

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


struct DetInstance // class storing data of an instance 
{
    Eigen::Vector4d box; 
    std::vector<Eigen::Vector3d> keypoints; 
    std::vector<std::vector<Eigen::Vector2d> > mask; // as contours
};

struct MatchedInstance{
    vector<int> view_ids; 
    vector<int> cand_ids; 
    vector<DetInstance> dets; 
}; 