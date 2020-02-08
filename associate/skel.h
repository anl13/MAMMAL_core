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
vector<std::pair<int, int>> getPigMapper(); 
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

struct BodyState {
	// Level5: social state 

	// Level4: discrete pose state label or continuous state vector
	Eigen::VectorXd pose;

	// Level3: parametric state
	Eigen::Vector3d trans;
	double scale; 
	double frameid;
	int id; 
	
	// Level2: body orientation
	vector<Eigen::Vector3d> points; 

	// Level1: center position 
	Eigen::Vector3d center; 
	
	void saveState(std::string filename); 
	void loadState(std::string filename); 
};
