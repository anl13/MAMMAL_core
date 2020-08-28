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
    std::vector<float> kpt_conf_thresh; 
};
SkelTopology getSkelTopoByType(std::string type); 

struct DetInstance // class storing data of an instance 
{
	DetInstance() {
		valid = false;
	}
	bool valid;
    Eigen::Vector4f box; // x1,y1,x2,y2
    std::vector<Eigen::Vector3f> keypoints; 
    std::vector<std::vector<Eigen::Vector2f> > mask; // as contours
	std::vector<std::vector<Eigen::Vector2f> > mask_norm; // normal of points
};

struct MatchedInstance{
    vector<int> view_ids; 
    //vector<int> cand_ids; 
    vector<DetInstance> dets; 
}; 

//struct BodyState {
//	// Level5: social state 
//
//	// Level4: discrete pose state label or continuous state vector
//	Eigen::VectorXd pose;
//
//	// Level3: parametric state
//	Eigen::Vector3d trans;
//	double scale; 
//	double frameid;
//	int id; 
//	
//	// Level2: body orientation
//	vector<Eigen::Vector3d> points; 
//
//	// Level1: center position 
//	Eigen::Vector3d center; 
//	
//	void saveState(std::string filename); 
//	void loadState(std::string filename); 
//};

vector<Eigen::Vector3f> convertMatToVec(const Eigen::MatrixXf& skel); 