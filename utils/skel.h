#pragma once 

#include <vector> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp> 
#include "image_utils.h" 

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
    vector<DetInstance> dets; 
}; 

vector<Eigen::Vector3f> convertMatToVec(const Eigen::MatrixXf& skel); 
Eigen::VectorXf convertStdVecToEigenVec(const std::vector<Eigen::Vector3f>& joints); 

void drawSkelDebug(cv::Mat& img, const vector<Eigen::Vector3f>& _skel2d,
	SkelTopology m_topo); 

void drawSkelMonoColor(cv::Mat& img, const vector<Eigen::Vector3f>& _skel2d, int colorid, SkelTopology m_topo); 