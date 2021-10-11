#pragma once 

#include <iostream> 
#include <fstream> 
#include <iomanip>
#include <Eigen/Eigen> 
#include <json/json.h> 
#include <vector> 
#include <opencv2/opencv.hpp> 
#include "../utils/math_utils.h"
#include "../utils/camera.h"
#include "../utils/image_utils.h"
#include "../utils/geometry.h" 
#include "../utils/skel.h" 

using std::vector;

// data of a frame, together with matching process
class Part1Data {
public:
	Part1Data() {}
	~Part1Data() {}

	// attributes 
	void set_frame_id(int _frameid) { m_frameid = _frameid; }
	int get_frame_id() { return m_frameid; }
	vector<Camera> get_cameras() { return m_camsUndist; }

	// io functions 
	void init(); 
	void readCameras();
	void read_labeling(); 
	void compute_3d_gt(); 
	cv::Mat visualizeProj(int id = -1);
	void reproject_skels();
	void read_imgs(); 
	cv::Mat visualize2D(int id = -1); 

	SkelTopology m_topo;

	int m_imh;
	int m_imw;
	int m_frameid;
	int m_camNum;
	std::vector<int>                          m_camids;
	std::vector<Eigen::Vector3i>              m_CM;
	vector<vector<vector<Eigen::Vector3f> > > m_projs; // [viewid, pigid, kptid]
	std::vector<Camera>                       m_cams; 
	std::vector<Camera>                       m_camsUndist;
	std::vector<cv::Mat>                      m_imgsUndist; 

	// io function and tmp data
	vector<vector<vector<Eigen::Vector3f> > > m_keypoints_undist; // camid, pigid, jointnum
	vector<vector<Eigen::Vector4f> >          m_boxes_processed; // camid, candid
	vector<vector<vector<vector<Eigen::Vector2f> > > > m_masks_undist; // camid, pigid, partnum, pointnum

	vector<vector<Eigen::Vector3f> > m_gt_keypoints_3d; // pigid, jointid; if no 3d, set as [0,0,0]
	
};

// [pigid, jointid]
vector<vector<Eigen::Vector3f>> load_skel(std::string folder, int frameid); 

// [pigid, jointid]
vector<vector<Eigen::Vector3f>> load_joint23(std::string folder, int frameid);

// input: est: [frameid, pigid, jointid]; gt: same to est
void eval_skel_3d(const vector<vector<vector<Eigen::Vector3f> > >& est,
	vector<vector<vector<Eigen::Vector3f>>>& gt); 

void process_generate_label3d(); 

void save_points(std::string folder, int pid, int fid, const std::vector<Eigen::Vector3f>& data); 
