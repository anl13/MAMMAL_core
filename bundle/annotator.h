#pragma once

#include <string> 
#include <Eigen/Eigen> 
#include "../utils/image_utils.h"
#include "../utils/math_utils.h"
#include "../associate/skel.h"
#include <json/json.h>
#include "../utils/camera.h"
#include <sstream>

using std::vector;

enum AnnoState {
	JOINT_LABEL =0, // default
	ID_LABEL,
	VISIBILITY,
	MOTION
};

struct SingleClickLabeledData {
	SingleClickLabeledData() {
		x = 0; y = 0;
		camid = -1;
		ready = false; 
		cancel = false;
		single_image = false;
	}
	double x;
	double y; 
	int camid;
	bool cancel;
	bool ready;
	bool single_image;
};

class Annotator
{
public:
	Annotator() {
		construct_panel_attr(); 
	}
	~Annotator() {}
	// panel 
	cv::Mat m_panel_attr;
	cv::Mat m_image_labeled;
	cv::Mat m_proj;
	SkelTopology m_topo; 
	std::vector<Eigen::Vector3i> m_CM; 
	std::string result_folder;
	int frameid;
	
	void setInitData(const vector<MatchedInstance>& matched);
	void getMatchedData(vector<MatchedInstance>& matched);
	int m_camNum;
	std::vector<Camera> m_cams; 
	std::vector<cv::Mat> m_imgs; //[camnum]
	vector<vector<DetInstance> >              m_data; // [pigid,viewid]
	vector<vector<DetInstance> >              m_unmatched; // [camnum, candnum]

	void show_panel(); 
	void save_label_result(std::string jsonname);
	void read_label_result(std::string filename);
	void read_label_result();
	void save_label_result();

	void drawSkel(cv::Mat& img, const vector<Eigen::Vector3d>& _skel2d);
	void update_image_labeled(const SingleClickLabeledData& input, const std::vector<int>& status);
	void update_data(const SingleClickLabeledData& input, const std::vector<int>& status);

private:
	void construct_panel_attr();
	void update_panel(const std::vector<int>& status);
};
