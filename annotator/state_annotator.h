#pragma once
#include <vector> 
#include <iostream> 
#include <fstream> 

#include <json/json.h> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>

#include "../utils/image_utils.h" 
#include "anno_utils.h"

struct single_record {
	single_record() {
	}
	int state;

};

class StateAnnotator
{
public: 
	StateAnnotator() {}
	std::vector<Eigen::Vector3i> m_CM; 
	cv::Mat m_panel_attr; 

	void show_panel(); 

private: 
	void construct_panel_attr();
};