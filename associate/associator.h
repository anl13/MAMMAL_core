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
#include "../utils/Hungarian.h"
#include "clusterclique.h"
#include "skel.h" 
#include "../bundle/annotator.h"

using std::vector; 

// This associator is based on tracklets
// per-view temporal tracking is top-priority
class Associator {
public: 
	Associator();
	~Associator(); 

	vector<int> m_camids; 
	int m_camNum; 
	int m_imw; 
	int m_imh;
	vector<Eigen::Vector3i> m_CM;
	


};