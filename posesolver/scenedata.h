#pragma once

#include <vector>
#include <iostream> 
#include <fstream> 
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "../utils/skel.h"

class SceneData {
public: 
	SceneData();


	std::string m_camDir; 

	int m_camNum; 
	std::vector<int> m_camids; 

	cv::Mat m_undist_mask; 
	std::vector<cv::Mat> m_scene_masks; 
	std::vector<cv::Mat> m_backgrounds;
	cv::Mat m_undist_mask_chamfer; 
	std::vector<cv::Mat> m_scene_mask_chamfer;

private:
	void readBackground();
	void readSceneMask();
};
