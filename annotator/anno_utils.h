#pragma once

#include <string> 
#include <opencv2/opencv.hpp>
#include <json/json.h>

void drawLabel(cv::Mat& img, int x, int y, int w, int h, std::string name, bool clicked);

class AnnoConfig
{
public: 
	AnnoConfig();
	std::string posesolver_config; 
	std::string pig_config; 
	int current_frame_id;
	int current_pig_id; 
};