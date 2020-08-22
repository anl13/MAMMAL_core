#pragma once

#include <opencv2/opencv.hpp>
#include <string> 

void drawLabel(cv::Mat& img, int x, int y, int w, int h, std::string name, bool clicked);