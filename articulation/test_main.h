#pragma once

#include <vector> 
#include <Eigen/Eigen>
#include "../utils/camera.h"
#include <opencv2/opencv.hpp>

std::vector<Camera> readCameras(); 

std::vector<cv::Mat> readImgs(); 

int render_mean_pose(); 

int visualize_artist_design(); 