#pragma once

#include <vector> 
#include <Eigen/Eigen>
#include "../utils/camera.h"
#include <opencv2/opencv.hpp>

std::vector<Camera> readCameras(); 

int render_mean_pose(); 
