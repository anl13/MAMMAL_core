#pragma once 

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>
#include <opencv2/opencv.hpp> 

#include "../utils/safe_call.hpp"

void gpupseudo_color(float4* imgdata, int W, int H, float maxv, float minv, cv::Mat & imgout, 
	cv::Mat &depthout); 