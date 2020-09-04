#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>

#include "../utils/safe_call.hpp"

#include <opencv2/core/mat.hpp>


void computeSDF2d_device(float* input, cv::Mat& sdf, int W, int H);

void convertDepthToMask_device(float* input, uchar* mask, int W, int H); 

// interface with opencv 
