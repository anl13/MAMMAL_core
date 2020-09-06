#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>

#include "../utils/safe_call.hpp"

#include <opencv2/core/mat.hpp>


void computeSDF2d_device(float* input, uchar* d_middle_mask, cv::Mat& sdf, int W, int H);

void convertDepthToMask_device(float* input, uchar* mask, int W, int H); 

// interface with opencv 
void overlay_render_on_raw_gpu(cv::Mat& render, cv::Mat &raw, cv::Mat& out);
void overlay_render_on_raw_device(uchar* render, uchar* raw, int W, int H, uchar* out); 