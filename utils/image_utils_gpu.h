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
void convertDepthToMaskHalfSize_device(float* depth, uchar* mask, int W, int H);


// interface with opencv 
void overlay_render_on_raw_gpu(cv::Mat& render, cv::Mat &raw, cv::Mat& out);
void overlay_render_on_raw_device(uchar* render, uchar* raw, int W, int H, uchar* out); 


void distanceTransform_device(uchar* d_inData, float* d_outData, unsigned int width, unsigned int height);
void sdf2d_device(uchar* d_inData, float* d_outData, unsigned int width, unsigned int height); 

int get_min(int, int); 

void sobel_device(float* d_in, float* d_out_x, float* d_out_y, unsigned int W, unsigned int H); 