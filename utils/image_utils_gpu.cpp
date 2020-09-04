#include "image_utils_gpu.h" 

#include <opencv2/opencv.hpp>

void computeSDF2d_device(float* depth, cv::Mat& sdf, int W, int H)
{
	cv::Mat mask(cv::Size(W, H), CV_8UC1);
	uchar* d_mask;
	cudaMalloc((void**)&d_mask, H*W * sizeof(uchar));
	convertDepthToMask_device(depth, d_mask, W, H);
	cudaMemcpy(mask.data, d_mask, W*H * sizeof(uchar), cudaMemcpyDeviceToHost);
	cv::Mat mask_inv = 255 - mask;
	cv::Mat dt_inner, dt_outer;
	cv::distanceTransform(mask, dt_inner, cv::DIST_L2, 5);
	cv::distanceTransform(mask_inv, dt_outer, cv::DIST_L2, 5);
	sdf = dt_inner - dt_outer;
}