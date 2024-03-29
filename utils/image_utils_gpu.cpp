#include "image_utils_gpu.h" 

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

void computeSDF2d_device(float* depth, uchar* d_middle_mask, cv::Mat& sdf, int W, int H)
{
	cv::Mat mask(cv::Size(W/2, H/2), CV_8UC1);

	convertDepthToMaskHalfSize_device(depth, d_middle_mask, W, H);
	cudaMemcpy(mask.data, d_middle_mask, W/2*H/2 * sizeof(uchar), cudaMemcpyDeviceToHost);
	
	cv::Mat mask_inv = 1 - mask;
	cv::Mat dt_inner, dt_outer;

	cv::distanceTransform(mask, dt_inner, cv::DIST_L2, 5);
	cv::distanceTransform(mask_inv, dt_outer, cv::DIST_L2, 5);
	sdf = dt_inner - dt_outer;
	cv::resize(sdf, sdf, cv::Size(1920, 1080)); 
	sdf = sdf * 2; 
}

void overlay_render_on_raw_gpu(cv::Mat& render, cv::Mat &raw, cv::Mat& out)
{
	out = raw.clone(); 
	int H = out.rows;
	int W = out.cols; 

	uchar* d_render;
	cudaMalloc((void**)&d_render, W*H * sizeof(uchar) * 3);
	cudaMemcpy(d_render, render.data, W*H * sizeof(uchar) *3, cudaMemcpyHostToDevice); 
	
	uchar* d_raw;
	cudaMalloc((void**)&d_raw, W*H * sizeof(uchar)*3);
	cudaMemcpy(d_raw, raw.data, W*H * sizeof(uchar)*3, cudaMemcpyHostToDevice);

	uchar* d_out;
	cudaMalloc((void**)&d_out, W*H * sizeof(uchar)*3);
	cudaMemcpy(out.data, d_raw, W*H * sizeof(uchar) * 3, cudaMemcpyDeviceToHost);

	overlay_render_on_raw_device(d_render, d_raw, W, H, d_out); 
	cudaMemcpy(out.data, d_out, W*H * sizeof(uchar)*3, cudaMemcpyDeviceToHost); 


	cudaFree(d_render);
	cudaFree(d_raw); 
	cudaFree(d_out); 

}