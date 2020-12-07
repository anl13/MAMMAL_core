#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils_render.h"
#include <opencv2/core/cuda.hpp>

__global__ void extract_depth_channel_kernel(
	const float4* position_device, const int W, const int H,
	float* depth_device
)
{
	int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int yIdx = blockIdx.y * blockDim.y + threadIdx.y; 

	if (xIdx < W && yIdx < H)
	{
		int index = yIdx * W + xIdx; 
		int flip_index = (H - 1 - yIdx) * W + xIdx; 
		float4 data = position_device[index];
		depth_device[flip_index] = -data.z; 
	}
}


void extract_depth_channel(const float4* position_device, const int W, const int H, float* depth_device)
{
	dim3 blocksize(32, 32); 
	dim3 gridsize(pcl::device::divUp(W, 32), pcl::device::divUp(H, 32)); 

	extract_depth_channel_kernel << <gridsize, blocksize >> > (
		position_device, W, H, depth_device); 
}

// ==========extract color image=============

__global__ void extract_color_kernel(
	const float4* rawdata, const int W, const int H,
	cv::cuda::PtrStep<uchar3> output
)
{
	int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	int yIdx = blockIdx.y * blockDim.y + threadIdx.y; 
	if (xIdx < W && yIdx < H)
	{
		int index = yIdx * W + xIdx; 
		int flip_index = (H - 1 - yIdx) * W + xIdx; 
		float4 data = rawdata[flip_index];
		uchar3 outdata;
		outdata.x = uchar( (data.z > 1? 1:data.z) * 255);
		outdata.y = uchar((data.y > 1 ? 1 : data.y) * 255);
		outdata.z = uchar((data.x > 1 ? 1 : data.x) * 255);

		output.ptr(yIdx)[xIdx] = outdata; 
	}
}

cv::Mat extract_bgr_mat(const float4* imgdata, const int W, const int H)
{
	cv::Mat img; 
	img.create(cv::Size(W, H), CV_8UC3); 

	cv::cuda::GpuMat d_mat; 
	d_mat.create(cv::Size(W, H), CV_8UC3); 
	
	dim3 blocksize(32, 32);
	dim3 gridsize(pcl::device::divUp(W, 32), pcl::device::divUp(H, 32));

	extract_color_kernel << < gridsize, blocksize >> > (
		imgdata, W, H, d_mat
		);
	d_mat.download(img); 
	return img; 
}