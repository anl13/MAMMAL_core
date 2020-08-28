#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils_render.h"

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