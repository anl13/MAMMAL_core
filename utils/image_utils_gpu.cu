#include "image_utils_gpu.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void convertDepthToMask_kernel(
	float* input, uchar* d_mask, int W, int H
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < W && y < H)
	{
		unsigned int index = y * W + x;
		float d = input[index];
		uchar value = 0;
		if (d > 0.00001) value = 255;
		d_mask[index] = value;
	}
}

void convertDepthToMask_device(float* input, uchar* mask, int W, int H)
{
	dim3 blocksize(32, 32);
	dim3 gridsize(pcl::device::divUp(W, blocksize.x),
		pcl::device::divUp(H, blocksize.y));

	convertDepthToMask_kernel << <gridsize, blocksize >> > (
		input, mask, W, H);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void convertDepthToMaskHalfSize_kernel(
	float* input, uchar* d_mask, int W, int H
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < W && y < H)
	{
		unsigned int index = 4 *y * W + 2*x;
		unsigned int target_index = y * W + x; 
		float d = input[index];
		float d2 = input[index + 1];
		uchar value = 0;
		if (d > 0.0001 || d2 > 0.0001) value = 255;
		d_mask[target_index] = value;
	}
}

void convertDepthToMaskHalfSize_device(float* depth, uchar* mask, int W, int H)
{
	int W_half = W / 2; 
	int H_half = H / 2; 
	dim3 blocksize(32, 32);
	dim3 gridsize(pcl::device::divUp(W_half, blocksize.x),
		pcl::device::divUp(H_half, blocksize.y));

	convertDepthToMaskHalfSize_kernel << <gridsize, blocksize >> > (
		depth, mask, W_half, H_half);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void overlay_render_on_raw_kernel(
	uchar* d_render,
	uchar* d_raw,
	uchar* out,
	int W, int H
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
	if (x < W && y < H)
	{
		int index0 = 3 *y * W + 3*x;
		int index1 = index0 + 1;
		int index2 = index1 + 1; 
		uchar r0 = d_render[index0];
		uchar r1 = d_render[index1];
		uchar r2 = d_render[index2];

		if (r0 > 0 || r1 > 0 || r2 > 0)
		{
			out[index0] = r0;
			out[index1] = r1;
			out[index2] = r2;
		}
		else
		{
			out[index0] = d_raw[index0];
			out[index1] = d_raw[index1]; 
			out[index2] = d_raw[index2];
		}
	}
}
void overlay_render_on_raw_device(
	uchar* d_render, uchar* d_raw, int W, int H, uchar* d_out
)
{
	dim3 blocksize(32, 32); 
	dim3 gridsize(pcl::device::divUp(W, blocksize.x), pcl::device::divUp(H, blocksize.y)); 
	overlay_render_on_raw_kernel << <gridsize, blocksize >> > (
		d_render, d_raw, d_out, W, H); 
	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 

}
