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

