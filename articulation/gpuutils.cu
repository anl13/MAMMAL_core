#include "gpuutils.h"

#include <stdlib.h>
#include <time.h> 
#include <stdio.h> 

#include <device_launch_parameters.h>

void cuda_set_device(int n)
{
	cudaError_t status = cudaSetDevice(n); 
	check_error(status); 
}

int cuda_get_device()
{
	int n = 0; 
	cudaError_t status = cudaGetDevice(&n); 
	check_error(status); 
	return n;
}

void check_error(cudaError_t status)
{
	if (status != cudaSuccess)
	{
		const char *s = cudaGetErrorString(status); 
		char buffer[256]; 
		printf("CUDA error: %s\n", s); 
		snprintf(buffer, 256, "CUDA error: %s", s); 
	}
}

__global__ void set_constant2d_kernel(
	pcl::gpu::PtrStepSz<float> data,
	const int W,
	const int H,
	const float value
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y; 
	if (xIdx < W && yIdx < H)
	{
		data(yIdx, xIdx) = value; 
	}
}

void setConstant2D_device(pcl::gpu::DeviceArray2D<float> data, const int W, const int H, const float value)
{
	dim3 blocksize(32, 32); 
	dim3 gridsize(pcl::device::divUp(W, 32), pcl::device::divUp(H, 32)); 
	set_constant2d_kernel << <gridsize, blocksize >> > (data, W, H, value); 

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}