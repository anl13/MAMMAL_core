#include "gpuutils.h"

#include <stdlib.h>
#include <time.h> 
#include <stdio.h> 

#include <device_launch_parameters.h>

// ========
// helper function 
// ========

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

// =======
// kernel function
// =======

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

//            dim: y
// | * --- = | | |
// |         | | |  dim: z
// |         | | |
// compute upper triangle y>=z 
__global__ void computeAAT_kernel(
	const pcl::gpu::PtrStepSz<float> A, // [y, x]
	const int W, const int H,
	pcl::gpu::PtrStepSz<float> AAT
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int zIdx = blockIdx.z * blockDim.z + threadIdx.z;
	if (xIdx < W && yIdx < H && zIdx <= yIdx)
	{
		float a_zy = A(zIdx, xIdx) *A(yIdx, xIdx);
		atomicAdd(&AAT(zIdx, yIdx), a_zy);
	}
}

__global__ void computeAAT_kernel2(
	const pcl::gpu::PtrStepSz<float> A,
	const int W, const int H,
	pcl::gpu::PtrStepSz<float> AAT
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (xIdx < W)
	{
#pragma unroll
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < H; j++)
			{
				atomicAdd(&AAT(i, j), A(i, xIdx) * A(j, xIdx));
			}
		}
	}
}

__global__ void computeA_plus_AT_kernel(
	pcl::gpu::PtrStepSz<float> A,
	const int H
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y; 
	if (xIdx < H && yIdx < H && yIdx > xIdx)
	{
		A(yIdx, xIdx) = A(xIdx, yIdx); 
	}
}


// ======
// interface function 
// ======

void setConstant2D_device(pcl::gpu::DeviceArray2D<float> &data, const int W, const int H, const float value)
{
	dim3 blocksize(32, 32); 
	dim3 gridsize(pcl::device::divUp(W, 32), pcl::device::divUp(H, 32)); 
	set_constant2d_kernel << <gridsize, blocksize >> > (data, W, H, value); 

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

void computeATA_device(const pcl::gpu::DeviceArray2D<float> &A, const int W, const int H, pcl::gpu::DeviceArray2D<float> &ATA)
{
	// infact, ATA here is AAT, [H, H]
	if (ATA.empty())
	{
		ATA.create(H, H);
	}

	setConstant2D_device(ATA, H, H, 0); 

	dim3 blocksize2(2, 16, 32);
	dim3 gridsize2(pcl::device::divUp(W, blocksize2.x),
		pcl::device::divUp(H, blocksize2.y),
		pcl::device::divUp(H, blocksize2.z)); 
	computeAAT_kernel << <gridsize2, blocksize2 >> > (
		A, W, H, ATA
		);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	dim3 blocksize3(32, 32); 
	dim3 gridsize3(pcl::device::divUp(H, blocksize3.x), 
		pcl::device::divUp(H, blocksize3.y)); 
	computeA_plus_AT_kernel << <gridsize3, blocksize3 >> > (
		ATA, H
		);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}