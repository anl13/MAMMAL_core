// This file is borrowed from https://github.com/1danielcoelho/distance-transform-cuda 
// CUDA implementation of Meijster's parallel algorithm for 
// calculating the distance transform 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <vector>

#include <cuda_runtime.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>
#include <device_launch_parameters.h>

#include "../utils/safe_call.hpp"

#include "distance_transform_cuda.h"

__global__ void edt_cols(uchar* d_input, uchar* d_output, unsigned int width, unsigned int height)
{
	// x in range [0, width-1]	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= width)
		return;

	extern __shared__ uchar g[]; // shared memory for the block. 

	// Initialize val to either 0 or 'infinity'
	uchar val = (1 - d_input[x]) * (width + height); // first row 
	g[0] = val;

	// Scan 1: Compute GT
	for (unsigned int y = 1; y < height; y++)
	{
		val = (1 - d_input[y * width + x]) * (1 + val);
		g[y] = val;
	}

	// Scan 2	
	// y < height is the same as y >= 0, as this unsigned int underflows
	for (unsigned int y = height - 2; y < height ; y--)
	{
		if (g[y] > val)
		{
			g[y] = 1 + val;
		}

		val = g[y];
	}

	for (unsigned int y = 0; y < height; y++)
		d_output[y * width + x] = g[y];
}

__global__ void edt_rows(uchar* d_output, float* d_final, unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // range [0, width-1]
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // range [0, height-1]

	if (x >= width)
		return;

	d_final[y*width + x] = d_output[y*width + x];
	extern __shared__ uchar d_localG[];

	for (unsigned int i = threadIdx.x; i < width; i += blockDim.x)
		d_localG[i] = d_output[y * width + i];

	__syncthreads();

	float minDist = FLT_MAX;
	for (unsigned int i = 0; i < width; i++)
	{
		float t = (float)d_localG[i];
		minDist = fminf(minDist, ((float)x - i)*((float)x - i) + t * t);
	}

	d_output[y * width + x] = sqrtf(minDist);
}

float get_min(const float& a, const float& b) {
	return a > b ? b : a;
}

void runCUDA(uchar* d_inData, uchar* d_middle,  float* d_outData, unsigned int width, unsigned int height)
{
	size_t numBytes = height * width * sizeof(float);
	assert(numBytes > 0);

	dim3 colblocksize(1); 
	dim3 colgridsize(pcl::device::divUp(width, 1)); 

	dim3 rowblocksize(int(get_min(1024,width)), 1); 
	dim3 rowgridsize(pcl::device::divUp(width, rowblocksize.x), pcl::device::divUp(height, rowblocksize.y)); 

	edt_cols << <colgridsize, colblocksize, height * sizeof(uchar)>> > (d_inData, d_middle, width, height);
	edt_rows << < rowgridsize, rowblocksize, width * sizeof(uchar) >> > (d_middle, d_outData, width, height); 

	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 
}
