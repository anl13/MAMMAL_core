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

/*
Distance transform 
*/

__global__ void edt_cols_kernel(uchar* d_input, float* d_output, unsigned int width, unsigned int height)
{
	// x in range [0, width-1]	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= width)
		return;

	extern __shared__ float g[];

	// Initialize val to either 0 or 'infinity'
	float val = (1 - float(d_input[x])) * (width + height);
	g[0] = val;

	// Scan 1
	for (unsigned int y = 1; y < height; y++)
	{
		val = (1 - float(d_input[y * width + x])) * (1 + val);
		g[y] = val;
	}

	// Scan 2	
	// y < height is the same as y >= 0, as this unsigned int underflows
	for (unsigned int y = height - 2; y < height; y--)
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

__global__ void edt_rows_kernel(float* d_output, unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // range [0, width-1]
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // range [0, height-1]

	if (x >= width)
		return;

	extern __shared__ float d_localG[];

	for (unsigned int i = threadIdx.x; i < width; i += blockDim.x)
		d_localG[i] = d_output[y * width + i];

	__syncthreads();

	float minDist = FLT_MAX;
	for (unsigned int i = 0; i < width; i++)
	{
		minDist = fminf(minDist, (x - i)*(x - i) + d_localG[i] * d_localG[i]);
	}

	d_output[y * width + x] = sqrtf(minDist);
}

int get_min(int a, int b) { return a < b ? a : b; }

void distanceTransform_device(uchar* d_inData, float* d_outData, unsigned int width, unsigned int height)
{

	size_t numBytes = height * width * sizeof(float);
	assert(numBytes > 0);

	dim3 colblocksize(1);
	dim3 colgridsize(pcl::device::divUp(width, 1));

	dim3 rowblocksize(int(get_min(1024, width)), 1);
	dim3 rowgridsize(pcl::device::divUp(width, rowblocksize.x), pcl::device::divUp(height, rowblocksize.y));

	edt_cols_kernel << <colgridsize, colblocksize, height * sizeof(float) >> > (d_inData, d_outData, width, height);
	edt_rows_kernel << < rowgridsize, rowblocksize, width * sizeof(float) >> > (d_outData, width, height);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

/*
SDF 2D Device 
*/

__global__ void edt_cols2_kernel(uchar* d_input, float* d_output, unsigned int width, unsigned int height)
{
	// x in range [0, width-1]	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= width)
		return;

	extern __shared__ float g[];

	// Initialize val to either 0 or 'infinity'
	float val = (1-float(d_input[x])) * (width + height);
	float val2 = (float(d_input[x])) * (width + height);
	
	g[0] = val;
	g[1] = val2; 

	// Scan 1
	for (unsigned int y = 1; y < height; y++)
	{
		val = (1-float(d_input[y * width + x])) * (1 + val);
		val2 = (float(d_input[y * width + x])) * (1 + val2);

		g[2*y] = val;
		g[2 * y + 1] = val2; 
	}

	// Scan 2	
	// y < height is the same as y >= 0, as this unsigned int underflows
	for (unsigned int y = height - 2; y < height; y--)
	{
		if (g[2*y] > val)
		{
			g[2*y] = 1 + val;
		}
		val = g[2*y];

		if (g[2 * y + 1] > val2)
		{
			g[2 * y + 1] = 1 + val2; 
		}
		val2 = g[2 * y + 1];
	}

	for (unsigned int y = 0; y < height; y++)
	{
		if(g[2*y]>0)
			d_output[y * width + x] = g[2*y];
		else d_output[y*width + x] = -g[2 * y + 1];
	}
}

__global__ void edt_rows2_kernel(uchar* mask, float* d_output, unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // range [0, width-1]
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // range [0, height-1]

	if (x >= width)
		return;
	uchar m = mask[y*width + x];
	extern __shared__ float d_localG[];

	for (unsigned int i = threadIdx.x; i < width; i += blockDim.x)
		d_localG[i] = d_output[y * width + i];

	__syncthreads();

	if (m == 0)
	{
		float minDist = FLT_MAX;
		for (unsigned int i = 0; i < width; i++)
		{
			float t = d_localG[i];
			t = t > 0 ? t : 0;
			minDist = fminf(minDist, (x - i)*(x - i) + t * t);
		}

		d_output[y * width + x] = sqrtf(minDist);
	}
	else
	{
		float minDist = FLT_MAX;
		for (unsigned int i = 0; i < width; i++)
		{
			float t = d_localG[i];
			t = t < 0 ? t : 0;
			minDist = fminf(minDist, (x - i)*(x - i) + t * t);
		}
		d_output[y*width + x] = -sqrtf(minDist); 
	}
}

void sdf2d_device(uchar* d_inData, float* d_outData, unsigned int width, unsigned int height)
{
	size_t numBytes = height * width * sizeof(float);
	assert(numBytes > 0);

	dim3 colblocksize(1);
	dim3 colgridsize(pcl::device::divUp(width, 1));

	dim3 rowblocksize(int(get_min(1024, width)), 1);
	dim3 rowgridsize(pcl::device::divUp(width, rowblocksize.x), pcl::device::divUp(height, rowblocksize.y));

	edt_cols2_kernel << <colgridsize, colblocksize, 2 * height * sizeof(float) >> > (d_inData, d_outData, width, height);
	edt_rows2_kernel << < rowgridsize, rowblocksize, width * sizeof(float) >> > (d_inData, d_outData, width, height);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}