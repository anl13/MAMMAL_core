
#include "distance_transform_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define BLOCK_SIZE 256

__global__ void euclidian_distance_transform_kernel(
	const unsigned char* img, float* dist, int w, int h)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int N = w * h;

	if (i >= N)
	{
		return;
	}

	int cx = i % w;
	int cy = i / w;

	float minv = INFINITY;

	if (img[i] > 0)
	{
		minv = 0.0f;
	}
	else
	{
		for (int j = 0; j < N; j++)
		{
			if (img[j] > 0)
			{
				int x = j % w;
				int y = j / w;
				float d = sqrtf(powf(float(x - cx), 2.0f) + powf(float(y - cy), 2.0f));
				if (d < minv) minv = d;
			}
		}
	}

	dist[i] = minv;
}

void euclidian_distance_transform(unsigned char* img, float* dist, int w, int h) {

	cudaError_t err;
	unsigned char *d_img;
	cudaMalloc((void**)&d_img, w*h * sizeof(unsigned char));
	cudaMemcpy(d_img, img, w*h * sizeof(unsigned char), cudaMemcpyHostToDevice);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
	}

	float* d_dist;
	cudaMalloc((void**)&d_dist, w*h * sizeof(float));
	//cudaMemset(d_dist, 0, w*h*sizeof(float));

	dim3 block(BLOCK_SIZE, 1, 1);

	int gx = (w*h + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 grid(gx, 1);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
	}

	euclidian_distance_transform_kernel << <grid, block >> > (d_img, d_dist, w, h);
	cudaThreadSynchronize();

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
	}

	cudaMemcpy(dist, d_dist, w*h * sizeof(float), cudaMemcpyDeviceToHost);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
	}

	cudaFree(d_img);
	cudaFree(d_dist);
}