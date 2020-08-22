#include "gpuutils.h"

#include <stdlib.h>
#include <time.h> 
#include <stdio.h> 

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