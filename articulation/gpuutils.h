#pragma once

#include "cuda_runtime.h"
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>
#include "../utils/safe_call.hpp"

void check_error(cudaError_t status); 
void cuda_set_device(int n); 
int cuda_get_device(); 

void setConstant2D_device(pcl::gpu::DeviceArray2D<float> &data, const int W, const int H, const float value);

// In fact, A is Eigen matrix
// So, here ATA is AAT on gpu 
void computeATA_device(const pcl::gpu::DeviceArray2D<float> &A, const int W, const int H,
	pcl::gpu::DeviceArray2D<float> &ATA);