#pragma once

#include "cuda_runtime.h"
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>
#include "../utils/safe_call.hpp"
#include <Eigen/Core>

void check_error(cudaError_t status); 
void cuda_set_device(int n); 
int cuda_get_device(); 

void setConstant2D_device(pcl::gpu::DeviceArray2D<float> &data, const float value);
void setConstant1D_device(pcl::gpu::DeviceArray<float> & data, const float value);

// In fact, A is Eigen matrix
// So, here ATA is AAT on gpu 
void computeATA_device(const pcl::gpu::DeviceArray2D<float> &A,
	pcl::gpu::DeviceArray2D<float> &ATA);


void computeATb_device(const pcl::gpu::DeviceArray2D<float> &AT,
	const pcl::gpu::DeviceArray<Eigen::Vector3f> source,
	const pcl::gpu::DeviceArray<Eigen::Vector3f> target,
	pcl::gpu::DeviceArray<float>& ATb
);