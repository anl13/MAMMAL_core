#pragma once

#include <cuda_runtime.h>
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
void computeATA_device(const pcl::gpu::DeviceArray2D<float> &AT,
	pcl::gpu::DeviceArray2D<float> &ATA);


void computeATb_device(const pcl::gpu::DeviceArray2D<float> &AT,
	const pcl::gpu::DeviceArray<Eigen::Vector3f> source,
	const pcl::gpu::DeviceArray<Eigen::Vector3f> target,
	pcl::gpu::DeviceArray<float>& ATb
);

void check_visibility(float* imgdata, int W, int H,
	pcl::gpu::DeviceArray<Eigen::Vector3f> points,
	Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Vector3f T,
	std::vector<unsigned char>& V);

void computeATb_device(const pcl::gpu::DeviceArray2D<float>& AT,
	const pcl::gpu::DeviceArray<float> &r,
	pcl::gpu::DeviceArray<float> ATb); 
