#pragma once 

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>

#include "../utils/safe_call.hpp"

void extract_depth_channel(const float4* imgdata, const int W, const int H, float* depth_device);

