#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>

#include "../utils/safe_call.hpp"

typedef unsigned char uchar; 

void runCUDA(uchar* d_inData,uchar* d_middle, float* d_outData, unsigned int width, unsigned int height); 

void euclidian_distance_transform(unsigned char* img, float* dist, int w, int h); 