#pragma once

#include "cuda_runtime.h"

void check_error(cudaError_t status); 
void cuda_set_device(int n); 
int cuda_get_device(); 
