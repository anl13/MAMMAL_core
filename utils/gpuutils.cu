#include "gpuutils.h"

#include <stdlib.h>
#include <time.h> 
#include <stdio.h> 

#include <device_launch_parameters.h>

// ========
// helper function 
// ========

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

// =======
// kernel function
// =======

__global__ void set_constant2d_kernel(
	pcl::gpu::PtrStepSz<float> data,
	const int W,
	const int H,
	const float value
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y; 
	if (xIdx < W && yIdx < H)
	{
		data(yIdx, xIdx) = value; 
	}
}

__global__ void set_constant1D_kernel(
	pcl::gpu::PtrSz<float> data,
	const int H,
	const float value
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	if (x < H)
	{
		data[x] = value; 
	}
}

//            dim: y
// | * --- = | | |
// |         | | |  dim: z
// |         | | |
// compute upper triangle y>=z 
__global__ void computeAAT_kernel(
	const pcl::gpu::PtrStepSz<float> A, // [y, x]
	const int W, const int H,
	pcl::gpu::PtrStepSz<float> AAT
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int zIdx = blockIdx.z * blockDim.z + threadIdx.z;
	if (xIdx < W && yIdx < H && zIdx <= yIdx)
	{
		float a_zy = A(zIdx, xIdx) *A(yIdx, xIdx);
		atomicAdd(&AAT(zIdx, yIdx), a_zy);
	}
}

#if 0
__global__ void computeAAT_kernel2(
	const pcl::gpu::PtrStepSz<float> A,
	const int W, const int H,
	pcl::gpu::PtrStepSz<float> AAT
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (xIdx < W)
	{
#pragma unroll
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < H; j++)
			{
				atomicAdd(&AAT(i, j), A(i, xIdx) * A(j, xIdx));
			}
		}
	}
}
#endif

__global__ void computeA_plus_AT_kernel(
	pcl::gpu::PtrStepSz<float> A,
	const int H
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y; 
	if (xIdx < H && yIdx < H && yIdx > xIdx)
	{
		A(yIdx, xIdx) = A(xIdx, yIdx); 
	}
}

__global__ void computeATb_kernel(
	const pcl::gpu::PtrStepSz<float> AT,//[paramnum, vertexnum*3]
	const int W, const int H,
	const pcl::gpu::PtrSz<Eigen::Vector3f> source,
	const pcl::gpu::PtrSz<Eigen::Vector3f> target,
	pcl::gpu::PtrSz<float> ATb
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
	if (x < W && y < H)
	{
		int vIdx = x / 3; 
		int aIdx = x % 3; 
		float r = - (source[vIdx](aIdx) - target[vIdx](aIdx));
		float value = AT(y, x) * r;
		atomicAdd(&ATb[y], value);
	}
}
// ======
// interface function 
// ======

void setConstant2D_device(pcl::gpu::DeviceArray2D<float> &data, const float value)
{
	const int W = data.cols(); 
	const int H = data.rows(); 
	if (W == 0 || H == 0) return; 
	dim3 blocksize(32, 32); 
	dim3 gridsize(pcl::device::divUp(W, 32), pcl::device::divUp(H, 32)); 
	set_constant2d_kernel << <gridsize, blocksize >> > (data, W, H, value); 

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

void setConstant1D_device(pcl::gpu::DeviceArray<float> & data, const float value)
{
	const int H = data.size();
	if (H == 0) return; 
	dim3 blocksize(32); 
	dim3 gridsize(pcl::device::divUp(H, 32));

	set_constant1D_kernel << <gridsize, blocksize >> > (
		data, H, value);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

void computeATA_device(const pcl::gpu::DeviceArray2D<float> &A, pcl::gpu::DeviceArray2D<float> &ATA)
{
	// infact, ATA here is AAT, [H, H]
	const int W = A.cols(); 
	const int H = A.rows(); 
	if (ATA.empty())
	{
		ATA.create(H, H);
	}

	setConstant2D_device(ATA, 0); 

	dim3 blocksize2(2, 16, 32);
	dim3 gridsize2(pcl::device::divUp(W, blocksize2.x),
		pcl::device::divUp(H, blocksize2.y),
		pcl::device::divUp(H, blocksize2.z)); 
	computeAAT_kernel << <gridsize2, blocksize2 >> > (
		A, W, H, ATA
		);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	dim3 blocksize3(32, 32); 
	dim3 gridsize3(pcl::device::divUp(H, blocksize3.x), 
		pcl::device::divUp(H, blocksize3.y)); 
	computeA_plus_AT_kernel << <gridsize3, blocksize3 >> > (
		ATA, H
		);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

void computeATb_device(const pcl::gpu::DeviceArray2D<float> &AT,
	const pcl::gpu::DeviceArray<Eigen::Vector3f> source,
	const pcl::gpu::DeviceArray<Eigen::Vector3f> target,
	pcl::gpu::DeviceArray<float>& ATb
)
{
	if (ATb.empty()) ATb.create(AT.rows()); 
	setConstant1D_device(ATb, 0); 

	int W = AT.cols(); 
	int H = AT.rows(); 
	
	dim3 blocksize(32,32); 
	dim3 gridsize(pcl::device::divUp(W, blocksize.x), pcl::device::divUp(H, blocksize.y));
	
	computeATb_kernel << < gridsize, blocksize >> > (
		AT, W, H, source, target, ATb
		);
	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 

}


__global__ void check_visibility_kernel(
	float* depth, int W, int H,
	pcl::gpu::PtrSz<Eigen::Vector3f> points,
	int Num,
	Eigen::Matrix3f K,
	Eigen::Matrix3f R,
	Eigen::Vector3f T,
	pcl::gpu::PtrSz<unsigned char> V
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (xIdx < Num)
	{
		Eigen::Vector3f p = points[xIdx];
		Eigen::Vector3f p_local = K * (R * p + T);
		float d = p_local(2);
		int u = int(p_local(0) / d + 0.5);
		int v = int(p_local(1) / d + 0.5);
		if (u < 0 || u >= W || v < 0 || v >= H) V[xIdx] = 0;
		else
		{
			float d_img = depth[v*W + u];
			float diff = d_img - d;
			if (diff > -0.02 && diff < 0.02) V[xIdx] = 1;
			else V[xIdx] = 0;
		}
	}
}

void check_visibility(float* imgdata_device, int W, int H,
	pcl::gpu::DeviceArray<Eigen::Vector3f> points,
	Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Vector3f T,
	std::vector<unsigned char>& visibility)
{
	int pointnum = points.size();
	dim3 blocksize(32);
	dim3 gridsize(pcl::device::divUp(pointnum, 32));

	pcl::gpu::DeviceArray<unsigned char> visibility_device;
	visibility_device.upload(visibility);

	check_visibility_kernel << <gridsize, blocksize >> > (
		imgdata_device, W, H, points, pointnum, K, R, T,
		visibility_device
		);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize()); 
	visibility_device.download(visibility);
	visibility_device.release(); 
}

// compute ATB
// TODO(20200904): use reduction to further speed up matrix operation. 
__global__ void multiply_AT_r_kernel(
	pcl::gpu::PtrStepSz<float> d_AT,
	pcl::gpu::PtrSz<float> d_r,
	int rows, int cols,
	pcl::gpu::PtrSz<float> output
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
	if (x >= cols || y >= rows) return; 

	float value = d_AT(y, x) * d_r[x];
	atomicAdd(&output[y], value); 

}

void computeATb_device(const pcl::gpu::DeviceArray2D<float>& AT,
	const pcl::gpu::DeviceArray<float> &r,
	pcl::gpu::DeviceArray<float> ATb)
{
	int rows = AT.rows(); 
	int cols = AT.cols(); 
	dim3 blocksize(64, 16); 
	dim3 gridsize(pcl::device::divUp(cols, blocksize.x), pcl::device::divUp(rows, blocksize.y));
	multiply_AT_r_kernel << <gridsize, blocksize >> > (AT, r, rows, cols, ATb); 
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize()); 
}