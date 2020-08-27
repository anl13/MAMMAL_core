#include <Eigen/Core>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "test_kernel.h"
#include "../utils/image_utils.h"


__global__ void gpupseudo_color_kernel(
	float4 *imgdata, const int W, const int H, const float maxv, const float minv,
	const pcl::gpu::PtrSz<Eigen::Vector3i> cm,
	uchar3 *output, float *depth
)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
	if (x < W && y < H)
	{
		int index = W * y + x; 
		float4 data = imgdata[index];
		float d = -data.z;
		int outid = 0; 
		if (d < 0 || d > 1) outid = 0; 

			outid = int((d - minv) / (maxv - minv + 0.00001) * 255);

		Eigen::Vector3i color = cm[outid];
		output[index].x = uchar(color.x());
		output[index].y = uchar(color.y());
		output[index].z = uchar(color.z());
		depth[index] = d; 
	}
}

void gpupseudo_color(float4* imgdata, int W, int H, float maxv, float minv, cv::Mat & imgout, cv::Mat& depthout)
{
	std::vector<Eigen::Vector3i> CM; 
	getColorMap("jet", CM); 
	dim3 blocksize(32, 32); 
	dim3 gridsize(pcl::device::divUp(W, 32), pcl::device::divUp(H, 32)); 
	pcl::gpu::DeviceArray<Eigen::Vector3i> CM_device; 
	CM_device.upload(CM); 

	uchar3 * imgout_device;
	cudaMalloc((void**)&imgout_device, W * H * sizeof(uchar3)); 

	float * depthout_device; 
	cudaMalloc((void**)&depthout_device, W*H * sizeof(float)); 
	gpupseudo_color_kernel << <gridsize, blocksize >> > (
		imgdata, 1920, 1080, maxv, minv, CM_device, imgout_device, depthout_device
		);

	imgout.create(cv::Size(1920, 1080), CV_8UC3); 
	cudaMemcpy(imgout.data, imgout_device, W*H * sizeof(uchar3), cudaMemcpyDeviceToHost); 
	depthout.create(cv::Size(1920, 1080), CV_32FC1);
	cudaMemcpy(depthout.data, depthout_device, W*H * sizeof(float), cudaMemcpyDeviceToHost); 
}

//__global__ void max(int *a, int *c)
//{
//	extern __shared__ int sdata[];
//
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//
//	sdata[tid] = a[i];
//
//	__syncthreads();
//	for (unsigned int s = blockDim.x / 2; s >= 1; s = s / 2)
//	{
//		if (tid< s)
//		{
//			if (s[tid]>sdata[tid + s])
//			{
//				sdata[tid] = sdata[tid + s];
//			}
//		}
//		//////////////////////////////
//		__syncthreads();
//	}
//	if (tid == 0) c[blockIdx.x] = sdata[0];
//}