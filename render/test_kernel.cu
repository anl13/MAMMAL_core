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
		int flip_index = (H - 1 - y) * W + x;

		output[flip_index].x = uchar(color.x());
		output[flip_index].y = uchar(color.y());
		output[flip_index].z = uchar(color.z());

		depth[flip_index] = d; 
	}
}

void gpupseudo_color(float4* imgdata, int W, int H, float maxv, float minv, cv::Mat & imgout, cv::Mat& depthout, 
	float* depth_device)
{
	std::vector<Eigen::Vector3i> CM; 
	getColorMap("jet", CM); 
	dim3 blocksize(32, 32); 
	dim3 gridsize(pcl::device::divUp(W, 32), pcl::device::divUp(H, 32)); 
	pcl::gpu::DeviceArray<Eigen::Vector3i> CM_device; 
	CM_device.upload(CM); 

	uchar3 * imgout_device;
	cudaMalloc((void**)&imgout_device, W * H * sizeof(uchar3)); 

	gpupseudo_color_kernel << <gridsize, blocksize >> > (
		imgdata, 1920, 1080, maxv, minv, CM_device, imgout_device, depth_device
		);

	imgout.create(cv::Size(1920, 1080), CV_8UC3); 
	cudaMemcpy(imgout.data, imgout_device, W*H * sizeof(uchar3), cudaMemcpyDeviceToHost); 
	depthout.create(cv::Size(1920, 1080), CV_32FC1);
	cudaMemcpy(depthout.data, depth_device, W*H * sizeof(float), cudaMemcpyDeviceToHost); 

	cudaFree(imgout_device); 
}

__global__ void check_visibility_kernel(
	float* depth, int W, int H,
	pcl::gpu::PtrSz<Eigen::Vector3f> points,
	int Num, 
	Eigen::Matrix3f K,
	Eigen::Matrix3f R,
	Eigen::Vector3f T,
	pcl::gpu::PtrSz<uchar> V
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (xIdx < Num)
	{
		Eigen::Vector3f p = points[xIdx];
		Eigen::Vector3f p_local = K * (R * p + T); 
		float d = p_local(2);
		int u = int(p_local(0)/d  + 0.5);
		int v = int(p_local(1)/d + 0.5);
		if (u < 0 || u >= W || v < 0 || v >= H) V[xIdx] = 0;
		else
		{
			float d_img = depth[v*W + u];
			float diff = d_img - d; 
			if (diff >= -0.02 && diff <= 0.02) V[xIdx] = 1; 
			else V[xIdx] = 0; 
		}
	}
}

void check_visibility(float* imgdata_device, int W, int H,
	pcl::gpu::DeviceArray<Eigen::Vector3f> points,
	Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Vector3f T,
	std::vector<uchar>& visibility)
{
	int pointnum = points.size(); 
	dim3 blocksize(32); 
	dim3 gridsize(pcl::device::divUp(pointnum, 32));

	pcl::gpu::DeviceArray<uchar> visibility_device; 
	visibility_device.upload(visibility); 

	check_visibility_kernel << <gridsize, blocksize >> > (
		imgdata_device, W, H, points, pointnum, K, R, T,
		visibility_device
		);
	visibility_device.download(visibility); 
}