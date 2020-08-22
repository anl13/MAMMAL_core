#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <host_defines.h>

#include "vector_operations.hpp"

#include "../utils/safe_call.hpp"

#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>

#ifdef __CUDACC__
#define cuda_launch_bounds(x,y) __launch_bounds__(x,y)
#else
#define cuda_launch_bounds(x,y)
#endif

#include <Eigen/Core>
#include "pigmodeldevice.h"

//======================
// skinning forward pass 
//======================

__global__ void 
skinning_kernel(pcl::gpu::PtrSz<Eigen::Vector3f> _vertices_device,
	const pcl::gpu::PtrSz<Eigen::Matrix4f> _normalized_affine,
	const pcl::gpu::PtrStepSz<float> _weights, 
	const int _vertex_num, 
	const int _joint_num)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (vidx < _vertex_num)
	{
		Eigen::Matrix4f A = Eigen::Matrix4f::Zero(); 
#pragma unroll 
		for (int i = 0; i < _joint_num; i++)
		{
			float w = _weights(vidx, i); // ptr(y,x), row major storage
			if (w < 1e-6)
			{
			}
			else
			{
				A += _normalized_affine[i] * w;
			}
		}
		_vertices_device[vidx] = A.block<3, 3>(0, 0) * _vertices_device[vidx]
			+ A.block<3, 1>(0, 3); 
	}
}

void PigModelDevice::UpdateVerticesPosed_device()
{
	pcl::gpu::DeviceArray<Eigen::Matrix4f> device_normalizedSE3;
	device_normalizedSE3.upload(m_host_normalizedSE3);

	m_device_vertices.upload(m_host_verticesDeformed);

	dim3 block_size(32); 
	dim3 grid_size(pcl::device::divUp(m_vertexNum, block_size.x)); 

	skinning_kernel << < grid_size, block_size >> > (
		m_device_vertices, device_normalizedSE3,
		m_device_lbsweights,
		m_vertexNum, m_jointNum
		);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	m_device_vertices.download(m_host_verticesPosed); 
}

// ==============
// scale model 
// ==============

__global__ void
scale_kernel(pcl::gpu::PtrSz<Eigen::Vector3f> _vertices,
	const float _scale, 
	const int _vertexNum)
{
	int vidx = blockDim.x * blockIdx.x + threadIdx.x; 
	if (vidx < _vertexNum)
	{
		_vertices[vidx] = _vertices[vidx] * _scale; 
	}
}

void PigModelDevice::UpdateScaled_device()
{
	m_device_vertices.upload(m_host_verticesOrigin); 
	dim3 block_size(32); 
	dim3 grid_size(pcl::device::divUp(m_vertexNum, block_size.x)); 

	scale_kernel << < grid_size, block_size >> > (
		m_device_vertices, m_host_scale, m_vertexNum
		);

	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 

	m_device_vertices.download(m_host_verticesScaled); 

	for (int i = 0; i < m_jointNum; i++) m_host_jointsScaled[i] = m_host_jointsOrigin[i] * m_host_scale;
}