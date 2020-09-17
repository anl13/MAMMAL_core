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
skinning_kernel(const pcl::gpu::PtrSz<Eigen::Vector3f> _vertices_device,
	const pcl::gpu::PtrSz<Eigen::Matrix4f> _normalized_affine,
	const pcl::gpu::PtrStepSz<float> _weights, 
	const int _vertex_num, 
	const int _joint_num,
	pcl::gpu::PtrSz<Eigen::Vector3f> _vertices_out)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (vidx < _vertex_num)
	{
		Eigen::Matrix4f A = Eigen::Matrix4f::Zero(); 
#pragma unroll 
		for (int i = 0; i < _joint_num; i++)
		{
			float w = _weights(vidx, i); // ptr(y,x), row major storage
			if (w > 1e-6)
			{
				A += _normalized_affine[i] * w;
			}
		}
		_vertices_out[vidx] = A.block<3, 3>(0, 0) * _vertices_device[vidx]
			+ A.block<3, 1>(0, 3); 
	}
}

void PigModelDevice::UpdateVerticesPosed_device()
{
	pcl::gpu::DeviceArray<Eigen::Matrix4f> device_normalizedSE3;
	device_normalizedSE3.upload(m_host_normalizedSE3);
	
	dim3 block_size(32); 
	dim3 grid_size(pcl::device::divUp(m_vertexNum, block_size.x)); 

	skinning_kernel << < grid_size, block_size >> > (
		m_device_verticesDeformed, device_normalizedSE3,
		m_device_lbsweights,
		m_vertexNum, m_jointNum,
		m_device_verticesPosed
		);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	m_device_verticesPosed.download(m_host_verticesPosed); 
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
	m_device_verticesDeformed.upload(m_host_verticesOrigin); 
	dim3 block_size(32); 
	dim3 grid_size(pcl::device::divUp(m_vertexNum, block_size.x)); 

	scale_kernel << < grid_size, block_size >> > (
		m_device_verticesDeformed, m_host_scale, m_vertexNum
		);

	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 

	m_device_verticesDeformed.download(m_host_verticesScaled); 

#pragma omp parallel for
	for (int i = 0; i < m_jointNum; i++) m_host_jointsScaled[i] = m_host_jointsOrigin[i] * m_host_scale;
}

// ==========
// UpdateNormalsFinal
// ==========

__global__ void copy_vector3f_to_float3(
	const pcl::gpu::PtrSz<Eigen::Vector3f> input,
	const int N,
	pcl::gpu::PtrSz<float3> output
)
{
	int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (xIdx < N)
	{
		output[xIdx].x = input[xIdx].x(); 
		output[xIdx].y = input[xIdx].y(); 
		output[xIdx].z = input[xIdx].z(); 
	}
}

__global__ void copy_float3_to_vector3f(
	const pcl::gpu::PtrSz<float3> input,
	const int N,
	pcl::gpu::PtrSz<Eigen::Vector3f> output
)
{
	int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (xIdx < N)
	{
		output[xIdx](0) = input[xIdx].x;
		output[xIdx](1) = input[xIdx].y;
		output[xIdx](2) = input[xIdx].z;
	}
}

__global__ void compute_normals_step1_kernel(
	const pcl::gpu::PtrSz<Eigen::Vector3f> _vertices,
	const pcl::gpu::PtrSz<Eigen::Vector3u> _faces,
	const int _face_num,
	pcl::gpu::PtrSz<Eigen::Vector3f> normals
)
{
	// This function may encounter with much bank-conflict
	int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (xIdx < _face_num)
	{
		Eigen::Vector3u f = _faces[xIdx]; 
		Eigen::Vector3f v1 = _vertices[f(0)];
		Eigen::Vector3f v2 = _vertices[f(1)];
		Eigen::Vector3f v3 = _vertices[f(2)];
		Eigen::Vector3f n = (v1 - v2).cross(v2 - v3);
		float a = n.norm();
		a = a * a;
		n = n / a;

		atomicAdd(&normals[f(0)](0), n(0)); 
		atomicAdd(&normals[f(0)](1), n(1)); 
		atomicAdd(&normals[f(0)](2), n(2)); 

		atomicAdd(&normals[f(1)](0), n(0));
		atomicAdd(&normals[f(1)](1), n(1));
		atomicAdd(&normals[f(1)](2), n(2));

		atomicAdd(&normals[f(2)](0), n(0));
		atomicAdd(&normals[f(2)](1), n(1));
		atomicAdd(&normals[f(2)](2), n(2));
	}
}

__global__ void compute_normals_step2_kernel(
	pcl::gpu::PtrSz<Eigen::Vector3f> normals,
	const int _vertex_num
)
{
	int xIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (xIdx < _vertex_num)
	{
		normals[xIdx].normalize(); 
	}
}

void PigModelDevice::UpdateNormalsFinal_device()
{
	m_host_normalsFinal.resize(m_vertexNum, Eigen::Vector3f::Zero()); 
	m_device_normals.upload(m_host_normalsFinal); 
	
	dim3 blocksize(32); 
	dim3 gridsize_step1(pcl::device::divUp(m_faceNum, blocksize.x));
	dim3 gridsize_step2(pcl::device::divUp(m_vertexNum, blocksize.x)); 

	compute_normals_step1_kernel<<<gridsize_step1, blocksize>>> (
		m_device_verticesPosed, m_device_faces, m_faceNum,
		m_device_normals);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	compute_normals_step2_kernel<<<gridsize_step2, blocksize>>> (m_device_normals, m_vertexNum); 

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	m_device_normals.download(m_host_normalsFinal); 
}