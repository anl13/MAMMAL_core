#include "pigsolverdevice.h"

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
#include "gpuutils.h"

// ========
// compute jacobi full pose 
// ========

// !!! Pay extreme attention!!!
// devicearray2d store matrix in row-major type. 
// so, when you copy matrix data from eigen to pcl, 
// what you get is transposed. 
__global__ void compute_jacobi_v_full_kernel(
	pcl::gpu::PtrStepSz<float> dst,
	const pcl::gpu::PtrSz<Eigen::Vector3f> tpose_v, // vertices of tpose
	const pcl::gpu::PtrSz<Eigen::Vector3f> tpose_j, // joints of tpoose 
	const pcl::gpu::PtrStepSz<float> J_joint, // jacobi of joints
	const pcl::gpu::PtrStepSz<float> weights, // skinning_weights
	const pcl::gpu::PtrSz<int> parents, 
	const int vertexNum,
	const int jointNum,
	const pcl::gpu::PtrStepSz<float> RP, 
	const pcl::gpu::PtrStepSz<float> LP
)
{
	int vIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (vIdx < vertexNum)
	{
		Eigen::Vector3f v0 = tpose_v[vIdx];
		for (int jIdx = 0; jIdx < jointNum; jIdx++)
		{
			if (weights(vIdx, jIdx) < 0.001)continue; 
			float w = weights(vIdx, jIdx); 
			
			// like: block += (m_lbsweights(jIdx, vIdx) * jointJacobiPose.middleRows(3 * jIdx, 3));
#pragma unroll
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3 * jointNum + 3; j++)
				{
					atomicAdd(&dst(j, 3*vIdx + i), w * J_joint(j, 3*jIdx + i));
				}
			}
			// T
			Eigen::Vector3f j0 = tpose_j[jIdx];
			for (int pIdx = jIdx; pIdx > -1; pIdx = parents[pIdx])
			{
				Eigen::Matrix3f T = Eigen::Matrix3f::Zero();
#pragma unroll
				for (int axis_id = 0; axis_id < 3; axis_id++)
				{
					Eigen::Matrix3f RPblock;
					RPblock(0, 0) = RP(0, 9 * pIdx + 3 * axis_id);
					RPblock(0, 1) = RP(1, 9 * pIdx + 3 * axis_id);
					RPblock(0, 2) = RP(2, 9 * pIdx + 3 * axis_id);
					RPblock(1, 0) = RP(0, 9 * pIdx + 3 * axis_id+1);
					RPblock(1, 1) = RP(1, 9 * pIdx + 3 * axis_id+1);
					RPblock(1, 2) = RP(2, 9 * pIdx + 3 * axis_id+1);
					RPblock(2, 0) = RP(0, 9 * pIdx + 3 * axis_id+2);
					RPblock(2, 1) = RP(1, 9 * pIdx + 3 * axis_id+2);
					RPblock(2, 2) = RP(2, 9 * pIdx + 3 * axis_id+2);

	
					Eigen::Matrix3f LPblock;
					LPblock(0, 0) = LP(3*pIdx,3*jIdx);
					LPblock(0, 1) = LP(3 * pIdx+1, 3 * jIdx);
					LPblock(0, 2) = LP(3 * pIdx+2, 3 * jIdx);
					LPblock(1, 0) = LP(3 * pIdx, 3 * jIdx+1);
					LPblock(1, 1) = LP(3 * pIdx+1, 3 * jIdx+1);
					LPblock(1, 2) = LP(3 * pIdx+2, 3 * jIdx+1);
					LPblock(2, 0) = LP(3 * pIdx, 3 * jIdx+2);
					LPblock(2, 1) = LP(3 * pIdx+1, 3 * jIdx+2);
					LPblock(2, 2) = LP(3 * pIdx+2, 3 * jIdx+2);

					Eigen::Matrix3f dR1 = RPblock * LPblock; 
					T.col(axis_id) = dR1 * (v0 - j0) * w; 
				}
#pragma unroll
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						atomicAdd(&dst( 3+3*pIdx+j, 3 * vIdx + i), T(i,j));
					}
				}
			}
		}
	}
}

__global__ void extract_jacobi_lines_kernel(
	const pcl::gpu::PtrStepSz<float> full, // [3+3*jointnum, 3*jointnum or 3 * vertexnum]
	const pcl::gpu::PtrSz<int> ids,
	const int ids_num,
	const int cols,
	pcl::gpu::PtrStepSz<float> out_partial // [ids_num, 3*jointnum or 3*vertexnum]
)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if (xIdx < cols && yIdx < ids_num)
	{
		int id = ids[yIdx]; 
		out_partial(yIdx, xIdx) = full(id, xIdx); 
	}
}


// ===========
// Non-kernel functions 
// ===========

void PigSolverDevice::calcPoseJacobiFullTheta_device(
	pcl::gpu::DeviceArray2D<float> &J_joint,
	pcl::gpu::DeviceArray2D<float> &J_vert
)
{
	int cpucols = 3 * m_jointNum + 3; // theta dimension 
	if(J_joint.empty())
		J_joint.create(cpucols, m_jointNum * 3); // J_joint_cpu.T, with same storage array  
	if (J_vert.empty())
		J_vert.create(cpucols, m_vertexNum * 3); // J_vert_cpu.T 
	setConstant2D_device(J_joint, m_jointNum * 3, cpucols, 0);
	setConstant2D_device(J_vert, m_vertexNum * 3, cpucols, 0); 

	TimerUtil::Timer<std::chrono::microseconds> tt; 
	tt.Start(); 

	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * m_jointNum);
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3f& pose = m_host_poseParam[jointId];
		rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiF(pose);
	}


	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> RP(9 * m_jointNum, 3);
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> LP(3 * m_jointNum, 3 * m_jointNum);
	RP.setZero();
	LP.setZero();

	for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
	{
		for (int aIdx = 0; aIdx < 3; aIdx++)
		{
			Eigen::Matrix3f dR = rodriguesDerivative.block<3, 3>(0, 3 * (3 * jIdx + aIdx));
			if (jIdx > 0)
			{
				dR = m_host_globalSE3[m_host_parents[jIdx]].block<3, 3>(0, 0) * dR;
			}
			RP.block<3, 3>(9 * jIdx + 3 * aIdx, 0) = dR;
		}
		LP.block<3, 3>(3 * jIdx, 3 * jIdx) = Eigen::Matrix3f::Identity();
		for (int child = jIdx + 1; child < m_jointNum; child++)
		{
			int father = m_host_parents[child];
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx) * m_host_localSE3[child].block<3,3>(0,0);
		}
	}

	Eigen::MatrixXf jointJacobiPose = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
	for (int jointDerivativeId = 0; jointDerivativeId < m_jointNum; jointDerivativeId++)
	{
		// update translation term
		jointJacobiPose.block<3, 3>(jointDerivativeId * 3, 0).setIdentity();

		// update poseParam term
		for (int axisDerivativeId = 0; axisDerivativeId < 3; axisDerivativeId++)
		{
			std::vector<std::pair<bool, Eigen::Matrix4f>> globalAffineDerivative(m_jointNum, std::make_pair(false, Eigen::Matrix4f::Zero()));
			globalAffineDerivative[jointDerivativeId].first = true;
			auto& affine = globalAffineDerivative[jointDerivativeId].second;
			affine.block<3, 3>(0, 0) = rodriguesDerivative.block<3, 3>(0, 3 * (3 * jointDerivativeId + axisDerivativeId));
			affine = jointDerivativeId == 0 ? affine : (m_host_globalSE3[ m_host_parents[jointDerivativeId]] * affine);

			for (int jointId = jointDerivativeId + 1; jointId < m_jointNum; jointId++)
			{
				if (globalAffineDerivative[m_host_parents[jointId]].first)
				{
					globalAffineDerivative[jointId].first = true;
					globalAffineDerivative[jointId].second = globalAffineDerivative[m_host_parents[jointId]].second * m_host_localSE3[jointId];
					// update jacobi for pose
					jointJacobiPose.block<3, 1>(jointId * 3, 3 + jointDerivativeId * 3 + axisDerivativeId) = globalAffineDerivative[jointId].second.block<3, 1>(0, 3);
				}
			}
		}
	}

	std::cout << "compute J_joint on cpu takes: " << tt.Elapsed() << " mcs" << std::endl; 
	tt.Start(); 

	J_joint.upload(jointJacobiPose.data(), (3*m_jointNum) * sizeof(float), cpucols, 3 * m_jointNum);
	m_device_jointsDeformed.upload(m_host_jointsDeformed); 
	pcl::gpu::DeviceArray2D<float> RP_device, LP_device;
	RP_device.upload(RP.data(), 9*m_jointNum * sizeof(float), 3, 9*m_jointNum); 
	LP_device.upload(LP.data(), 3 * m_jointNum * sizeof(float), 3 * m_jointNum, 3*m_jointNum); 

	dim3 blocksize(32);
	dim3 gridsize(pcl::device::divUp(m_vertexNum, 32));

	
	compute_jacobi_v_full_kernel << <gridsize, blocksize >> > (
		J_vert,
		m_device_verticesDeformed, // vertices of tpose
		m_device_jointsDeformed, // joints of tpoose 
		J_joint, // jacobi of joints
		m_device_lbsweights, // skinning_weights
		m_device_parents,
		m_vertexNum,
		m_jointNum,
		RP_device,
		LP_device
		);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	RP_device.release();
	LP_device.release(); 

	std::cout << "compute J_vert on gpu takes:  " << tt.Elapsed() << std::endl; 
}


void PigSolverDevice::calcPoseJacobiPartTheta_device(pcl::gpu::DeviceArray2D<float> &J_joint,
	pcl::gpu::DeviceArray2D<float> &J_vert)
{
	dim3 blocksize(32, 32); 
	std::cout << "lines: " << m_host_paramLines.size() << std::endl; 
	dim3 gridsize1(pcl::device::divUp(3 * m_jointNum, blocksize.x), pcl::device::divUp(m_host_paramLines.size(), blocksize.y));
	dim3 gridsize2(pcl::device::divUp(3 * m_vertexNum, blocksize.x), pcl::device::divUp(m_host_paramLines.size(), blocksize.y)); 

	pcl::gpu::DeviceArray2D<float> J_joint_full, J_vert_full;
	calcPoseJacobiFullTheta_device(J_joint_full, J_vert_full); 

	TimerUtil::Timer<std::chrono::microseconds> tt; 
	tt.Start(); 

	if (J_joint.empty())
	{
		J_joint.create(m_host_paramLines.size(), 3 * m_jointNum);
	}
	if (J_vert.empty())
	{
		J_vert.create(m_host_paramLines.size(), 3 * m_vertexNum); 
	}
	extract_jacobi_lines_kernel << <gridsize1, blocksize >> > (
		J_joint_full, m_device_paramLines, m_host_paramLines.size(),
		3 * m_jointNum, J_joint
		);
	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 

	extract_jacobi_lines_kernel << <gridsize2, blocksize >> > (
		J_vert_full, m_device_paramLines, m_host_paramLines.size(),
		3 * m_vertexNum, J_vert
		);
	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 

	J_joint_full.release(); 
	J_vert_full.release(); 

	std::cout << "compute partial jacobi on gpu takes: " << tt.Elapsed() << std::endl; 
}