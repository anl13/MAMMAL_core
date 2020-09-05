#include "pigsolverdevice.h"

#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <host_defines.h>

#include "vector_operations.hpp"

#include "../utils/safe_call.hpp"

#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>

#include <fstream> 

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
			if (weights(vIdx, jIdx) < 0.00001)continue; 
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
	setConstant2D_device(J_joint, 0);
	setConstant2D_device(J_vert, 0); 

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

	//std::ofstream out("G:/pig_results/J_joint_in.txt"); 
	//out << jointJacobiPose;
	//out.close(); 
	std::cout << " we are the champione." << std::endl;
		
	J_joint.upload(jointJacobiPose.data(), (3*m_jointNum) * sizeof(float), cpucols, 3 * m_jointNum);
	m_device_jointsDeformed.upload(m_host_jointsDeformed); 
	m_device_verticesDeformed.upload(m_host_verticesDeformed); 
	d_RP.upload(RP.data(), 9*m_jointNum * sizeof(float), 3, 9*m_jointNum); 
	d_LP.upload(LP.data(), 3 * m_jointNum * sizeof(float), 3 * m_jointNum, 3*m_jointNum); 

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
		d_RP,
		d_LP
		);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}


void PigSolverDevice::calcPoseJacobiPartTheta_device(pcl::gpu::DeviceArray2D<float> &J_joint,
	pcl::gpu::DeviceArray2D<float> &J_vert)
{
	dim3 blocksize(32, 32); 
	dim3 gridsize1(pcl::device::divUp(3 * m_jointNum, blocksize.x), pcl::device::divUp(m_host_paramLines.size(), blocksize.y));
	dim3 gridsize2(pcl::device::divUp(3 * m_vertexNum, blocksize.x), pcl::device::divUp(m_host_paramLines.size(), blocksize.y)); 

	calcPoseJacobiFullTheta_device(d_J_joint_full, d_J_vert_full); 

	extract_jacobi_lines_kernel << <gridsize1, blocksize >> > (
		d_J_joint_full, m_device_paramLines, m_host_paramLines.size(),
		3 * m_jointNum, J_joint
		);
	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 

	extract_jacobi_lines_kernel << <gridsize2, blocksize >> > (
		d_J_vert_full, m_device_paramLines, m_host_paramLines.size(),
		3 * m_vertexNum, J_vert
		);
	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 
}

__global__ void construct_sil_A_kernel(
	pcl::gpu::PtrStepSz<float> d_J_vert, // [paramNum, 3*vertexnum]
	pcl::gpu::PtrSz<BODY_PART> d_parts,
	pcl::gpu::PtrSz<Eigen::Vector3f> d_points3d,
	Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Vector3f T,
	float* d_depth,
	pcl::gpu::PtrStepSz<uchar> d_mask, // mask for all pigs
	pcl::gpu::PtrStepSz<uchar> d_scene_mask, // mask for scene staff
	pcl::gpu::PtrStepSz<uchar> d_distort_mask, // mask for distortion area
	pcl::gpu::PtrStepSz<float> d_det_sdf, // sdf for detection 
	pcl::gpu::PtrStepSz<float> d_det_gradx,
	pcl::gpu::PtrStepSz<float> d_det_grady,
	pcl::gpu::PtrStepSz<float> d_rend_sdf, // sdf for rendering
	int W, int H, int pointNum, int paramNum, int id,
	pcl::gpu::PtrStepSz<float> AT, // [paramNum, pointNum]
	pcl::gpu::PtrSz<float> b, //[pointnum],
	float* count
)
{
	unsigned int vIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int paramIdx = blockIdx.y * blockDim.y + threadIdx.y; 
	if (vIdx >= pointNum || paramIdx>=paramNum) return; 
	
	if (d_parts[vIdx] == TAIL) return; 
	if (d_parts[vIdx] == L_EAR) return;
	if (d_parts[vIdx] == R_EAR) return; 
	Eigen::Vector3f point2d = K * (R*d_points3d[vIdx] + T);
	int u = int(point2d(0) / point2d(2) + 0.5);
	int v = int(point2d(1) / point2d(2) + 0.5);
	if (u < 0 || v < 0 || u >= W || v >= H)return; //out of image
	int index = v * W + u;
	float rend_sdf_value = d_rend_sdf(v,u);
	if (rend_sdf_value > 10)return; // only consider contours 
	float depth_value = point2d(2);
	if (d_distort_mask(v,u) == 0) return; // out of image
	if (d_scene_mask(v,u) > 0)return;// occluded by scene
	if (fabsf(d_depth[index] - depth_value) >= 0.02)return; // invisible

	uchar code = d_mask(v,u);
	if (code > 0 && code != id)return; // occlued by other pig

	float det_sdf_value = d_det_sdf(v,u);
	if (det_sdf_value < -9999) return;
	
	Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
	D(0, 0) = 1 / point2d(2);
	D(1, 1) = 1 / point2d(2);
	D(0, 2) = -point2d(0) / (point2d(2)*point2d(2));
	D(1, 2) = -point2d(1) / (point2d(2) * point2d(2)); 
	Eigen::Vector3f dp;
	dp(0) = d_J_vert(paramIdx, 3 * vIdx + 0);
	dp(1) = d_J_vert(paramIdx, 3 * vIdx + 1);
	dp(2) = d_J_vert(paramIdx, 3 * vIdx + 2);
	Eigen::Vector2f dpsil;
	dpsil = D * K*R*dp;
	float dx = d_det_gradx(v,u);
	float dy = d_det_grady(v,u);
	b[vIdx] = (rend_sdf_value - det_sdf_value) * 0.01;
	AT(paramIdx, vIdx) = ( dpsil(0) * dx + dpsil(1) * dy ) * 0.01;

	atomicAdd(count, 1); 
}


void PigSolverDevice::calcSilhouetteJacobi_device(
	Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Vector3f T,
	float* d_depth, int idcode, int paramNum
)
{
	dim3 blocksize(32, 32);
	dim3 gridsize(pcl::device::divUp(m_vertexNum, blocksize.x), pcl::device::divUp(paramNum, blocksize.y));

	pcl::gpu::DeviceArray<float> d_count; 
	std::vector<float> h_count(1, 0); 
	d_count.upload(h_count); 

	construct_sil_A_kernel << <gridsize, blocksize >> > (
		d_J_vert, m_device_bodyParts, m_device_verticesPosed, K, R, T,
		d_depth, d_det_mask, d_const_scene_mask, d_const_distort_mask,
		d_det_sdf, d_det_gradx, d_det_grady, d_rend_sdf,
		1920, 1080, m_vertexNum, paramNum, idcode, 
		d_JT_sil, d_r_sil, d_count
		);
	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 

	d_count.download(h_count);
	printf("visible: %f\n", h_count[0] / paramNum);
}