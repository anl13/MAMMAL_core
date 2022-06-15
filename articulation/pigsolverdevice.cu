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
#include "../utils/gpuutils.h"

// artist model 
#define NUM_MODEL_JOINT 62
#define NUM_POSE_JOINT 22

// my pig model
//#define NUM_MODEL_JOINT 43
//#define NUM_POSE_JOINT 17

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
			// allocate data space once for all. 
		Eigen::Vector3f v0 = tpose_v[vIdx];
		Eigen::Vector3f j0;
		Eigen::Matrix3f dR1;
		Eigen::Matrix3f Tmp = Eigen::Matrix3f::Zero();
		Eigen::Matrix3f RPblock;
		Eigen::Matrix3f LPblock;
		float J[3*(3*NUM_MODEL_JOINT+3)]; // 567 = 3 * (3+62*3)
#pragma unroll 
		for (int i = 0; i < 3 * (3 * NUM_MODEL_JOINT + 3); i++) J[i] = 0;
		for (int jIdx = 0; jIdx < jointNum; jIdx++)
		{
			if (weights(vIdx, jIdx) < 0.00001)continue;
			float w = weights(vIdx, jIdx);

			// like: block += (m_lbsweights(jIdx, vIdx) * jointJacobiPose.middleRows(3 * jIdx, 3));
#pragma unroll

			for (int j = 0; j < 3 * jointNum + 3; j++)
			{
				for (int i = 0; i < 3; i++)
				{
					//atomicAdd(&dst(j, 3 * vIdx + i), w * J_joint(j, 3 * jIdx + i));
					J[3 * j + i] += w * J_joint(j, 3 * jIdx + i);
				}
			}
			// T
			j0 = tpose_j[jIdx];

			for (int pIdx = jIdx; pIdx > -1; pIdx = parents[pIdx])
			{
				Tmp.setZero();
#pragma unroll
				for (int axis_id = 0; axis_id < 3; axis_id++)
				{

					RPblock(0, 0) = RP(0, 9 * pIdx + 3 * axis_id);
					RPblock(0, 1) = RP(1, 9 * pIdx + 3 * axis_id);
					RPblock(0, 2) = RP(2, 9 * pIdx + 3 * axis_id);
					RPblock(1, 0) = RP(0, 9 * pIdx + 3 * axis_id + 1);
					RPblock(1, 1) = RP(1, 9 * pIdx + 3 * axis_id + 1);
					RPblock(1, 2) = RP(2, 9 * pIdx + 3 * axis_id + 1);
					RPblock(2, 0) = RP(0, 9 * pIdx + 3 * axis_id + 2);
					RPblock(2, 1) = RP(1, 9 * pIdx + 3 * axis_id + 2);
					RPblock(2, 2) = RP(2, 9 * pIdx + 3 * axis_id + 2);

					LPblock(0, 0) = LP(3 * pIdx, 3 * jIdx);
					LPblock(0, 1) = LP(3 * pIdx + 1, 3 * jIdx);
					LPblock(0, 2) = LP(3 * pIdx + 2, 3 * jIdx);
					LPblock(1, 0) = LP(3 * pIdx, 3 * jIdx + 1);
					LPblock(1, 1) = LP(3 * pIdx + 1, 3 * jIdx + 1);
					LPblock(1, 2) = LP(3 * pIdx + 2, 3 * jIdx + 1);
					LPblock(2, 0) = LP(3 * pIdx, 3 * jIdx + 2);
					LPblock(2, 1) = LP(3 * pIdx + 1, 3 * jIdx + 2);
					LPblock(2, 2) = LP(3 * pIdx + 2, 3 * jIdx + 2);
					dR1 = RPblock * LPblock;
					Tmp.col(axis_id) = dR1 * (v0 - j0) * w;
				}
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						J[3 * (3 + 3 * pIdx) + 3 * i + j] += Tmp(j, i);
					}
				}
			}
		}
		for (int j = 0; j < 3 * jointNum + 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				dst(j, 3 * vIdx + i) = J[3 * j + i];
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


void PigSolverDevice::calcPoseJacobiPartTheta_device(pcl::gpu::DeviceArray2D<float> &J_joint,
	pcl::gpu::DeviceArray2D<float> &J_vert, bool with_vert)
{
	dim3 blocksize(32, 32); 
	dim3 gridsize1(pcl::device::divUp(3 * m_jointNum, blocksize.x), pcl::device::divUp(m_host_paramLines.size(), blocksize.y));
	dim3 gridsize2(pcl::device::divUp(3 * m_vertexNum, blocksize.x), pcl::device::divUp(m_host_paramLines.size(), blocksize.y)); 

	calcPoseJacobiFullTheta_device(d_J_joint_full, d_J_vert_full, with_vert); 
	extract_jacobi_lines_kernel << <gridsize1, blocksize >> > (
		d_J_joint_full, m_device_paramLines, m_host_paramLines.size(),
		3 * m_jointNum, J_joint
		);
	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 

	if (with_vert)
	{
		extract_jacobi_lines_kernel << <gridsize2, blocksize >> > (
			d_J_vert_full, m_device_paramLines, m_host_paramLines.size(),
			3 * m_vertexNum, J_vert
			);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDeviceSynchronize());
	}
}

__global__ void construct_sil_A_kernel(
	const pcl::gpu::PtrSz<Eigen::Vector3f> tpose_v, // vertices of tpose
	const pcl::gpu::PtrSz<Eigen::Vector3f> tpose_j, // joints of tpoose 
	const pcl::gpu::PtrStepSz<float> J_joint, // jacobi of joints
	const pcl::gpu::PtrStepSz<float> weights, // skinning_weights
	const pcl::gpu::PtrSz<int> parents,
	const int vertexNum,
	const int jointNum,
	const pcl::gpu::PtrStepSz<float> RP,
	const pcl::gpu::PtrStepSz<float> LP,
	pcl::gpu::PtrSz<int> d_paramLines,

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
	pcl::gpu::PtrStepSz<float> d_ATA_sil, // [paramNum, paramNum]
	pcl::gpu::PtrSz<float> d_ATb_sil //[paramNum],
	//pcl::gpu::PtrStepSz<float> d_AT, // [paramNum, pointnum]
	//pcl::gpu::PtrSz<float> d_b //[pointnum],
)
{
	unsigned int vIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (vIdx >= pointNum) return; 
	
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
	float dx = d_det_gradx(v, u);
	float dy = d_det_grady(v, u);

	// allocate data space once for all. 
	Eigen::Vector3f v0 = tpose_v[vIdx];
	Eigen::Vector3f j0; 
	Eigen::Matrix3f dR1;
	Eigen::Matrix3f Tmp = Eigen::Matrix3f::Zero();
	Eigen::Matrix3f RPblock;
	Eigen::Matrix3f LPblock;
	float J[3 * (3 * NUM_MODEL_JOINT + 3)]; // 567 = 3 * (3+62*3)
#pragma unroll 
	for (int i = 0; i < 3 * (3 * NUM_MODEL_JOINT + 3); i++) J[i] = 0;
	for (int jIdx = 0; jIdx < jointNum; jIdx++)
	{
		if (weights(vIdx, jIdx) < 0.00001)continue;
		float w = weights(vIdx, jIdx);

		// like: block += (m_lbsweights(jIdx, vIdx) * jointJacobiPose.middleRows(3 * jIdx, 3));
#pragma unroll

		for (int j = 0; j < 3*jointNum+3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				//atomicAdd(&dst(j, 3 * vIdx + i), w * J_joint(j, 3 * jIdx + i));
				J[3*j+i] += w * J_joint(j, 3 * jIdx + i);
			}
		}
		// T
		j0 = tpose_j[jIdx];
		
		for (int pIdx = jIdx; pIdx > -1; pIdx = parents[pIdx])
		{
			Tmp.setZero(); 
#pragma unroll
			for (int axis_id = 0; axis_id < 3; axis_id++)
			{
				
				RPblock(0, 0) = RP(0, 9 * pIdx + 3 * axis_id);
				RPblock(0, 1) = RP(1, 9 * pIdx + 3 * axis_id);
				RPblock(0, 2) = RP(2, 9 * pIdx + 3 * axis_id);
				RPblock(1, 0) = RP(0, 9 * pIdx + 3 * axis_id + 1);
				RPblock(1, 1) = RP(1, 9 * pIdx + 3 * axis_id + 1);
				RPblock(1, 2) = RP(2, 9 * pIdx + 3 * axis_id + 1);
				RPblock(2, 0) = RP(0, 9 * pIdx + 3 * axis_id + 2);
				RPblock(2, 1) = RP(1, 9 * pIdx + 3 * axis_id + 2);
				RPblock(2, 2) = RP(2, 9 * pIdx + 3 * axis_id + 2);

				LPblock(0, 0) = LP(3 * pIdx, 3 * jIdx);
				LPblock(0, 1) = LP(3 * pIdx + 1, 3 * jIdx);
				LPblock(0, 2) = LP(3 * pIdx + 2, 3 * jIdx);
				LPblock(1, 0) = LP(3 * pIdx, 3 * jIdx + 1);
				LPblock(1, 1) = LP(3 * pIdx + 1, 3 * jIdx + 1);
				LPblock(1, 2) = LP(3 * pIdx + 2, 3 * jIdx + 1);
				LPblock(2, 0) = LP(3 * pIdx, 3 * jIdx + 2);
				LPblock(2, 1) = LP(3 * pIdx + 1, 3 * jIdx + 2);
				LPblock(2, 2) = LP(3 * pIdx + 2, 3 * jIdx + 2);
				dR1 = RPblock * LPblock; 
				Tmp.col(axis_id) = dR1 * (v0 - j0) * w;
			}
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					J[3 * (3 + 3 * pIdx) + 3 * i + j] += Tmp(j,i);
				}
			}
		}
	}
	
	// reuse Tmp matrix as D. 
	Tmp.setZero();
	Tmp(0, 0) = 1 / point2d(2);
	Tmp(1, 1) = 1 / point2d(2);
	Tmp(0, 2) = -point2d(0) / (point2d(2)*point2d(2));
	Tmp(1, 2) = -point2d(1) / (point2d(2) * point2d(2)); 
	// reuse v0 as dp 
	// reuse j0 as dpsil
	int J_V_index = -1;
	float A_col[3*NUM_POSE_JOINT+3]; // 75 is paramNum
#pragma unroll
	for (int i = 0; i < paramNum; i++)
	{
		J_V_index = d_paramLines[i];
		v0(0) = J[3*J_V_index];
		v0(1) = J[3*J_V_index+1];
		v0(2) = J[3*J_V_index+2];
		
		j0 = Tmp * K*R* v0;
		A_col[i] = (j0(0) * dx + j0(1) * dy) * 0.01;
	}
	float b = (rend_sdf_value - det_sdf_value) * 0.01;

	//for (int i = 0; i < paramNum; i++)d_AT(i, vIdx) = A_col[i];
	//d_b[vIdx] = b;

	for (int i = 0; i < paramNum; i++)
	{
		for (int j = 0; j < paramNum; j++)
		{
			atomicAdd(&d_ATA_sil(j, i), A_col[i] * A_col[j]);
		}
		atomicAdd(&d_ATb_sil[i], A_col[i] * b);
	}
}


__global__ void construct_sil_A_kernel2(
	const pcl::gpu::PtrSz<Eigen::Vector3f> tpose_v, // vertices of tpose
	const pcl::gpu::PtrSz<Eigen::Vector3f> tpose_j, // joints of tpoose 
	const pcl::gpu::PtrStepSz<float> J_joint, // jacobi of joints
	const pcl::gpu::PtrStepSz<float> weights, // skinning_weights
	const pcl::gpu::PtrSz<int> parents,
	const int vertexNum,
	const int jointNum,
	const pcl::gpu::PtrStepSz<float> RP,
	const pcl::gpu::PtrStepSz<float> LP,
	pcl::gpu::PtrSz<int> d_paramLines,

	pcl::gpu::PtrSz<BODY_PART> d_parts,
	pcl::gpu::PtrSz<Eigen::Vector3f> d_points3d,
	Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Vector3f T,
	float* d_depth, // 1920*1080
	float* d_depth_interact, 
	uchar* d_mask, // mask for all pigs, 1920*1080
	uchar* d_scene_mask, // mask for scene oclussion, 1920*1080
	uchar* d_distort_mask, // mask for distortion, 1920*1080
	float* d_det_sdf, // sdf for detection, 960*540 
	float* d_det_gradx, // sobel x: 960*540
	float* d_det_grady, // sobel y: 960 * 540
	float* d_rend_sdf, // sdf for rendering, 960*540
	int W, int H, int pointNum, int paramNum, int idcode,
#ifdef DEBUG_SOLVER
	pcl::gpu::PtrStepSz<float> d_AT, // [paramNum, pointnum]
pcl::gpu::PtrSz<float> d_b //[pointnum],
#else
	pcl::gpu::PtrStepSz<float> d_ATA_sil, // [paramNum, paramNum]
	pcl::gpu::PtrSz<float> d_ATb_sil //[paramNum],
#endif 
)
{
	unsigned int vIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vIdx >= pointNum) return;

	if (d_parts[vIdx] == TAIL) return;
	if (d_parts[vIdx] == L_EAR) return;
	if (d_parts[vIdx] == R_EAR) return;
	Eigen::Vector3f point2d = K * (R*d_points3d[vIdx] + T);
	int u = int(point2d(0) / point2d(2) + 0.5);
	int v = int(point2d(1) / point2d(2) + 0.5);
	int u_half = int(point2d(0) / point2d(2) / 2 + 0.5); 
	int v_half = int(point2d(1) / point2d(2) / 2 + 0.5); 
	if (u < 0 || v < 0 || u >= W || v >= H)return; //out of image
	int index = v * W + u;
	
	if (d_distort_mask[index] == 0) return; // out of image
	if (d_scene_mask[index] > 0)return;// occluded by scene
	uchar code = d_mask[index];
	if (code > 0 && code != idcode)return; // occlued by other pig

	float depth_value = point2d(2);
	if (fabsf(d_depth_interact[index] - depth_value) >= 0.02)return; // invisible

	int index_half = v_half * W / 2 + u_half;
	float rend_sdf_value = d_rend_sdf[index_half]*2 ;
	if (rend_sdf_value > 10)return; // only consider contours 

	float det_sdf_value = d_det_sdf[index_half]*2;
	if (det_sdf_value > 40) return; //ignore too far away points which may be caused by other errors
	float dx = d_det_gradx[index_half]*2;
	float dy = d_det_grady[index_half]*2;

	// allocate data space once for all. 
	Eigen::Vector3f v0 = tpose_v[vIdx];
	Eigen::Vector3f j0;
	Eigen::Matrix3f dR1;
	Eigen::Matrix3f Tmp = Eigen::Matrix3f::Zero();
	Eigen::Matrix3f RPblock;
	Eigen::Matrix3f LPblock;
	float J[3*(3+3*NUM_MODEL_JOINT)]; // 567 = 3 * (3+62*3)
#pragma unroll 
	for (int i = 0; i < 3 * (3 + 3 * NUM_MODEL_JOINT); i++) J[i] = 0;
	for (int jIdx = 0; jIdx < jointNum; jIdx++)
	{
		if (weights(vIdx, jIdx) < 0.00001)continue;
		float w = weights(vIdx, jIdx);

		// like: block += (m_lbsweights(jIdx, vIdx) * jointJacobiPose.middleRows(3 * jIdx, 3));
#pragma unroll

		for (int j = 0; j < 3 * jointNum + 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				//atomicAdd(&dst(j, 3 * vIdx + i), w * J_joint(j, 3 * jIdx + i));
				J[3 * j + i] += w * J_joint(j, 3 * jIdx + i);
			}
		}
		// T
		j0 = tpose_j[jIdx];

		for (int pIdx = jIdx; pIdx > -1; pIdx = parents[pIdx])
		{
			Tmp.setZero();
#pragma unroll
			for (int axis_id = 0; axis_id < 3; axis_id++)
			{

				RPblock(0, 0) = RP(0, 9 * pIdx + 3 * axis_id);
				RPblock(0, 1) = RP(1, 9 * pIdx + 3 * axis_id);
				RPblock(0, 2) = RP(2, 9 * pIdx + 3 * axis_id);
				RPblock(1, 0) = RP(0, 9 * pIdx + 3 * axis_id + 1);
				RPblock(1, 1) = RP(1, 9 * pIdx + 3 * axis_id + 1);
				RPblock(1, 2) = RP(2, 9 * pIdx + 3 * axis_id + 1);
				RPblock(2, 0) = RP(0, 9 * pIdx + 3 * axis_id + 2);
				RPblock(2, 1) = RP(1, 9 * pIdx + 3 * axis_id + 2);
				RPblock(2, 2) = RP(2, 9 * pIdx + 3 * axis_id + 2);

				LPblock(0, 0) = LP(3 * pIdx, 3 * jIdx);
				LPblock(0, 1) = LP(3 * pIdx + 1, 3 * jIdx);
				LPblock(0, 2) = LP(3 * pIdx + 2, 3 * jIdx);
				LPblock(1, 0) = LP(3 * pIdx, 3 * jIdx + 1);
				LPblock(1, 1) = LP(3 * pIdx + 1, 3 * jIdx + 1);
				LPblock(1, 2) = LP(3 * pIdx + 2, 3 * jIdx + 1);
				LPblock(2, 0) = LP(3 * pIdx, 3 * jIdx + 2);
				LPblock(2, 1) = LP(3 * pIdx + 1, 3 * jIdx + 2);
				LPblock(2, 2) = LP(3 * pIdx + 2, 3 * jIdx + 2);
				dR1 = RPblock * LPblock;
				Tmp.col(axis_id) = dR1 * (v0 - j0) * w;
			}
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					J[3 * (3 + 3 * pIdx) + 3 * i + j] += Tmp(j, i);
				}
			}
		}
	}

	// reuse Tmp matrix as D. 
	Tmp.setZero();
	Tmp(0, 0) = 1 / point2d(2);
	Tmp(1, 1) = 1 / point2d(2);
	Tmp(0, 2) = -point2d(0) / (point2d(2)*point2d(2));
	Tmp(1, 2) = -point2d(1) / (point2d(2) * point2d(2));
	// reuse v0 as dp 
	// reuse j0 as dpsil
	int J_V_index = -1;
	float A_col[3 * NUM_POSE_JOINT + 3];
#pragma unroll
	for (int i = 0; i < paramNum; i++)
	{
		J_V_index = d_paramLines[i];
		v0(0) = J[3 * J_V_index];
		v0(1) = J[3 * J_V_index + 1];
		v0(2) = J[3 * J_V_index + 2];

		j0 = Tmp * K*R* v0;
		A_col[i] = (j0(0) * dx + j0(1) * dy) * 0.01;
	}
	float b = (rend_sdf_value - det_sdf_value) * 0.01;

#ifdef DEBUG_SOLVER
	//for (int i = 0; i < paramNum; i++)d_AT(i, vIdx) = A_col[i];
	//d_b[vIdx] = b;
#else 
#pragma unroll
	for (int i = 0; i < 3 * NUM_POSE_JOINT + 3; i++)
	{
		for (int j = 0; j < 3 * NUM_POSE_JOINT + 3; j++)
		{
			atomicAdd(&d_ATA_sil(j, i), A_col[i] * A_col[j]);
		}
		atomicAdd(&d_ATb_sil[i], A_col[i] * b);
	}
#endif
}


void PigSolverDevice::calcSilhouetteJacobi_device(
	Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Vector3f T,
	float* d_depth, float* d_depth_interact, int idcode, int paramNum, int view
)
{
	dim3 blocksize(32);
	dim3 gridsize(pcl::device::divUp(m_vertexNum, blocksize.x));

	int camid = m_viewids[view];

	construct_sil_A_kernel2 << <gridsize, blocksize >> > (
		m_device_verticesDeformed, // vertices of tpose
		m_device_jointsDeformed, // joints of tpoose 
		d_J_joint_full, // jacobi of joints
		m_device_lbsweights, // skinning_weights
		m_device_parents,
		m_vertexNum,
		m_jointNum,
		d_RP,
		d_LP,
		m_device_paramLines, 
		m_device_bodyParts, m_device_verticesPosed, K, R, T,
		d_depth, d_depth_interact, d_det_mask[view], d_const_scene_mask[camid], d_const_distort_mask,
		d_det_sdf[view], d_det_gradx[view], d_det_grady[view], d_rend_sdf,
		1920, 1080, m_vertexNum, paramNum, idcode, 
#ifdef DEBUG_SOLVER
		d_AT_sil, d_b_sil
#else 
		d_ATA_sil, d_ATb_sil
#endif 
		
		);
	cudaSafeCall(cudaGetLastError()); 
	cudaSafeCall(cudaDeviceSynchronize()); 
}

void PigSolverDevice::calcPoseJacobiFullTheta_V_device(
	pcl::gpu::DeviceArray2D<float> J_vert,
	pcl::gpu::DeviceArray2D<float> J_joint, 
	pcl::gpu::DeviceArray2D<float> d_RP, 
	pcl::gpu::DeviceArray2D<float> d_LP
)
{
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


