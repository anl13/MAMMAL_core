#include "pigsolver.h"

void PigSolver::CalcSkelJacobiPartThetaByMapper(Eigen::MatrixXf& J)
{
	/*
	AN Liang. 20191227 comments. This function was written by ZHANG Yuxiang.
	y: 3*N vector(N is joint num)
	x: 3+3*N vector(freedom variable, global t, and r for each joint (lie algebra)
	dy/dx^T: Jacobi matrix J[3*N, 3+3*N]
	below function update this matrix in colwise manner, in articulated tree structure.
	*/
	int N = m_topo.joint_num;
	int M = m_poseToOptimize.size();
	J = Eigen::MatrixXf::Zero(3 * N, 3 + 3 * M);

	// calculate delta rodrigues
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * m_jointNum);
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3f& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
		rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiF(pose);
	}

	//dJ_dtheta
	Eigen::MatrixXf m_jointJacobiPose = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
	for (int jointDerivativeId = 0; jointDerivativeId < m_jointNum; jointDerivativeId++)
	{
		// update translation term
		m_jointJacobiPose.block<3, 3>(jointDerivativeId * 3, 0).setIdentity();

		// update poseParam term
		for (int axisDerivativeId = 0; axisDerivativeId < 3; axisDerivativeId++)
		{
			std::vector<std::pair<bool, Eigen::Matrix4f>> globalAffineDerivative(m_jointNum, std::make_pair(false, Eigen::Matrix4f::Zero()));
			globalAffineDerivative[jointDerivativeId].first = true;
			auto& affine = globalAffineDerivative[jointDerivativeId].second;
			affine.block<3, 3>(0, 0) = rodriguesDerivative.block<3, 3>(0, 3 * (3 * jointDerivativeId + axisDerivativeId));
			affine = jointDerivativeId == 0 ? affine : (m_globalAffine.block<4, 4>(0, 4 * m_parent(jointDerivativeId)) * affine);

			for (int jointId = jointDerivativeId + 1; jointId < m_jointNum; jointId++)
			{
				if (globalAffineDerivative[m_parent(jointId)].first)
				{
					globalAffineDerivative[jointId].first = true;
					globalAffineDerivative[jointId].second = globalAffineDerivative[m_parent(jointId)].second * m_singleAffine.block<4, 4>(0, 4 * jointId);
					// update jacobi for pose
					m_jointJacobiPose.block<3, 1>(jointId * 3, 3 + jointDerivativeId * 3 + axisDerivativeId) = globalAffineDerivative[jointId].second.block<3, 1>(0, 3);
				}
			}
		}
	}
	for (int i = 0; i < N; i++)
	{
		if (m_mapper[i].first != 0) continue;
		int jIdx = m_mapper[i].second;
		J.block<3, 3>(3 * i, 0) = m_jointJacobiPose.block<3, 3>(3 * jIdx, 0);
		for (int k = 0; k < M; k++)
		{
			int thetaIdx = m_poseToOptimize[k];
			J.block<3, 3>(3 * i, 3 + 3 * k) = m_jointJacobiPose.block<3, 3>(3 * jIdx, 3 + 3 * thetaIdx);
		}
	}

	// this version was deduced by AN Liang, 20191231
	// assume that you have computed Pose Jacobi
	// O(n^2)
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
				dR = m_globalAffine.block<3, 3>(0, 4 * m_parent(jIdx)) * dR;
			}
			RP.block<3, 3>(9 * jIdx + 3 * aIdx, 0) = dR;
		}
		LP.block<3, 3>(3 * jIdx, 3 * jIdx) = Eigen::Matrix3f::Identity();
		for (int child = jIdx + 1; child < m_jointNum; child++)
		{
			int father = m_parent(child);
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx) * m_singleAffine.block<3, 3>(0, 4 * child);
		}
	}

	Eigen::MatrixXf m_vertJacobiPose = Eigen::MatrixXf::Zero(3, 3 + 3 * m_jointNum);
	for (int i = 0; i<N; i++)
	{
		if (m_mapper[i].first != 1) continue;
		int vIdx = m_mapper[i].second;
		Eigen::Vector3f v0 = m_verticesDeformed.col(vIdx);
		m_vertJacobiPose.setZero();
		for (int jIdx = 0; jIdx<m_jointNum; jIdx++)
		{
			if (m_lbsweights(jIdx, vIdx) < 0.00001) continue;
			Eigen::Vector3f j0 = m_jointsDeformed.col(jIdx);
			m_vertJacobiPose += (m_lbsweights(jIdx, vIdx) * m_jointJacobiPose.middleRows(3 * jIdx, 3));
			for (int pIdx = jIdx; pIdx>-1; pIdx = m_parent(pIdx)) // poseParamIdx to derive
			{
				Eigen::Matrix3f T = Eigen::Matrix3f::Zero();
				for (int axis_id = 0; axis_id<3; axis_id++)
				{
					Eigen::Matrix3f dR1 = RP.block<3, 3>(9 * pIdx + 3 * axis_id, 0) * LP.block<3, 3>(3 * jIdx, 3 * pIdx);
					T.col(axis_id) = dR1 * (v0 - j0) * m_lbsweights(jIdx, vIdx);
				}
				m_vertJacobiPose.block<3, 3>(0, 3 + 3 * pIdx) += T;
			}
		}
		J.block<3, 3>(3 * i, 0) = m_vertJacobiPose.block<3, 3>(0, 0);
		for (int k = 0; k < M; k++)
		{
			int thetaIdx = m_poseToOptimize[k];
			J.block<3, 3>(3 * i, 3 + 3 * k) = m_vertJacobiPose.block<3, 3>(0, 3 + 3 * thetaIdx);
		}
	}
}

void PigSolver::CalcSkelJacobiPartThetaByPairs(Eigen::MatrixXf& J)
{
	Eigen::MatrixXf J_jointfull, J_joint;
	int N = m_optimPairs.size();
	int M = m_poseToOptimize.size();
	J = Eigen::MatrixXf::Zero(3 * N, 3 + 3 * M);
	
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * m_jointNum);
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3f& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
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
				dR = m_globalAffine.block<3, 3>(0, 4 * m_parent(jIdx)) * dR;
			}
			RP.block<3, 3>(9 * jIdx + 3 * aIdx, 0) = dR;
		}
		LP.block<3, 3>(3 * jIdx, 3 * jIdx) = Eigen::Matrix3f::Identity();
		for (int child = jIdx + 1; child < m_jointNum; child++)
		{
			int father = m_parent(child);
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx) * m_singleAffine.block<3, 3>(0, 4 * child);
		}
	}

	//dJ_dtheta
	J_jointfull = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
	for (int jointDerivativeId = 0; jointDerivativeId < m_jointNum; jointDerivativeId++)
	{
		// update translation term
		J_jointfull.block<3, 3>(jointDerivativeId * 3, 0).setIdentity();

		// update poseParam term
		for (int axisDerivativeId = 0; axisDerivativeId < 3; axisDerivativeId++)
		{
			std::vector<std::pair<bool, Eigen::Matrix4f>> globalAffineDerivative(m_jointNum, std::make_pair(false, Eigen::Matrix4f::Zero()));
			globalAffineDerivative[jointDerivativeId].first = true;
			auto& affine = globalAffineDerivative[jointDerivativeId].second;
			affine.block<3, 3>(0, 0) = rodriguesDerivative.block<3, 3>(0, 3 * (3 * jointDerivativeId + axisDerivativeId));
			affine = jointDerivativeId == 0 ? affine : (m_globalAffine.block<4, 4>(0, 4 * m_parent(jointDerivativeId)) * affine);

			for (int jointId = jointDerivativeId + 1; jointId < m_jointNum; jointId++)
			{
				if (globalAffineDerivative[m_parent(jointId)].first)
				{
					globalAffineDerivative[jointId].first = true;
					globalAffineDerivative[jointId].second = globalAffineDerivative[m_parent(jointId)].second * m_singleAffine.block<4, 4>(0, 4 * jointId);
					// update jacobi for pose
					J_jointfull.block<3, 1>(jointId * 3, 3 + jointDerivativeId * 3 + axisDerivativeId) = globalAffineDerivative[jointId].second.block<3, 1>(0, 3);
				}
			}
		}
	}
	J_joint.resize(3 * m_jointNum, 3 + 3 * M);
	for (int i = 0; i < m_jointNum; i++)
	{
		J_joint.block<3, 3>(3 * i, 0) = J_jointfull.block<3, 3>(3 * i, 0);
		for (int k = 0; k < M; k++)
		{
			int thetaidx = m_poseToOptimize[k];
			J_joint.block<3, 3>(3 * i, 3 + 3 * k) = J_jointfull.block<3, 3>(3 * i, 3 + 3 * thetaidx);
		}
	}

	Eigen::MatrixXf block = Eigen::MatrixXf::Zero(3, 3 + 3 * m_jointNum);
	for (int i = 0; i < N; i++)
	{
		CorrPair P = m_optimPairs[i];

		if (P.type == 0)
		{
			J.middleRows(3 * i, 3) =
				J_joint.middleRows(3 * P.index, 3);
			continue;
		}
		if(P.type == 1)
		{
			block.setZero();
			int vIdx = P.index; 
			Eigen::Vector3f v0 = m_verticesDeformed.col(vIdx);
			block.setZero();
			for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
			{
				if (m_lbsweights(jIdx, vIdx) < 0.00001) continue;
				Eigen::Vector3f j0 = m_jointsDeformed.col(jIdx);
				block += (m_lbsweights(jIdx, vIdx) * J_jointfull.middleRows(3 * jIdx, 3));
				for (int pIdx = jIdx; pIdx > -1; pIdx = m_parent(pIdx)) // poseParamIdx to derive
				{
					Eigen::Matrix3f T = Eigen::Matrix3f::Zero();
					for (int axis_id = 0; axis_id < 3; axis_id++)
					{
						Eigen::Matrix3f dR1 = RP.block<3, 3>(9 * pIdx + 3 * axis_id, 0) * LP.block<3, 3>(3 * jIdx, 3 * pIdx);
						T.col(axis_id) = dR1 * (v0 - j0) * m_lbsweights(jIdx, vIdx);
					}
					block.block<3, 3>(0, 3 + 3 * pIdx) += T;
				}
			}
			J.block<3, 3>(3 * i, 0) = block.block<3, 3>(0, 0);
			for (int k = 0; k < M; k++)
			{
				int thetaidx = m_poseToOptimize[k];
				J.block<3, 3>(3 * i, 3 + 3 * k) = block.block<3, 3>(0, 3 + 3 * thetaidx);
			}
		}
	}
}

void PigSolver::CalcPose2DTermByMapper(
	const int k,
	const Eigen::MatrixXf& skel,
	const Eigen::MatrixXf& Jacobi3d,
	Eigen::MatrixXf& ATA,
	Eigen::VectorXf& ATb
)
{
	int view = m_source.view_ids[k];
	Camera cam = m_cameras[view];
	const DetInstance& det = m_source.dets[k];
	Eigen::Matrix3f R = cam.R;
	Eigen::Matrix3f K = cam.K;
	Eigen::Vector3f T = cam.T;

	int N = m_topo.joint_num;
	int M = m_poseToOptimize.size();
	Eigen::MatrixXf J = Eigen::MatrixXf::Zero(2 * N, 3 + 3 * M);
	Eigen::VectorXf r = Eigen::VectorXf::Zero(2 * N);

	for (int i = 0; i < N; i++)
	{
		Eigen::Vector3f x_local = K * (R * skel.col(i) + T);
		Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
		D(0, 0) = 1 / x_local(2);
		D(1, 1) = 1 / x_local(2);
		D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
		D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));
		J.middleRows(2 * i, 2) = D * K * R * Jacobi3d.middleRows(3 * i, 3);
		Eigen::Vector2f u;
		u(0) = x_local(0) / x_local(2);
		u(1) = x_local(1) / x_local(2);

		if (m_mapper[i].first < 0) continue;
		if (det.keypoints[i](2) < m_topo.kpt_conf_thresh[i]) continue;
		r.segment<2>(2 * i) = u - det.keypoints[i].segment<2>(0);
	}
	// std::cout << "K: " << k << std::endl; 
	// std::cout << "J: " << std::endl << J.middleCols(3,6) << std::endl; 
	ATb = -J.transpose() * r;
	ATA = J.transpose() * J;
}


void PigSolver::CalcPose2DTermByPairs(
	const int viewindex,
	const Eigen::MatrixXf& skel2d,
	const Eigen::MatrixXf& Jacobi3d,
	Eigen::MatrixXf& ATA,
	Eigen::VectorXf& ATb
)
{
	int view = m_source.view_ids[viewindex];
	Camera cam = m_cameras[view];
	const DetInstance& det = m_source.dets[viewindex];
	Eigen::Matrix3f R = cam.R;
	Eigen::Matrix3f K = cam.K;
	Eigen::Vector3f T = cam.T;

	int N = m_optimPairs.size();
	int M = m_poseToOptimize.size();
	Eigen::VectorXf r = Eigen::VectorXf::Zero(2 * N);
	Eigen::MatrixXf J; 
	if (m_isLatent)
	{
		J = Eigen::MatrixXf::Zero(2 * N, 38); 
	}
	else
	{
		J = Eigen::MatrixXf::Zero(2 * N, 3 + 3 * M);
	}

	for (int i = 0; i < N; i++)
	{
		CorrPair P = m_optimPairs[i];
		int t = P.target; 
		if (det.keypoints[t](2) < m_topo.kpt_conf_thresh[t]) continue;
		
		Eigen::Vector3f x_local = K * (R * skel2d.col(t) + T);
		Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
		D(0, 0) = 1 / x_local(2);
		D(1, 1) = 1 / x_local(2);
		D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
		D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));
		J.middleRows(2 * i, 2) = P.weight * D * K * R * Jacobi3d.middleRows(3 * i, 3);
		Eigen::Vector2f u;
		u(0) = x_local(0) / x_local(2);
		u(1) = x_local(1) / x_local(2);

		r.segment<2>(2 * i) = P.weight * (u - det.keypoints[t].segment<2>(0));
	}

	ATb = -J.transpose() * r;
	ATA = J.transpose() * J;

	//std::cout << "pose r: " << r.norm() << std::endl; 
}

void PigSolver::CalcPoseJacobiNumeric()
{	/*
	auto m_jointJacobiPoseNumeric = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
	auto m_vertJacobiPoseNumeric = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>::Zero(3 * m_vertexNum, 3 + 3 * m_jointNum);

	UpdateVertices();
	Eigen::MatrixXd previousJ = m_jointsFinal;
	Eigen::MatrixXd previousV = m_verticesFinal;
	Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(3 * m_jointNum);
	float alpha = 0.000001;
	float inv_alpha = 1.0 / alpha;
	for (int i = 0; i < 3; i++)
	{
		Eigen::Vector3d delta_t = Eigen::Vector3d::Zero();
		delta_t(i) = alpha;
		m_translation += delta_t;
		UpdateVertices();
		Eigen::MatrixXd delta_j = m_jointsFinal - previousJ;
		m_jointJacobiPoseNumeric.col(i) = Eigen::Map<Eigen::VectorXd>(delta_j.data(), 3 * m_jointNum) * inv_alpha;
		Eigen::MatrixXd delta_v = m_verticesFinal - previousV;
		m_vertJacobiPoseNumeric.col(i) = Eigen::Map<Eigen::VectorXd>(delta_v.data(), 3 * m_vertexNum) * inv_alpha;
		m_translation -= delta_t;
	}
	for (int i = 0; i < 3 * m_jointNum; i++)
	{
		delta_x.setZero();
		delta_x(i) = alpha;
		m_poseParam += delta_x;
		UpdateVertices();
		Eigen::MatrixXd delta_j = m_jointsFinal - previousJ;
		m_jointJacobiPoseNumeric.col(3 + i) = Eigen::Map<Eigen::VectorXd>(delta_j.data(), 3 * m_jointNum) * inv_alpha;
		Eigen::MatrixXd delta_v = m_verticesFinal - previousV;
		m_vertJacobiPoseNumeric.col(3 + i) = Eigen::Map<Eigen::VectorXd>(delta_v.data(), 3 * m_vertexNum) * inv_alpha;
		m_poseParam -= delta_x;
	}
	*/
}


void PigSolver::CalcShapeJacobiNumeric()
{
	/*
	auto m_jointJacobiShapeNumeric = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, m_shapeNum);
	auto m_vertJacobiShapeNumeric = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>::Zero(3 * m_vertexNum, m_shapeNum);
	UpdateVertices();
	Eigen::MatrixXd previousJ = m_jointsFinal;
	Eigen::MatrixXd previousV = m_verticesFinal;
	Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(m_shapeNum);
	float alpha = 0.0001;
	float inv_alpha = 1.0 / alpha;
	for (int i = 0; i < m_shapeNum; i++)
	{
		delta_x.setZero();
		delta_x(i) = alpha;
		m_shapeParam += delta_x;
		UpdateVertices();
		Eigen::MatrixXd delta_j = m_jointsFinal - previousJ;
		m_jointJacobiShapeNumeric.col(i) = Eigen::Map<Eigen::VectorXd>(delta_j.data(), 3 * m_jointNum) * inv_alpha;
		Eigen::MatrixXd delta_v = m_verticesFinal - previousV;
		m_vertJacobiShapeNumeric.col(i) = Eigen::Map<Eigen::VectorXd>(delta_v.data(), 3 * m_vertexNum) * inv_alpha;
		m_shapeParam -= delta_x;
	}
	*/
}

void PigSolver::CalcShapeJacobi(
Eigen::MatrixXf& jointJacobiShape, 
Eigen::MatrixXf& vertJacobiShape
)
{ // this function c2338 error 20200430
	/*
	// compute d_joint_d_beta
	jointJacobiShape.resize(3 * m_jointNum, m_shapeNum);
	jointJacobiShape.setZero();
	for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
	{
		if (jIdx == 0)
			jointJacobiShape.middleRows(3 * jIdx, 3) += m_shapeBlendJ.middleRows(3 * jIdx, 3);
		else
		{
			int pIdx = m_parent(jIdx);
			jointJacobiShape.middleRows(3 * jIdx, 3) = jointJacobiShape.middleRows(3 * pIdx, 3) + 
				m_globalAffine.block<3, 3>(0, 4 * pIdx)
				* (m_shapeBlendJ.middleRows(3 * jIdx, 3) - m_shapeBlendJ.middleRows(3 * pIdx, 3));
		}
	}

	// compute d_v_d_beta
	vertJacobiShape.resize(3 * m_vertexNum, m_shapeNum);
	vertJacobiShape.setZero();
	for (int vIdx = 0; vIdx < m_vertexNum; vIdx++)
	{
		for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
		{
			if (m_lbsweights(jIdx, vIdx) < 0.00001) continue;
			vertJacobiShape.middleRows(3 * vIdx, 3) += ((jointJacobiShape.middleRows(3 * jIdx, 3)
				+ m_globalAffine.block<3, 3>(0, 4 * jIdx) * (m_shapeBlendV.middleRows(3 * vIdx, 3) - m_shapeBlendJ.middleRows(3 * jIdx, 3))
				) * m_lbsweights(jIdx, vIdx));
		}
	}
	*/
}

void PigSolver::CalcPoseJacobiFullTheta(Eigen::MatrixXf& jointJacobiPose, Eigen::MatrixXf& J_vert, 
	bool with_vert)
{
	// calculate delta rodrigues
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * m_jointNum);
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3f& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
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
				dR = m_globalAffine.block<3, 3>(0, 4 * m_parent(jIdx)) * dR;
			}
			RP.block<3, 3>(9 * jIdx + 3 * aIdx, 0) = dR;
		}
		LP.block<3, 3>(3 * jIdx, 3 * jIdx) = Eigen::Matrix3f::Identity();
		for (int child = jIdx + 1; child < m_jointNum; child++)
		{
			int father = m_parent(child);
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx) * m_singleAffine.block<3, 3>(0, 4 * child);
		}
	}

	//dJ_dtheta
	jointJacobiPose = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
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
			affine = jointDerivativeId == 0 ? affine : (m_globalAffine.block<4, 4>(0, 4 * m_parent(jointDerivativeId)) * affine);

			for (int jointId = jointDerivativeId + 1; jointId < m_jointNum; jointId++)
			{
				if (globalAffineDerivative[m_parent(jointId)].first)
				{
					globalAffineDerivative[jointId].first = true;
					globalAffineDerivative[jointId].second = globalAffineDerivative[m_parent(jointId)].second * m_singleAffine.block<4, 4>(0, 4 * jointId);
					// update jacobi for pose
					jointJacobiPose.block<3, 1>(jointId * 3, 3 + jointDerivativeId * 3 + axisDerivativeId) = globalAffineDerivative[jointId].second.block<3, 1>(0, 3);
				}
			}
		}
	}

	if (!with_vert) return; 
	J_vert = Eigen::MatrixXf::Zero(3 * m_vertexNum, 3 + 3 * m_jointNum);
	Eigen::MatrixXf block = Eigen::MatrixXf::Zero(3, 3 + 3 * m_jointNum);
	for (int vIdx = 0; vIdx < m_vertexNum; vIdx++)
	{
		Eigen::Vector3f v0 = m_verticesDeformed.col(vIdx);
		block.setZero();
		for (int jIdx = 0; jIdx<m_jointNum; jIdx++)
		{
			if (m_lbsweights(jIdx, vIdx) < 0.00001) continue;
			block += (m_lbsweights(jIdx, vIdx) * jointJacobiPose.middleRows(3 * jIdx, 3));
			Eigen::Vector3f j0 = m_jointsDeformed.col(jIdx);
			for (int pIdx = jIdx; pIdx>-1; pIdx = m_parent(pIdx)) // poseParamIdx to derive
			{
				Eigen::Matrix3f T = Eigen::Matrix3f::Zero();
				for (int axis_id = 0; axis_id<3; axis_id++)
				{
					Eigen::Matrix3f dR1 = RP.block<3, 3>(9 * pIdx + 3 * axis_id, 0) * LP.block<3, 3>(3 * jIdx, 3 * pIdx);
					T.col(axis_id) = dR1 * (v0 - j0) * m_lbsweights(jIdx, vIdx);
				}
				
				block.block<3, 3>(0, 3 + 3 * pIdx) += T;
			}
		}
		J_vert.middleRows(3 * vIdx, 3) = block;
	}
}

void PigSolver::CalcPoseJacobiPartTheta(Eigen::MatrixXf& J_joint, Eigen::MatrixXf& J_vert, 
	bool with_vert)
{
	Eigen::MatrixXf J_jointfull, J_vertfull;
	CalcPoseJacobiFullTheta(J_jointfull, J_vertfull, with_vert);
	int M = m_poseToOptimize.size();
	J_joint.resize(3 * m_jointNum, 3 + 3 * M);
	if (with_vert)
	{
		J_vert.resize(3 * m_vertexNum, 3 + 3 * M);
		J_vert.middleCols(0, 3) = J_vertfull.middleCols(0, 3);
	}
	
	J_joint.middleCols(0, 3) = J_jointfull.middleCols(0, 3);
	
	for (int i = 0; i < M; i++)
	{
		int thetaid = m_poseToOptimize[i];
		J_joint.middleCols(3 + 3 * i, 3) = J_jointfull.middleCols(3 + 3 * thetaid, 3);
		if(with_vert)
			J_vert.middleCols(3 + 3 * i, 3) = J_vertfull.middleCols(3 + 3 * thetaid, 3);
	}
}

void PigSolver::Calc2DJacobiNumeric(
	const int k, const Eigen::MatrixXf& skel,
	Eigen::MatrixXf& H, Eigen::VectorXf& b)
{
	/*
	int view = m_source.view_ids[k];
	Camera cam = m_cameras[view];
	const DetInstance& det = m_source.dets[k];
	Eigen::Matrix3d R = cam.R;
	Eigen::Matrix3d K = cam.K;
	Eigen::Vector3d T = cam.T;
	int N = m_topo.joint_num;
	Eigen::VectorXd target_vec = Eigen::VectorXd::Zero(2 * N);
	for (int i = 0; i < N; i++)
	{
		if (det.keypoints[i](2) < m_topo.kpt_conf_thresh[i]) continue;
		target_vec.segment<2>(2 * i) = det.keypoints[i].segment<2>(0);
	}

	int M = m_poseToOptimize.size();
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2 * N, 3 + 3 * M);
	float alpha = 0.0001;
	float inv_alpha = 1 / alpha;
	Eigen::VectorXd previous_proj = getRegressedSkelProj(K, R, T);
	Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(3 * m_jointNum);

	for (int i = 0; i < 3; i++)
	{
		Eigen::Vector3d delta_t = Eigen::Vector3d::Zero();
		delta_t(i) = alpha;
		m_translation += delta_t;
		UpdateVertices();
		Eigen::VectorXd proj = getRegressedSkelProj(K, R, T);
		J.col(i) = (proj - previous_proj) * inv_alpha;
		m_translation -= delta_t;
	}
	for (int i = 0; i < 3 * M; i++)
	{
		delta_x.setZero();
		int a = i / 3;
		int b = i % 3;
		int pose_id = m_poseToOptimize[a];
		delta_x(3 * pose_id + b) = alpha;
		m_poseParam += delta_x;
		UpdateVertices();
		Eigen::VectorXd proj = getRegressedSkelProj(K, R, T);
		J.col(3 + i) = (proj - previous_proj) * inv_alpha;
		m_poseParam -= delta_x;
	}
	std::cout << "K: " << k << std::endl;
	std::cout << "numeric J: " << std::endl << J.middleCols(3, 6);

	Eigen::VectorXd r = previous_proj - target_vec;
	for (int i = 0; i < N; i++)
	{
		if (det.keypoints[i](2) < m_topo.kpt_conf_thresh[i])
			r.segment<2>(2 * i) = Eigen::Vector2d::Zero();
	}
	b = -J.transpose() * r;
	H = J.transpose() * J;
	*/
}


void PigSolver::CalcShapeJacobiToSkel(Eigen::MatrixXf& J)
{
	/*
	Eigen::MatrixXd jointJacobiShape, vertJacobiShape;
	CalcShapeJacobi(jointJacobiShape, vertJacobiShape); 
	int N = m_topo.joint_num;
	J = Eigen::MatrixXd::Zero(3 * N, m_shapeNum);
	for (int i = 0; i < N; i++)
	{
		if (m_mapper[i].first < 0)continue;
		if (m_mapper[i].first == 0)
		{
			int jid = m_mapper[i].second;
			J.middleRows(3 * i, 3) = jointJacobiShape.middleRows(3 * jid, 3);
		}
		else
		{
			int vid = m_mapper[i].second;
			J.middleRows(3 * i, 3) = vertJacobiShape.middleRows(3 * vid, 3);
		}
	}
	*/
}

void PigSolver::CalcPoseJacobiLatent(
	Eigen::MatrixXf& J_joint,
	Eigen::MatrixXf& J_vert,
	bool is_joint_only
)
{
	m_decoder.computeJacobi();
	Eigen::MatrixXf& RJ = m_decoder.J; // [(62*9) * 32]
	// calculate delta rodrigues (global rotation)
	Eigen::Matrix<float, 3, 9, Eigen::ColMajor> globalRJ = RodriguesJacobiF(m_poseParam.segment<3>(0));

	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> LP(3 * m_jointNum, 3 * m_jointNum);
	LP.setZero();
	for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
	{
		LP.block<3, 3>(3 * jIdx, 3 * jIdx) = Eigen::Matrix3f::Identity();
		for (int child = jIdx + 1; child < m_jointNum; child++)
		{
			int father = m_parent(child);
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx) * m_singleAffine.block<3, 3>(0, 4 * child);
		}
	}

	// rotation 
	/// dQ 
	/// 20200801: dQ pass numeric comparison 
	Eigen::MatrixXf dQ = Eigen::MatrixXf::Zero(3 * m_jointNum, 3 * 35);

	for (int i = 0; i < m_jointNum; i++)
	{
		for (int paramId = 0; paramId < 35; paramId++)
		{
			Eigen::Matrix3f dR;
			if (i == 0)
			{
				if (paramId < 3) dR = globalRJ.block<3, 3>(0, 3 * paramId);
				else dR = Eigen::Matrix3f::Zero();
			}
			else
			{
				if (paramId < 3) dR = Eigen::Matrix3f::Zero(); 
				else 
				    dR = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(RJ.block<9, 1>(9 * i, paramId - 3).data());
			}

			if (i == 0)
			{
				dQ.block<3, 3>(3 * i, 3 * paramId) = dR;
			}
			else
			{
				int p = m_parent(i);
				dQ.block<3, 3>(3 * i, 3 * paramId) = dQ.block<3, 3>(3 * p, 3 * paramId) * m_singleAffine.block<3, 3>(0, 4 * i)
					+ m_globalAffine.block<3, 3>(0, 4 * p)*dR;
			}
		}
	}

	/// 20200801 16:30 J_joint pass numeeric test 
	J_joint = Eigen::MatrixXf::Zero(3 * m_jointNum, 32 + 6);
	for (int jid = 0; jid < m_jointNum; jid++)
	{
		J_joint.block<3, 3>(3 * jid, 0) = Eigen::Matrix3f::Identity(); 
		for (int paramId = 0; paramId < 35; paramId++)
		{
			if (jid > 0)
			{
				int p = m_parent(jid);
				J_joint.block<3, 1>(3 * jid, paramId+3) = dQ.block<3, 3>(3 * p, 3 * paramId) * m_singleAffine.block<3, 1>(0, 4 * jid + 3)
					+ J_joint.block<3, 1>(3 * p, paramId+3);
			}
		}
	}

	if (is_joint_only) return; 
	/// 20200801 16:33 J_vert pass numeric test 
	J_vert = Eigen::MatrixXf::Zero(3 * m_vertexNum, 32 + 6);
	for (int vid = 0; vid < m_vertexNum; vid++)
	{
		for (int jid = 0; jid < m_jointNum; jid++)
		{
			if (m_lbsweights(jid, vid) < 1e-6) continue;
			else
			{
				for (int i = 0; i < 35; i++)
				{
					J_vert.block<3, 1>(3 * vid, i + 3) += m_lbsweights(jid, vid) *
						(
							dQ.block<3, 3>(3 * jid, 3 * i) * (m_verticesDeformed.col(vid) - m_jointsDeformed.col(jid))
							);
				}
				J_vert.middleRows(3 * vid, 3) += m_lbsweights(jid, vid) *
					J_joint.middleRows(3 * jid, 3);
			}
		}
	}
}

void PigSolver::CalcSkelJacobiByPairsLatent(Eigen::MatrixXf& J)
{
	m_decoder.computeJacobi();
	Eigen::MatrixXf& RJ = m_decoder.J; // [(62*9) * 32]
									   // calculate delta rodrigues (global rotation)
	Eigen::Matrix<float, 3, 9, Eigen::ColMajor> globalRJ = RodriguesJacobiF(m_poseParam.segment<3>(0));

	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> LP(3 * m_jointNum, 3 * m_jointNum);
	LP.setZero();
	for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
	{
		LP.block<3, 3>(3 * jIdx, 3 * jIdx) = Eigen::Matrix3f::Identity();
		for (int child = jIdx + 1; child < m_jointNum; child++)
		{
			int father = m_parent(child);
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx) * m_singleAffine.block<3, 3>(0, 4 * child);
		}
	}

	// rotation 
	/// dQ 
	/// 20200801: dQ pass numeric comparison 
	Eigen::MatrixXf dQ = Eigen::MatrixXf::Zero(3 * m_jointNum, 3 * 35);

	for (int i = 0; i < m_jointNum; i++)
	{
		for (int paramId = 0; paramId < 35; paramId++)
		{
			Eigen::Matrix3f dR;
			if (i == 0)
			{
				if (paramId < 3) dR = globalRJ.block<3, 3>(0, 3 * paramId);
				else dR = Eigen::Matrix3f::Zero();
			}
			else
			{
				if (paramId < 3) dR = Eigen::Matrix3f::Zero();
				else
					dR = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(RJ.block<9, 1>(9 * i, paramId - 3).data());
			}

			if (i == 0)
			{
				dQ.block<3, 3>(3 * i, 3 * paramId) = dR;
			}
			else
			{
				int p = m_parent(i);
				dQ.block<3, 3>(3 * i, 3 * paramId) = dQ.block<3, 3>(3 * p, 3 * paramId) * m_singleAffine.block<3, 3>(0, 4 * i)
					+ m_globalAffine.block<3, 3>(0, 4 * p)*dR;
			}
		}
	}

	/// 20200801 16:30 J_joint pass numeeric test 
	Eigen::MatrixXf J_joint = Eigen::MatrixXf::Zero(3 * m_jointNum, 32 + 6);
	for (int jid = 0; jid < m_jointNum; jid++)
	{
		J_joint.block<3, 3>(3 * jid, 0) = Eigen::Matrix3f::Identity();
		for (int paramId = 0; paramId < 35; paramId++)
		{
			if (jid > 0)
			{
				int p = m_parent(jid);
				J_joint.block<3, 1>(3 * jid, paramId + 3) = dQ.block<3, 3>(3 * p, 3 * paramId) * m_singleAffine.block<3, 1>(0, 4 * jid + 3)
					+ J_joint.block<3, 1>(3 * p, paramId + 3);
			}
		}
	}

	int N = m_optimPairs.size(); 
	J.resize(3 * N, 38); 
	J.setZero();

	for (int i = 0; i < N; i++)
	{
		CorrPair P = m_optimPairs[i];

		if (P.type == 0)
		{
			J.middleRows(3 * i, 3) =
				J_joint.middleRows(3 * P.index, 3);
			continue;
		}
		else if(P.type==1)
		{
			int vid = P.index;
			Eigen::MatrixXf block = Eigen::MatrixXf::Zero(3, 38); 
			for (int jid = 0; jid < m_jointNum; jid++)
			{
				if (m_lbsweights(jid, vid) < 1e-6) continue;
				else
				{
					for (int i = 0; i < 35; i++)
					{
						block.col(i + 3) += m_lbsweights(jid, vid) *
							(
								dQ.block<3, 3>(3 * jid, 3 * i) * (m_verticesDeformed.col(vid) - m_jointsDeformed.col(jid))
								);
					}
					block += m_lbsweights(jid, vid) *
						J_joint.middleRows(3 * jid, 3);
				}
			}
			J.middleRows(3 * i, 3) = block; 
		}
	}
}

void PigSolver::debug_numericJacobiLatent()
{
	Eigen::MatrixXf J_joint_numeric = Eigen::MatrixXf::Zero(3 * m_jointNum, 38);
	Eigen::MatrixXf J_vert_numeric = Eigen::MatrixXf::Zero(3 * m_vertexNum, 38); 
	Eigen::MatrixXf dQ_numeric = Eigen::MatrixXf::Zero(3 * m_jointNum, 3 * 35);
	Eigen::MatrixXf V0 = m_verticesFinal; 
	Eigen::MatrixXf J0 = m_jointsFinal;
	Eigen::MatrixXf Q0 = m_globalAffine;
	float alpha = 0.00001;
	float inv_alpha = 1 / alpha; 
	for (int i = 0; i < 3; i++)
	{
		m_translation(i) += alpha; 
		UpdateVertices();
		Eigen::MatrixXf delta_J = m_jointsFinal - J0;
		Eigen::MatrixXf delta_V = m_verticesFinal - V0; 
		J_joint_numeric.col(i) = Eigen::Map < Eigen::Matrix<float, 3 * 62, 1> >(delta_J.data()) * inv_alpha;
		J_vert_numeric.col(i) = Eigen::Map< Eigen::Matrix<float, 3 * 11239, 1> >(delta_V.data()) * inv_alpha; 
		m_translation(i) -= alpha; 
	}
	for (int i = 0; i < 3; i++)
	{
		m_poseParam(i) += alpha; 
		UpdateVertices(); 
		Eigen::MatrixXf delta_J = m_jointsFinal - J0;
		Eigen::MatrixXf delta_V = m_verticesFinal - V0;
		J_joint_numeric.col(i+3) = Eigen::Map < Eigen::Matrix<float, 3 * 62, 1> >(delta_J.data()) * inv_alpha;
		J_vert_numeric.col(i+3) = Eigen::Map< Eigen::Matrix<float, 3 * 11239, 1> >(delta_V.data()) * inv_alpha;
		for (int jid = 0; jid < m_jointNum; jid++)
			dQ_numeric.block<3, 3>(3 * jid, 3 * i) = (m_globalAffine.block<3, 3>(0, 4 * jid)
				- Q0.block<3, 3>(0, 4 * jid)) * inv_alpha; 
		m_poseParam(i) -= alpha;
	}
	for (int i = 0; i < 32; i++)
	{
		m_latentCode(i) += alpha; 
		UpdateVertices();
		Eigen::MatrixXf delta_J = m_jointsFinal - J0;
		Eigen::MatrixXf delta_V = m_verticesFinal - V0;
		J_joint_numeric.col(i+6) = Eigen::Map< Eigen::Matrix<float, 3 * 62, 1> >(delta_J.data()) * inv_alpha;
		J_vert_numeric.col(i+6) = Eigen::Map< Eigen::Matrix<float, 3 * 11239, 1> >(delta_V.data()) * inv_alpha;
		for (int jid = 0; jid < m_jointNum; jid++)
			dQ_numeric.block<3, 3>(3 * jid, 3 * (i+3)) = (m_globalAffine.block<3, 3>(0, 4 * jid)
				- Q0.block<3, 3>(0, 4 * jid)) * inv_alpha;
		m_latentCode(i) -= alpha;
	}

	Eigen::MatrixXf J_joint, J_vert;
	CalcPoseJacobiLatent(J_joint, J_vert); 

	Eigen::MatrixXf diff1 = J_vert - J_vert_numeric; 
	std::cout << diff1.norm() << std::endl;

	std::cout << "Numeric block:  " << std::endl; 
	std::cout << J_vert_numeric.block<10, 10>(0, 0) << std::endl; 
	std::cout << std::endl << "Analytic block: " << std::endl;
	std::cout << J_vert.block<10, 10>(0, 0) << std::endl; 
}

void PigSolver::debug_numericJacobiAA()
{
	Eigen::MatrixXf J_joint_numeric = Eigen::MatrixXf::Zero(3 * m_jointNum, 3+62*3);
	Eigen::MatrixXf J_vert_numeric = Eigen::MatrixXf::Zero(3 * m_vertexNum, 3+62*3);
	Eigen::MatrixXf V0 = m_verticesFinal;
	Eigen::MatrixXf J0 = m_jointsFinal;
	float alpha = 0.001;
	float inv_alpha = 1 / alpha;
	for (int i = 0; i < 3; i++)
	{
		m_translation(i) += alpha;
		UpdateVertices();
		Eigen::MatrixXf delta_J = m_jointsFinal - J0;
		Eigen::MatrixXf delta_V = m_verticesFinal - V0;
		J_joint_numeric.col(i) = Eigen::Map < Eigen::Matrix<float, 3 * 62, 1> >(delta_J.data()) * inv_alpha;
		J_vert_numeric.col(i) = Eigen::Map< Eigen::Matrix<float, 3 * 11239, 1> >(delta_V.data()) * inv_alpha;
		m_translation(i) -= alpha;
	}
	for (int i = 0; i < 62*3; i++)
	{
		m_poseParam(i) += alpha;
		UpdateVertices();
		Eigen::MatrixXf delta_J = m_jointsFinal - J0;
		Eigen::MatrixXf delta_V = m_verticesFinal - V0;
		J_joint_numeric.col(i + 3) = Eigen::Map < Eigen::Matrix<float, 3 * 62, 1> >(delta_J.data()) * inv_alpha;
		J_vert_numeric.col(i + 3) = Eigen::Map< Eigen::Matrix<float, 3 * 11239, 1> >(delta_V.data()) * inv_alpha;
		m_poseParam(i) -= alpha;
	}

	Eigen::MatrixXf J_joint; 
	Eigen::MatrixXf J_vert; 
	CalcPoseJacobiFullTheta(J_joint, J_vert); 
	Eigen::MatrixXf diff = J_joint - J_joint_numeric;
	std::cout << "diff.norm(): " << diff.norm() << std::endl; 
	Eigen::MatrixXf diff2 = J_vert - J_vert_numeric;
	std::cout << "diff2.norm(): " << diff2.norm() << std::endl; 

	std::cout << "Numeric block:  " << std::endl;
	std::cout << J_joint_numeric.block<10, 10>(0, 0) << std::endl;
	std::cout << std::endl << "Analytic block: " << std::endl;
	std::cout << J_joint.block<10, 10>(0, 0) << std::endl;
}