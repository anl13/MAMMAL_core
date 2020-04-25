#include "pigsolver.h"

void PigSolver::CalcPoseJacobi()
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
	m_JacobiPose = Eigen::MatrixXd::Zero(3 * N, 3 + 3 * M);

	// calculate delta rodrigues
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * m_jointNum);
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3d& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
		rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiD(pose);
	}

	//dJ_dtheta
	Eigen::MatrixXd m_jointJacobiPose = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
	for (int jointDerivativeId = 0; jointDerivativeId < m_jointNum; jointDerivativeId++)
	{
		// update translation term
		m_jointJacobiPose.block<3, 3>(jointDerivativeId * 3, 0).setIdentity();

		// update poseParam term
		for (int axisDerivativeId = 0; axisDerivativeId < 3; axisDerivativeId++)
		{
			std::vector<std::pair<bool, Eigen::Matrix4d>> globalAffineDerivative(m_jointNum, std::make_pair(false, Eigen::Matrix4d::Zero()));
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
		m_JacobiPose.block<3, 3>(3 * i, 0) = m_jointJacobiPose.block<3, 3>(3 * jIdx, 0);
		for (int k = 0; k < M; k++)
		{
			int thetaIdx = m_poseToOptimize[k];
			m_JacobiPose.block<3, 3>(3 * i, 3 + 3 * k) = m_jointJacobiPose.block<3, 3>(3 * jIdx, 3 + 3 * thetaIdx);
		}
	}

	// this version was deduced by AN Liang, 20191231
	// assume that you have computed Pose Jacobi
	// O(n^2)
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> RP(9 * m_jointNum, 3);
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> LP(3 * m_jointNum, 3 * m_jointNum);
	RP.setZero();
	LP.setZero();
	for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
	{
		for (int aIdx = 0; aIdx < 3; aIdx++)
		{
			Eigen::Matrix3d dR = rodriguesDerivative.block<3, 3>(0, 3 * (3 * jIdx + aIdx));
			if (jIdx > 0)
			{
				dR = m_globalAffine.block<3, 3>(0, 4 * m_parent(jIdx)) * dR;
			}
			RP.block<3, 3>(9 * jIdx + 3 * aIdx, 0) = dR;
		}
		LP.block<3, 3>(3 * jIdx, 3 * jIdx) = Eigen::Matrix3d::Identity();
		for (int child = jIdx + 1; child < m_jointNum; child++)
		{
			int father = m_parent(child);
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx) * m_singleAffine.block<3, 3>(0, 4 * child);
		}
	}

	Eigen::MatrixXd m_vertJacobiPose = Eigen::MatrixXd::Zero(3, 3 + 3 * m_jointNum);
	for (int i = 0; i<N; i++)
	{
		if (m_mapper[i].first != 1) continue;
		int vIdx = m_mapper[i].second;
		Eigen::Vector3d v0 = m_verticesShaped.col(vIdx);
		m_vertJacobiPose.setZero();
		for (int jIdx = 0; jIdx<m_jointNum; jIdx++)
		{
			if (m_lbsweights(jIdx, vIdx) < 0.00001) continue;
			Eigen::Vector3d j0 = m_jointsShaped.col(jIdx);
			m_vertJacobiPose += (m_lbsweights(jIdx, vIdx) * m_jointJacobiPose.middleRows(3 * jIdx, 3));
			for (int pIdx = jIdx; pIdx>-1; pIdx = m_parent(pIdx)) // poseParamIdx to derive
			{
				Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
				for (int axis_id = 0; axis_id<3; axis_id++)
				{
					Eigen::Matrix3d dR1 = RP.block<3, 3>(9 * pIdx + 3 * axis_id, 0) * LP.block<3, 3>(3 * jIdx, 3 * pIdx);
					T.col(axis_id) = dR1 * (v0 - j0) * m_lbsweights(jIdx, vIdx);
				}
				m_vertJacobiPose.block<3, 3>(0, 3 + 3 * pIdx) += T;
			}
		}
		m_JacobiPose.block<3, 3>(3 * i, 0) = m_vertJacobiPose.block<3, 3>(0, 0);
		for (int k = 0; k < M; k++)
		{
			int thetaIdx = m_poseToOptimize[k];
			m_JacobiPose.block<3, 3>(3 * i, 3 + 3 * k) = m_vertJacobiPose.block<3, 3>(0, 3 + 3 * thetaIdx);
		}
	}

	// 	// calculate vertex jacobi, deduced by ZHANG Yuxiang. Use twist approximation at end joints. 
	// 	m_vertJacobiPose = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_vertexNum, 3 + 3 * m_jointNum);
	// #pragma omp parallel for
	// 	for (int vIdx = 0; vIdx < m_vertexNum; vIdx++) 
	// 	{
	// 		for (int jIdx = 0; jIdx < m_jointNum; jIdx++) {
	// 			if (m_lbsweights(jIdx, vIdx) < 0.00001)
	// 				continue;
	// 			m_vertJacobiPose.middleRows(3 * vIdx, 3) += m_lbsweights(jIdx, vIdx) * m_jointJacobiPose.middleRows(3 * jIdx, 3);
	// 			const Eigen::Matrix3d skew = - GetSkewMatrix(m_verticesFinal.col(vIdx) - m_jointsFinal.col(jIdx));
	// 			for (int _prtIdx = jIdx; _prtIdx != -1; _prtIdx = m_parent(_prtIdx) )
	// 				m_vertJacobiPose.block<3, 3>(3 * vIdx, 3 + 3 * _prtIdx) += m_lbsweights(jIdx, vIdx) * skew * m_globalAffine.block<3, 3>(0, 4 * _prtIdx);
	// 		}
	// 	}
}

void PigSolver::Calc2DJacobi(
	const int k,
	const Eigen::MatrixXd& skel,
	Eigen::MatrixXd& H,
	Eigen::VectorXd& b
)
{
	int view = m_source.view_ids[k];
	Camera cam = m_cameras[view];
	const DetInstance& det = m_source.dets[k];
	Eigen::Matrix3d R = cam.R;
	Eigen::Matrix3d K = cam.K;
	Eigen::Vector3d T = cam.T;

	int N = m_topo.joint_num;
	int M = m_poseToOptimize.size();
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2 * N, 3 + 3 * M);
	Eigen::VectorXd r = Eigen::VectorXd::Zero(2 * N);

	for (int i = 0; i < N; i++)
	{
		Eigen::Vector3d x_local = K * (R * skel.col(i) + T);
		Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2, 3);
		D(0, 0) = 1 / x_local(2);
		D(1, 1) = 1 / x_local(2);
		D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
		D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));
		J.middleRows(2 * i, 2) = D * K * R * m_JacobiPose.middleRows(3 * i, 3);
		Eigen::Vector2d u;
		u(0) = x_local(0) / x_local(2);
		u(1) = x_local(1) / x_local(2);

		if (m_mapper[i].first < 0) continue;
		if (det.keypoints[i](2) < m_topo.kpt_conf_thresh[i]) continue;
		r.segment<2>(2 * i) = u - det.keypoints[i].segment<2>(0);
	}
	// std::cout << "K: " << k << std::endl; 
	// std::cout << "J: " << std::endl << J.middleCols(3,6) << std::endl; 
	b = -J.transpose() * r;
	H = J.transpose() * J;
}

void PigSolver::CalcPoseJacobiNumeric()
{
	m_jointJacobiPoseNumeric = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
	m_vertJacobiPoseNumeric = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_vertexNum, 3 + 3 * m_jointNum);

	UpdateVertices();
	Eigen::MatrixXd previousJ = m_jointsFinal;
	Eigen::MatrixXd previousV = m_verticesFinal;
	Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(3 * m_jointNum);
	double alpha = 0.000001;
	double inv_alpha = 1.0 / alpha;
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
}


void PigSolver::CalcShapeJacobiNumeric()
{
	m_jointJacobiShapeNumeric = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, m_shapeNum);
	m_vertJacobiShapeNumeric = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_vertexNum, m_shapeNum);
	UpdateVertices();
	Eigen::MatrixXd previousJ = m_jointsFinal;
	Eigen::MatrixXd previousV = m_verticesFinal;
	Eigen::VectorXd delta_x = Eigen::VectorXd::Zero(m_shapeNum);
	double alpha = 0.0001;
	double inv_alpha = 1.0 / alpha;
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
}

void PigSolver::CalcShapeJacobi()
{
	// compute d_joint_d_beta
	m_jointJacobiShape.resize(3 * m_jointNum, m_shapeNum);
	m_jointJacobiShape.setZero();
	for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
	{
		if (jIdx == 0)
			m_jointJacobiShape.middleRows(3 * jIdx, 3) += m_shapeBlendJ.middleRows(3 * jIdx, 3);
		else
		{
			int pIdx = m_parent(jIdx);
			m_jointJacobiShape.middleRows(3 * jIdx, 3) = m_jointJacobiShape.middleRows(3 * pIdx, 3) + 
				m_globalAffine.block<3, 3>(0, 4 * pIdx)
				* (m_shapeBlendJ.middleRows(3 * jIdx, 3) - m_shapeBlendJ.middleRows(3 * pIdx, 3));
		}
	}

	// compute d_v_d_beta
	m_vertJacobiShape.resize(3 * m_vertexNum, m_shapeNum);
	m_vertJacobiShape.setZero();
	for (int vIdx = 0; vIdx < m_vertexNum; vIdx++)
	{
		for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
		{
			if (m_lbsweights(jIdx, vIdx) < 0.00001) continue;
			m_vertJacobiShape.middleRows(3 * vIdx, 3) += ((m_jointJacobiShape.middleRows(3 * jIdx, 3)
				+ m_globalAffine.block<3, 3>(0, 4 * jIdx) * (m_shapeBlendV.middleRows(3 * vIdx, 3) - m_shapeBlendJ.middleRows(3 * jIdx, 3))
				) * m_lbsweights(jIdx, vIdx));
		}
	}
}

void PigSolver::CalcVertJacobiPose(Eigen::MatrixXd& J)
{
	int N = m_topo.joint_num;
	int M = m_poseToOptimize.size();
	m_JacobiPose = Eigen::MatrixXd::Zero(3 * N, 3 + 3 * M);
	// calculate delta rodrigues
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * m_jointNum);
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3d& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
		rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiD(pose);
	}


	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> RP(9 * m_jointNum, 3);
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> LP(3 * m_jointNum, 3 * m_jointNum);
	RP.setZero();
	LP.setZero();
	for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
	{
		for (int aIdx = 0; aIdx < 3; aIdx++)
		{
			Eigen::Matrix3d dR = rodriguesDerivative.block<3, 3>(0, 3 * (3 * jIdx + aIdx));
			if (jIdx > 0)
			{
				dR = m_globalAffine.block<3, 3>(0, 4 * m_parent(jIdx)) * dR;
			}
			RP.block<3, 3>(9 * jIdx + 3 * aIdx, 0) = dR;
		}
		LP.block<3, 3>(3 * jIdx, 3 * jIdx) = Eigen::Matrix3d::Identity();
		for (int child = jIdx + 1; child < m_jointNum; child++)
		{
			int father = m_parent(child);
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx) * m_singleAffine.block<3, 3>(0, 4 * child);
		}
	}

	//dJ_dtheta
	Eigen::MatrixXd jointJacobiPose = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * m_jointNum, 3 + 3 * m_jointNum);
	for (int jointDerivativeId = 0; jointDerivativeId < m_jointNum; jointDerivativeId++)
	{
		// update translation term
		jointJacobiPose.block<3, 3>(jointDerivativeId * 3, 0).setIdentity();

		// update poseParam term
		for (int axisDerivativeId = 0; axisDerivativeId < 3; axisDerivativeId++)
		{
			std::vector<std::pair<bool, Eigen::Matrix4d>> globalAffineDerivative(m_jointNum, std::make_pair(false, Eigen::Matrix4d::Zero()));
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

	J = Eigen::MatrixXd::Zero(3 * m_vertexNum, 3 + 3 * M);
	Eigen::MatrixXd block = Eigen::MatrixXd::Zero(3, 3 + 3 * m_jointNum);
	for (int vIdx = 0; vIdx < m_vertexNum; vIdx++)
	{
		Eigen::Vector3d v0 = m_verticesShaped.col(vIdx);
		block.setZero();
		for (int jIdx = 0; jIdx<m_jointNum; jIdx++)
		{
			if (m_lbsweights(jIdx, vIdx) < 0.00001) continue;
			Eigen::Vector3d j0 = m_jointsShaped.col(jIdx);
			block += (m_lbsweights(jIdx, vIdx) * jointJacobiPose.middleRows(3 * jIdx, 3));
			for (int pIdx = jIdx; pIdx>-1; pIdx = m_parent(pIdx)) // poseParamIdx to derive
			{
				Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
				for (int axis_id = 0; axis_id<3; axis_id++)
				{
					Eigen::Matrix3d dR1 = RP.block<3, 3>(9 * pIdx + 3 * axis_id, 0) * LP.block<3, 3>(3 * jIdx, 3 * pIdx);
					T.col(axis_id) = dR1 * (v0 - j0) * m_lbsweights(jIdx, vIdx);
				}
				block.block<3, 3>(0, 3 + 3 * pIdx) += T;
			}
		}
		J.block<3, 3>(3 * vIdx, 0) = block.block<3, 3>(0, 0);

		for (int k = 0; k < M; k++)
		{
			int thetaIdx = m_poseToOptimize[k];
			J.block<3, 3>(3 * vIdx, 3 + 3 * k) = block.block<3, 3>(0, 3 + 3 * thetaIdx);
		}
	}
}

void PigSolver::debug_jacobi()
{
	UpdateVertices();
	CalcPoseJacobi();
	CalcPoseJacobiNumeric();
	Eigen::MatrixXd C = (m_jointJacobiPose - m_jointJacobiPoseNumeric);
	Eigen::MatrixXd D = (m_vertJacobiPose - m_vertJacobiPoseNumeric);
	std::cout << BLUE_TEXT("D") << std::endl;
	// std::cout << D.middleRows(0, 3).middleCols(3,3) << std::endl; 
	std::cout << D.middleRows(0, 3).transpose() << std::endl;

	std::cout << "analytic: " << std::endl;
	// std::cout << m_vertJacobiPose.middleRows(0,3).middleCols(3,3) << std::endl; 
	std::cout << m_vertJacobiPose.middleRows(0, 3).transpose() << std::endl;

	std::cout << "numeric: " << std::endl;
	// std::cout << m_vertJacobiPoseNumeric.middleRows(0,3).middleCols(3,3) << std::endl; 
	std::cout << m_vertJacobiPoseNumeric.middleRows(0, 3).transpose() << std::endl;

	std::cout << "max diff: " << D.array().abs().matrix().maxCoeff() << std::endl;

	// CalcShapeJacobi(); 
	// CalcShapeJacobiNumeric(); 
	// Eigen::MatrixXd A = m_jointJacobiShape - m_jointJacobiShapeNumeric; 
	// Eigen::MatrixXd B = m_vertJacobiShape - m_vertJacobiShapeNumeric;
	// std::cout << RED_TEXT("A") << std::endl; 
	// std::cout << A.middleRows(0, 6).transpose() << std::endl; 
	// std::cout << GREEN_TEXT("B") << std::endl; 
	// std::cout << B.middleRows(0,6).transpose() << std::endl; 
}

void PigSolver::Calc2DJacobiNumeric(
	const int k, const Eigen::MatrixXd& skel,
	Eigen::MatrixXd& H, Eigen::VectorXd& b)
{
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
	double alpha = 0.0001;
	double inv_alpha = 1 / alpha;
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
}


Eigen::MatrixXd PigSolver::CalcShapeJacobiToSkel()
{
	CalcShapeJacobi(); 
	int N = m_topo.joint_num;
	Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3 * N, m_shapeNum);
	for (int i = 0; i < N; i++)
	{
		if (m_mapper[i].first < 0)continue;
		if (m_mapper[i].first == 0)
		{
			int jid = m_mapper[i].second;
			J.middleRows(3 * i, 3) = m_jointJacobiShape.middleRows(3 * jid, 3);
		}
		else
		{
			int vid = m_mapper[i].second;
			J.middleRows(3 * i, 3) = m_vertJacobiShape.middleRows(3 * vid, 3);
		}
	}
	return J; 
}

