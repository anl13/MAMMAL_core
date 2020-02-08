#include "pigsolver.h"

#include <iostream> 
#include <iomanip>
#include <fstream> 
#include "../utils/colorterminal.h"

//#define DEBUG_SOLVER

void PigSolver::setCameras(const vector<Camera>& _cameras)
{
	m_cameras = _cameras; 
}

void PigSolver::setSource(const MatchedInstance& _source)
{
	m_source = _source;
}

void PigSolver::normalizeCamera()
{
	for (int i = 0; i < m_cameras.size(); i++)
	{
		m_cameras[i].NormalizeK();
	}
}

void PigSolver::normalizeSource()
{
	for (int i = 0; i < m_source.dets.size(); i++)
	{
		for (int k = 0; k < m_source.dets[i].keypoints.size(); k++)
		{
			m_source.dets[i].keypoints[k](0) /= 1920;
			m_source.dets[i].keypoints[k](1) /= 1080;
		}
	}
}


void PigSolver::CalcZ()
{
	int N = m_topo.joint_num; 
	Z = Eigen::MatrixXd::Zero(3, N);
	for (int i = 0; i < N; i++)
	{
		Eigen::Vector3d X = Eigen::Vector3d::Zero(); // joint position to solve.  
		int validnum = 0;
		for (int k = 0; k < m_source.view_ids.size(); k++)
		{
			if (m_source.dets[k].keypoints[i](2) < m_topo.kpt_conf_thresh[i]) continue;
			validnum++;
		}
		if (validnum < 2)
		{
			Z.col(i) = X;
			continue;
		}

		// usually, converge in 3 iteractions 
		for (int iter = 0; iter < 100; iter++)
		{
			Eigen::Matrix3d H1 = Eigen::Matrix3d::Zero();
			Eigen::Vector3d b1 = Eigen::Vector3d::Zero();
			for (int k = 0; k < m_source.view_ids.size(); k++)
			{
				int view = m_source.view_ids[k];
				Camera cam = m_cameras[view];
				Eigen::Vector3d keypoint = m_source.dets[k].keypoints[i];
				if (keypoint(2) < m_topo.kpt_conf_thresh[i]) continue;
				Eigen::Vector3d x_local = cam.K * (cam.R * X + cam.T);
				Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2, 3);
				D(0, 0) = 1 / x_local(2);
				D(1, 1) = 1 / x_local(2);
				D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
				D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));
				Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 3);
				J = D * cam.K * cam.R;
				Eigen::Vector2d u;
				u(0) = x_local(0) / x_local(2);
				u(1) = x_local(1) / x_local(2);
				Eigen::Vector2d r = u - keypoint.segment<2>(0);
				H1 += J.transpose() * J;
				b1 += -J.transpose() * r;
			}

			Eigen::Matrix3d DTD = Eigen::Matrix3d::Identity();
			double w1 = 1;
			double lambda = 0.0;
			Eigen::Matrix3d H = H1 * w1 + DTD * lambda;
			Eigen::Vector3d b = b1 * w1;
			Eigen::Vector3d delta = H.ldlt().solve(b);
			X = X + delta;
			if (delta.norm() < 0.00001) break;
			else
			{
//#ifdef DEBUG_SOLVER
//				std::cout << "iter : " << iter
//					<< " delta: " << delta.norm() << std::endl;
//#endif 
			}
		}

		// // as a comparison, solve with ceres 
		// std::vector<Camera> v_cams; 
		// std::vector<Eigen::Vector3d> v_joints; 
		// for(int k = 0; k < m_source.view_ids.size(); k++)
		// {
		//     if(m_source.dets[k].keypoints[i](2) < m_topo.kpt_conf_thresh[i]) continue; 
		//     int view = m_source.view_ids[k];
		//     v_cams.push_back(m_cameras[view]); 
		//     v_joints.push_back(m_source.dets[k].keypoints[i]); 
		// }
		// if(v_cams.size() < 2)
		// {
		//     Z.col(i) = X; 
		// }
		// else 
		// {
		//     Eigen::Vector3d W = triangulate_ceres(v_cams, v_joints);
		//     Z.col(i) = W; 
		// }
		Z.col(i) = X;
	}
}

Eigen::MatrixXd PigSolver::getRegressedSkel()
{
	int N = m_topo.joint_num; 
	Eigen::MatrixXd skel = Eigen::MatrixXd::Zero(3, N);
	for (int i = 0; i < N; i++)
	{
		if (m_mapper[i].first < 0) continue;
		if (m_mapper[i].first == 0)
		{
			int jIdx = m_mapper[i].second;
			skel.col(i) = m_jointsFinal.col(jIdx);
		}
		else if (m_mapper[i].first == 1)
		{
			int vIdx = m_mapper[i].second;
			skel.col(i) = m_verticesFinal.col(vIdx);
		}
		else {
			std::cout << RED_TEXT("fatal error: in SMAL_2DSOLVER::getRegressedSkel.") << std::endl;
			exit(-1);
		}
	}
	return skel;
}


void PigSolver::globalAlign() // procrustes analysis, rigid align (R,t), and transform Y
{
	CalcZ(); 
	if (m_frameid > 0.5) return; 
	int N = m_topo.joint_num; 
	m_weights.resize(N);
	m_weightsEigen = Eigen::VectorXd::Zero(N * 3);
	Eigen::MatrixXd skel = getRegressedSkel(); 
	// STEP 1: compute scale
	std::vector<double> target_bone_lens;
	std::vector<double> source_bone_lens;
	for (int bid = 0; bid < m_topo.bones.size(); bid++)
	{
		int sid = m_topo.bones[bid](0);
		int eid = m_topo.bones[bid](1);
		if (Z.col(sid).norm() == 0 || Z.col(eid).norm() == 0) continue;
		double target_len = (Z.col(sid) - Z.col(eid)).norm();
		double source_len = (skel.col(sid) - skel.col(eid)).norm();
		target_bone_lens.push_back(target_len);
		source_bone_lens.push_back(source_len);
	}
	double a = 0;
	double b = 0;
	for (int i = 0; i < target_bone_lens.size(); i++)
	{
		a += target_bone_lens[i] * source_bone_lens[i];
		b += source_bone_lens[i] * source_bone_lens[i];
	}
	double alpha = a / b;
	if (m_frameid > 0)
	{
		//double scale = m_scale * alpha; 
		//double m_scale = (m_scale * m_frameid + scale) / (m_frameid + 1); 
	}
	else
	{
		m_scale = alpha;
	}
	
	m_jointsOrigin *= m_scale; 
	m_verticesOrigin *= m_scale; 
	if (m_shapeNum > 0)
	{
		m_shapeBlendV *= alpha;
		m_shapeBlendJ *= alpha;
	}

	std::cout << "scale: " << m_scale << std::endl; 
	UpdateVertices();
	if (m_frameid > 0) return; 
	m_frameid += 1;

	// STEP 2: compute translation 
	Eigen::Vector3d barycenter_target = Z.col(18);
	int center_id = m_mapper[18].second; 
	Eigen::Vector3d barycenter_source = m_jointsShaped.col(center_id);
	m_translation = barycenter_target - barycenter_source;

	// STEP 3 : compute global rotation 
	Eigen::MatrixXd A, B;
	int nonzero = 0;
	for (int i = 0; i < N; i++) {
		if (Z.col(i).norm() > 0) {
			nonzero++;
			m_weights[i] = 1;
			m_weightsEigen.segment<3>(3 * i) = Eigen::Vector3d::Ones();
		}
		else m_weights[i] = 0;
	}
	A.resize(3, nonzero); B.resize(3, nonzero);
	int k = 0;
	for (int i = 0; i < Z.cols(); i++)
	{
		if (m_mapper[i].first < 0) continue;
		if (Z.col(i).norm() > 0)
		{
			A.col(k) = Z.col(i);
			if (m_mapper[i].first == 0) B.col(k) = m_jointsFinal.col(m_mapper[i].second);
			else B.col(k) = m_verticesFinal.col(m_mapper[i].second);
			k++;
		}
	}
	Eigen::MatrixXd V_target = A.colwise() - barycenter_target;
	Eigen::MatrixXd V_source = B.colwise() - barycenter_source;
	Eigen::Matrix3d S = V_source * V_target.transpose();
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
#ifdef DEBUG_SOLVER
	std::cout << BLUE_TEXT("svd singular values: ") << svd.singularValues().transpose() << std::endl;
#endif 
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::Matrix3d R = V * U.transpose();
	Eigen::AngleAxisd ax(R);
	m_poseParam.segment<3>(0) = ax.axis() * ax.angle();
#ifdef DEBUG_SOLVER
	std::cout << BLUE_TEXT("m_translation: ") << m_translation.transpose() << std::endl;
#endif 
	UpdateVertices();
}


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

Eigen::VectorXd PigSolver::getRegressedSkelProj(
	const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& T)
{
	int N = m_topo.joint_num;
	Eigen::MatrixXd skel = getRegressedSkel();
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> proj;
	proj.resize(2, N); proj.setZero();
	Eigen::MatrixXd local = K * ((R * skel).colwise() + T);
	for (int i = 0; i < N; i++) proj.col(i) = local.block<2, 1>(0, i) / local(2, i);
	Eigen::VectorXd proj_vec = Eigen::Map<Eigen::VectorXd>(proj.data(), 2 * N);
	return proj_vec;
}

void PigSolver::optimizePose(const int maxIterTime, const double updateTolerance)
{
#ifdef DEBUG_SOLVER
	std::cout << GREEN_TEXT("solving pose ... ") << std::endl;
#endif 
	int M = m_poseToOptimize.size();
	int N = m_topo.joint_num;
	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
#ifdef DEBUG_SOLVER
		std::cout << "iter time: " << iterTime << std::endl;
#endif 
		UpdateVertices();
		CalcPoseJacobi();
		Eigen::MatrixXd skel = getRegressedSkel();

		Eigen::VectorXd theta(3 + 3 * M);
		theta.segment<3>(0) = m_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
		}
		Eigen::VectorXd theta0 = theta;
		// solve
		Eigen::MatrixXd H1 = Eigen::MatrixXd::Zero(3 + 3 * M, 3 + 3 * M); // data term 
		Eigen::VectorXd b1 = Eigen::VectorXd::Zero(3 + 3 * M);  // data term 
		for (int k = 0; k < m_source.view_ids.size(); k++)
		{
			Eigen::MatrixXd H_view;
			Eigen::VectorXd b_view;
			Calc2DJacobi(k, skel, H_view, b_view);
			// Calc2DJacobiNumeric(k,skel, H_view, b_view); 
			H1 += H_view;
			b1 += b_view;
		}
		double lambda = 0.001;
		double w1 = 1;
		double w_reg = 0.01;
		double w_temp = 0.01; 
		Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXd H_reg = DTD;  // reg term 
		Eigen::VectorXd b_reg = -theta; // reg term 
		Eigen::VectorXd b_temp = -theta0; 

		Eigen::MatrixXd H = H1 * w1 + H_reg * w_reg + DTD * lambda + DTD * w_temp;
		Eigen::VectorXd b = b1 * w1 + b_reg * w_reg + b_temp * w_temp;

		Eigen::VectorXd delta = H.ldlt().solve(b);

		// update 
		m_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
		}
#ifdef DEBUG_SOLVER
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(H);
		double cond = svd.singularValues()(0)
			/ svd.singularValues()(svd.singularValues().size() - 1);
		std::cout << "H cond: " << cond << std::endl;
		std::cout << "delta.norm() : " << delta.norm() << std::endl;
#endif 
		// if(iterTime == 1) break; 
		if (delta.norm() < updateTolerance) break;
	}
}

void PigSolver::computePivot()
{
	// assume center is always observed 
	Eigen::MatrixXd skel = getRegressedSkel(); 
	Eigen::Vector3d headz = Z.col(0); 
	Eigen::Vector3d centerz = Z.col(20);
	Eigen::Vector3d tailz = Z.col(18); 
	Eigen::Vector3d heads = skel.col(0); 
	Eigen::Vector3d centers = skel.col(20); 
	Eigen::Vector3d tails = skel.col(18); 

	// m_pivot[0]: head(nose)
	// m_pivot[1]: center
	// m_pivot[2]: tail(tail root)
	m_pivot.resize(3);
	
	// choosing policy 
	m_pivot[1] = centers; 
	
	m_pivot[0] = heads; 
	m_pivot[2] = tails; 
	
	m_bodystate.trans = m_translation; 
	m_bodystate.pose = m_poseParam; 
	m_bodystate.frameid = m_frameid; 
	m_bodystate.id = m_id; 
	m_bodystate.points = m_pivot; 
	m_bodystate.center = m_pivot[1];
	m_bodystate.scale = m_scale;
}

void PigSolver::readBodyState(std::string filename)
{
	m_bodystate.loadState(filename); 
	m_translation = m_bodystate.trans;
	m_poseParam = m_bodystate.pose;
	m_frameid = m_bodystate.id; 
	m_id = m_bodystate.id;
	m_pivot = m_bodystate.points; 
	m_scale = m_bodystate.scale; 

	UpdateVertices(); 
	auto skel = getRegressedSkel(); 
	vector<Eigen::Vector3d> est(3); 
	est[0] = skel.col(0); 
	est[1] = skel.col(20); 
	est[2] = skel.col(18); 
	m_scale = ((m_pivot[0] - m_pivot[1]).norm() + (m_pivot[1] - m_pivot[2]).norm() + (m_pivot[2] - m_pivot[0]).norm())
		/ ((est[0] - est[1]).norm() + (est[1] - est[2]).norm() + (est[2] - est[0]).norm());
	m_jointsOrigin *= m_scale; 
	m_verticesOrigin *= m_scale; 
	UpdateVertices(); 
}