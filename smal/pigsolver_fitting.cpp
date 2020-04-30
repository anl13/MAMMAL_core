#include "pigsolver.h"

#include <iostream> 
#include <iomanip>
#include <fstream> 
#include "../utils/colorterminal.h"
#include <cstdlib> 
#include <json/json.h>
//#define DEBUG_SOLVER

#define OPTIM_BY_PAIR

PigSolver::PigSolver(const std::string& _configfile):PigModel(_configfile) 
{
	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(_configfile);
	if (!instream.is_open())
	{
		std::cout << "can not open " << _configfile << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}

	std::string topo_type = root["topo"].asString(); 
	m_topo = getSkelTopoByType(topo_type);
	m_poseToOptimize.clear(); 
	for (const auto& c : root["pose_to_solve"])
	{
		m_poseToOptimize.push_back(c.asInt()); 
	}
	m_mapper.clear(); 
	for (const auto& c : root["mapper"])
	{
		std::pair<int, int> a_map_pair;
		a_map_pair.first = c[0].asInt(); 
		a_map_pair.second = c[1].asInt(); 
		m_mapper.push_back(a_map_pair); 
	}

	m_scale = 1;
	m_frameid = 0.0;
	tmp_init = false;

	std::string sym_file = m_folder + "/sym.txt";
	std::ifstream is(sym_file);
	if (is.is_open())
	{
		m_symIdx.resize(m_vertexNum);
		for (int i = 0; i < m_vertexNum; i++)
		{
			is >> m_symIdx[i];
		}
	}

	// read optim pair;
	m_optimPairs.clear(); 
	for (auto const&c : root["optimize_pair"])
	{
		CorrPair pair; 
		pair.target = c[0].asInt();
		pair.type = c[1].asInt();
		pair.index = c[2].asInt();
		pair.weight = c[3].asDouble();
		m_optimPairs.push_back(pair);
	}

	instream.close();
}

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

Eigen::MatrixXd PigSolver::getRegressedSkelbyPairs()
{
	int N = m_topo.joint_num;
	Eigen::MatrixXd joints = Eigen::MatrixXd::Zero(3, N);
	for (int i = 0; i < m_optimPairs.size(); i++)
	{
		CorrPair P = m_optimPairs[i];
		int t = P.target; 
		if (P.type == 1)
		{
			joints.col(t) += m_verticesFinal.col(P.index) * P.weight;
		}
		else if(P.type==0)
		{
			joints.col(t) += m_jointsFinal.col(P.index) * P.weight;
		}
	}
	return joints; 
}

void PigSolver::globalAlign() // procrustes analysis, rigid align (R,t), and transform Y
{
	CalcZ(); 
	if (m_frameid > 0) return; 
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
	m_scale = alpha;
	RescaleOriginVertices();

	UpdateVertices();
	if (m_frameid > 0) return; 
	m_frameid += 1;

	// STEP 2: compute translation 
	Eigen::Vector3d barycenter_target = Z.col(20);
	int center_id = m_mapper[20].second; 
	Eigen::Vector3d barycenter_source = m_jointsDeformed.col(center_id);
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
	int M = m_poseToOptimize.size();
	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();
		Eigen::MatrixXd poseJ3d;
#ifndef OPTIM_BY_PAIR
		CalcSkelJacobiPartThetaByMapper(poseJ3d);
		Eigen::MatrixXd skel = getRegressedSkel();
#else
		CalcSkelJacobiPartThetaByPairs(poseJ3d);
		Eigen::MatrixXd skel = getRegressedSkelbyPairs();
#endif 

		Eigen::VectorXd theta(3 + 3 * M);
		theta.segment<3>(0) = m_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
		}

		// solve
		Eigen::MatrixXd H1 = Eigen::MatrixXd::Zero(3 + 3 * M, 3 + 3 * M); // data term 
		Eigen::VectorXd b1 = Eigen::VectorXd::Zero(3 + 3 * M);  // data term 
		for (int k = 0; k < m_source.view_ids.size(); k++)
		{
			Eigen::MatrixXd H_view;
			Eigen::VectorXd b_view;
#ifndef OPTIM_BY_PAIR
			CalcPose2DTermByMapper(k, skel, poseJ3d, H_view, b_view);
#else 
			CalcPose2DTermByPairs(k, skel, poseJ3d, H_view, b_view);
#endif 
			H1 += H_view;
			b1 += b_view;
		}
		double lambda = 0.0005;
		double w1 = 1;
		double w_reg = 0.001; 
		Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXd H_reg = DTD;  // reg term 
		Eigen::VectorXd b_reg = -theta; // reg term 

		Eigen::MatrixXd H = H1 * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXd b = b1 * w1 + b_reg * w_reg;

		Eigen::VectorXd delta = H.ldlt().solve(b);
		//std::cout << "data term b: " << b1.norm() << std::endl;
		//std::cout << "reg  term b: " << b_reg.norm() << std::endl; 

		// update 
		m_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
		}
#if 1
		//Eigen::JacobiSVD<Eigen::MatrixXd> svd(H);
		//double cond = svd.singularValues()(0)
		//	/ svd.singularValues()(svd.singularValues().size() - 1);
		//std::cout << "H cond: " << cond << std::endl;
		//std::cout << "delta.norm() : " << delta.norm() << std::endl;
#endif 
		// if(iterTime == 1) break; 
		if (delta.norm() < updateTolerance) break;
	}
}

void PigSolver::computePivot()
{
	// assume center is always observed 
	Eigen::MatrixXd skel = getRegressedSkel(); 
	//Eigen::Vector3d headz = Z.col(0); 
	//Eigen::Vector3d centerz = Z.col(20);
	//Eigen::Vector3d tailz = Z.col(18); 
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
	m_frameid = m_bodystate.frameid; 
	m_id = m_bodystate.id;
	m_pivot = m_bodystate.points; 
	m_scale = m_bodystate.scale; 

	//UpdateVertices(); 
	//auto skel = getRegressedSkel(); 
	//vector<Eigen::Vector3d> est(3); 
	//est[0] = skel.col(0); 
	//est[1] = skel.col(20); 
	//est[2] = skel.col(18); 
	//m_scale = ((m_pivot[0] - m_pivot[1]).norm() + (m_pivot[1] - m_pivot[2]).norm() + (m_pivot[2] - m_pivot[0]).norm())
	//	/ ((est[0] - est[1]).norm() + (est[1] - est[2]).norm() + (est[2] - est[0]).norm());
	
	if (!tmp_init)
	{
		tmp_init = true; 
		m_jointsOrigin *= m_scale;
		m_verticesOrigin *= m_scale;
	}

	UpdateVertices(); 
}


// toy function: optimize shape without pose. 
void PigSolver::FitShapeToVerticesSameTopo(const int maxIterTime, const double terminal)
{
	std::cout << GREEN_TEXT("solving shape ... ") << std::endl;
	Eigen::VectorXd V_target = Eigen::Map<Eigen::VectorXd>(m_targetVSameTopo.data(), 3 * m_vertexNum);
	int iter = 0;
	for (; iter < maxIterTime; iter++)
	{
		UpdateVertices();
		Eigen::MatrixXd jointJacobiShape, vertJacobiShape;
		CalcShapeJacobi(jointJacobiShape, vertJacobiShape);
		Eigen::VectorXd r = Eigen::Map<Eigen::VectorXd>(m_verticesFinal.data(), 3 * m_vertexNum) - V_target;
		Eigen::MatrixXd H1 = vertJacobiShape.transpose() * vertJacobiShape;
		Eigen::VectorXd b1 = -vertJacobiShape.transpose() * r;
		Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(m_shapeNum, m_shapeNum);  // Leveberg Marquart
		Eigen::MatrixXd H_reg = DTD;
		Eigen::VectorXd b_reg = -m_shapeParam;
		double lambda = 0.001;
		double w1 = 1;
		double w_reg = 0.01;
		Eigen::MatrixXd H = H1 * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXd b = b1 * w1 + b_reg * w_reg;

		Eigen::VectorXd delta = H.ldlt().solve(b);
		m_shapeParam = m_shapeParam + delta;
#ifdef DEBUG_SOLVER
		std::cout << "residual     : " << r.norm() << std::endl;
		std::cout << "delta.norm() : " << delta.norm() << std::endl;
#endif 
		if (delta.norm() < terminal) break;
	}
#ifdef DEBUG_SOLVER
	std::cout << "iter times: " << iter << std::endl;
#endif  
}

void PigSolver::globalAlignToVerticesSameTopo()
{
	UpdateVertices();
	Eigen::Vector3d barycenter_target = m_targetVSameTopo.rowwise().mean();
	Eigen::Vector3d barycenter_source = m_verticesFinal.rowwise().mean();

	m_translation = barycenter_target - barycenter_source;
	Eigen::MatrixXd V_target = m_targetVSameTopo.colwise() - barycenter_target;
	Eigen::MatrixXd V_source = m_verticesFinal.colwise() - barycenter_source;
	Eigen::Matrix3d S = V_source * V_target.transpose();
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
	std::cout << svd.singularValues() << std::endl;
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::Matrix3d R = V * U.transpose();
	Eigen::AngleAxisd ax(R);
	m_poseParam.segment<3>(0) = ax.axis() * ax.angle();
}

Eigen::MatrixXd PigSolver::getRegressedSkelTPose()
{
	int N = m_topo.joint_num;
	Eigen::MatrixXd skel = Eigen::MatrixXd::Zero(3, N);
	for (int i = 0; i < N; i++)
	{
		if (m_mapper[i].first < 0) continue;
		if (m_mapper[i].first == 0)
		{
			int jIdx = m_mapper[i].second;
			skel.col(i) = m_jointsShaped.col(jIdx);
		}
		else if (m_mapper[i].first == 1)
		{
			int vIdx = m_mapper[i].second;
			skel.col(i) = m_verticesShaped.col(vIdx);
		}
		else {
			std::cout << RED_TEXT("fatal error: in ::getRegressedSkelTPose.") << std::endl;
			exit(-1);
		}
	}
	return skel;
}
