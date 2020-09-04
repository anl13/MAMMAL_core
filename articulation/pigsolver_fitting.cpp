#include "pigsolver.h"

#include <iostream> 
#include <iomanip>
#include <fstream> 
#include "../utils/colorterminal.h"
#include "../utils/timer_util.h"
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
	m_symNum = root["sym_num"].asInt(); 

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
			m_symIdx[i].resize(m_symNum);
			for(int k = 0;  k < m_symNum; k++)
			is >> m_symIdx[i][k];
		}
	}
	is.close();

	std::string symweight_file = m_folder + "/symweights.txt";
	std::ifstream is_symweight(symweight_file);
	if (is_symweight.is_open())
	{
		m_symweights.resize(m_vertexNum);
		for (int i = 0; i < m_vertexNum; i++)
		{
			m_symweights[i].resize(m_symNum);
			for (int k = 0; k < m_symNum; k++)
				is_symweight >> m_symweights[i][k];
		}
	}
	is_symweight.close(); 

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
	Z = Eigen::MatrixXf::Zero(3, N);
	for (int i = 0; i < N; i++)
	{
		Eigen::Vector3f X = Eigen::Vector3f::Zero(); // joint position to solve.  
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
			Eigen::Matrix3f H1 = Eigen::Matrix3f::Zero();
			Eigen::Vector3f b1 = Eigen::Vector3f::Zero();
			for (int k = 0; k < m_source.view_ids.size(); k++)
			{
				int view = m_source.view_ids[k];
				Camera cam = m_cameras[view];
				Eigen::Vector3f keypoint = m_source.dets[k].keypoints[i];
				if (keypoint(2) < m_topo.kpt_conf_thresh[i]) continue;
				Eigen::Vector3f x_local = cam.K * (cam.R * X + cam.T);
				Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
				D(0, 0) = 1 / x_local(2);
				D(1, 1) = 1 / x_local(2);
				D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
				D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));
				Eigen::MatrixXf J = Eigen::MatrixXf::Zero(2, 3);
				J = D * cam.K * cam.R;
				Eigen::Vector2f u;
				u(0) = x_local(0) / x_local(2);
				u(1) = x_local(1) / x_local(2);
				Eigen::Vector2f r = u - keypoint.segment<2>(0);
				H1 += J.transpose() * J;
				b1 += -J.transpose() * r;
			}

			Eigen::Matrix3f DTD = Eigen::Matrix3f::Identity();
			float w1 = 1;
			float lambda = 0.0;
			Eigen::Matrix3f H = H1 * w1 + DTD * lambda;
			Eigen::Vector3f b = b1 * w1;
			Eigen::Vector3f delta = H.ldlt().solve(b);
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

Eigen::MatrixXf PigSolver::getRegressedSkel()
{
	int N = m_topo.joint_num; 
	Eigen::MatrixXf skel = Eigen::MatrixXf::Zero(3, N);
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

Eigen::MatrixXf PigSolver::getRegressedSkelbyPairs()
{
	int N = m_topo.joint_num;
	Eigen::MatrixXf joints = Eigen::MatrixXf::Zero(3, N);
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
	m_weightsEigen = Eigen::VectorXf::Zero(N * 3);
	Eigen::MatrixXf skel = getRegressedSkel(); 
	// STEP 1: compute scale
	std::vector<float> target_bone_lens;
	std::vector<float> source_bone_lens;
	for (int bid = 0; bid < m_topo.bones.size(); bid++)
	{
		int sid = m_topo.bones[bid](0);
		int eid = m_topo.bones[bid](1);
		if (Z.col(sid).norm() == 0 || Z.col(eid).norm() == 0) continue;
		float target_len = (Z.col(sid) - Z.col(eid)).norm();
		float source_len = (skel.col(sid) - skel.col(eid)).norm();
		target_bone_lens.push_back(target_len);
		source_bone_lens.push_back(source_len);
	}
	float a = 0;
	float b = 0;
	for (int i = 0; i < target_bone_lens.size(); i++)
	{
		a += target_bone_lens[i] * source_bone_lens[i];
		b += source_bone_lens[i] * source_bone_lens[i];
	}

	/*std::cout << "a: " << a << "  b: " << b << std::endl; */
	float alpha = a / b;
	m_scale = alpha * m_scale;
	RescaleOriginVertices(alpha);

	UpdateVertices();
	if (m_frameid > 0) return; 
	m_frameid += 1;

	// STEP 2: compute translation 
	Eigen::Vector3f barycenter_target = Z.col(18);
	int center_id = m_mapper[18].second; 
	Eigen::Vector3f barycenter_source = m_jointsDeformed.col(center_id);
	m_translation += barycenter_target - barycenter_source;

	// STEP 3 : compute global rotation 
	Eigen::MatrixXf A, B;
	int nonzero = 0;
	for (int i = 0; i < N; i++) {
		if (Z.col(i).norm() > 0) {
			nonzero++;
			m_weights[i] = 1;
			m_weightsEigen.segment<3>(3 * i) = Eigen::Vector3f::Ones();
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
	Eigen::MatrixXf V_target = A.colwise() - barycenter_target;
	Eigen::MatrixXf V_source = B.colwise() - barycenter_source;
	Eigen::Matrix3f S = V_source * V_target.transpose();
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
#ifdef DEBUG_SOLVER
	std::cout << BLUE_TEXT("svd singular values: ") << svd.singularValues().transpose() << std::endl;
#endif 
	Eigen::MatrixXf U = svd.matrixU();
	Eigen::MatrixXf V = svd.matrixV();
	Eigen::Matrix3f R = V * U.transpose();
	Eigen::Matrix3f R0 = GetRodrigues(m_poseParam.segment<3>(0));
	Eigen::AngleAxisf ax(R * R0);
	m_poseParam.segment<3>(0) = ax.axis() * ax.angle();
#ifdef DEBUG_SOLVER
	std::cout << BLUE_TEXT("m_translation: ") << m_translation.transpose() << std::endl;
#endif 
	UpdateVertices();
}


Eigen::VectorXf PigSolver::getRegressedSkelProj(
	const Eigen::Matrix3f& K, const Eigen::Matrix3f& R, const Eigen::Vector3f& T)
{
	int N = m_topo.joint_num;
	Eigen::MatrixXf skel = getRegressedSkel();
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> proj;
	proj.resize(2, N); proj.setZero();
	Eigen::MatrixXf local = K * ((R * skel).colwise() + T);
	for (int i = 0; i < N; i++) proj.col(i) = local.block<2, 1>(0, i) / local(2, i);
	Eigen::VectorXf proj_vec = Eigen::Map<Eigen::VectorXf>(proj.data(), 2 * N);
	return proj_vec;
}



void PigSolver::optimizePose(const int maxIterTime, const float updateTolerance)
{
	int M = m_poseToOptimize.size();
	int paramNum = m_isLatent? 38: (3+3*M);
	if (theta_last.size() == 0)
	{
		theta_last.resize(paramNum);
		theta_last.setZero(); 
		theta_last.segment<3>(0) = m_translation; 
		theta_last.segment<3>(3) = m_poseParam.segment<3>(0); 
	}
	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();

		//std::stringstream ss;
		//ss << "G:/pig_results/debug/fitting_" << iterTime << ".obj";
		//SaveObj(ss.str()); 

		Eigen::MatrixXf poseJ3d;

		TimerUtil::Timer<std::chrono::milliseconds> tt; 
		tt.Start();
		if (m_isLatent) CalcSkelJacobiByPairsLatent(poseJ3d); 
		else CalcSkelJacobiPartThetaByPairs(poseJ3d);
		
		Eigen::MatrixXf skel = getRegressedSkelbyPairs();

		Eigen::VectorXf theta(paramNum);
		if (m_isLatent)
		{
			theta.segment<3>(0) = m_translation;
			theta.segment<3>(3) = m_poseParam.segment<3>(0); 
			theta.segment<32>(6) = m_latentCode;
		}
		else
		{
			theta.segment<3>(0) = m_translation;
			for (int i = 0; i < M; i++)
			{
				int jIdx = m_poseToOptimize[i]; 
				theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
			}
		}

		// solve
		Eigen::MatrixXf H1 = Eigen::MatrixXf::Zero(paramNum, paramNum); // data term 
		Eigen::VectorXf b1 = Eigen::VectorXf::Zero(paramNum);  // data term 
		for (int k = 0; k < m_source.view_ids.size(); k++)
		{
			Eigen::MatrixXf H_view;
			Eigen::VectorXf b_view;
			CalcPose2DTermByPairs(k, skel, poseJ3d, H_view, b_view);
			H1 += H_view;
			b1 += b_view;
		}
		float lambda = 0.0005;
		float w1 = 1;
		float w_reg = 0.01; 
		float w_temp = 0.0; 
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(paramNum, paramNum);
		Eigen::MatrixXf H_reg = DTD;  // reg term 
		Eigen::VectorXf b_reg = -theta; // reg term 

		Eigen::MatrixXf H = H1 * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXf b = b1 * w1 + b_reg * w_reg;

		if (m_frameid >= 0)
		{
			Eigen::MatrixXf H_temp = DTD; 
			Eigen::VectorXf b_temp = theta_last - theta; 
			H += DTD * w_temp; 
			b += b_temp * w_temp; 
		}

		Eigen::VectorXf delta = H.ldlt().solve(b);
		//std::cout << "data term b: " << b1.norm() << std::endl;
		//std::cout << "reg  term b: " << b_reg.norm() << std::endl; 

		// update 
		if (m_isLatent)
		{
			float learning_rate = 1; 
			m_translation += delta.segment<3>(0) * learning_rate; 
			m_poseParam.segment<3>(0) += delta.segment<3>(3) * learning_rate; 
			m_latentCode += delta.segment<32>(6) * learning_rate; 
		}
		else
		{
			m_translation += delta.segment<3>(0);
			for (int i = 0; i < M; i++)
			{
				int jIdx = m_poseToOptimize[i];
				m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
			}
		}

#if 1
		//Eigen::JacobiSVD<Eigen::MatrixXd> svd(H);
		//double cond = svd.singularValues()(0)
		//	/ svd.singularValues()(svd.singularValues().size() - 1);
		//std::cout << "H cond: " << cond << std::endl;
		//std::cout << "delta.norm() : " << delta.norm() << std::endl;
#endif 
		// if(iterTime == 1) break; 
		//std::cout << "pose delta.norm(): " << delta.norm() << std::endl; 
		if (delta.norm() < updateTolerance) break;
	}
}

//void PigSolver::computePivot()
//{
//	// assume center is always observed 
//	Eigen::MatrixXf skel = getRegressedSkel(); 
//	//Eigen::Vector3d headz = Z.col(0); 
//	//Eigen::Vector3d centerz = Z.col(20);
//	//Eigen::Vector3d tailz = Z.col(18); 
//	Eigen::Vector3d heads = skel.col(0); 
//	Eigen::Vector3d centers = skel.col(20); 
//	Eigen::Vector3d tails = skel.col(18); 
//
//	// m_pivot[0]: head(nose)
//	// m_pivot[1]: center
//	// m_pivot[2]: tail(tail root)
//	m_pivot.resize(3);
//	
//	// choosing policy 
//	m_pivot[1] = centers; 
//	
//	m_pivot[0] = heads; 
//	m_pivot[2] = tails; 
//	
//	m_bodystate.trans = m_translation; 
//	m_bodystate.pose = m_poseParam; 
//	m_bodystate.frameid = m_frameid; 
//	m_bodystate.id = m_id; 
//	m_bodystate.points = m_pivot; 
//	m_bodystate.center = m_pivot[1];
//	m_bodystate.scale = m_scale;
//}

//void PigSolver::readBodyState(std::string filename)
//{
//	m_bodystate.loadState(filename); 
//	m_translation = m_bodystate.trans;
//	m_poseParam = m_bodystate.pose;
//	m_frameid = m_bodystate.frameid; 
//	m_id = m_bodystate.id;
//	m_pivot = m_bodystate.points; 
//	m_scale = m_bodystate.scale; 
//
//	//UpdateVertices(); 
//	//auto skel = getRegressedSkel(); 
//	//vector<Eigen::Vector3d> est(3); 
//	//est[0] = skel.col(0); 
//	//est[1] = skel.col(20); 
//	//est[2] = skel.col(18); 
//	//m_scale = ((m_pivot[0] - m_pivot[1]).norm() + (m_pivot[1] - m_pivot[2]).norm() + (m_pivot[2] - m_pivot[0]).norm())
//	//	/ ((est[0] - est[1]).norm() + (est[1] - est[2]).norm() + (est[2] - est[0]).norm());
//	
//	if (!tmp_init)
//	{
//		tmp_init = true; 
//		m_jointsOrigin *= m_scale;
//		m_verticesOrigin *= m_scale;
//	}
//
//	UpdateVertices(); 
//}


// toy function: optimize shape without pose. 
void PigSolver::FitShapeToVerticesSameTopo(const int maxIterTime, const float terminal)
{
	std::cout << GREEN_TEXT("solving shape ... ") << std::endl;
	Eigen::VectorXf V_target = Eigen::Map<Eigen::VectorXf>(m_targetVSameTopo.data(), 3 * m_vertexNum);
	int iter = 0;
	for (; iter < maxIterTime; iter++)
	{
		UpdateVertices();
		Eigen::MatrixXf jointJacobiShape, vertJacobiShape;
		CalcShapeJacobi(jointJacobiShape, vertJacobiShape);
		Eigen::VectorXf r = Eigen::Map<Eigen::VectorXf>(m_verticesFinal.data(), 3 * m_vertexNum) - V_target;
		Eigen::MatrixXf H1 = vertJacobiShape.transpose() * vertJacobiShape;
		Eigen::VectorXf b1 = -vertJacobiShape.transpose() * r;
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(m_shapeNum, m_shapeNum);  // Leveberg Marquart
		Eigen::MatrixXf H_reg = DTD;
		Eigen::VectorXf b_reg = -m_shapeParam;
		float lambda = 0.001;
		float w1 = 1;
		float w_reg = 0.01;
		Eigen::MatrixXf H = H1 * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXf b = b1 * w1 + b_reg * w_reg;

		Eigen::VectorXf delta = H.ldlt().solve(b);
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
	Eigen::Vector3f barycenter_target = m_targetVSameTopo.rowwise().mean();
	Eigen::Vector3f barycenter_source = m_verticesFinal.rowwise().mean();

	m_translation = barycenter_target - barycenter_source;
	Eigen::MatrixXf V_target = m_targetVSameTopo.colwise() - barycenter_target;
	Eigen::MatrixXf V_source = m_verticesFinal.colwise() - barycenter_source;
	Eigen::Matrix3f S = V_source * V_target.transpose();
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
	std::cout << svd.singularValues() << std::endl;
	Eigen::MatrixXf U = svd.matrixU();
	Eigen::MatrixXf V = svd.matrixV();
	Eigen::Matrix3f R = V * U.transpose();
	Eigen::AngleAxisf ax(R);
	m_poseParam.segment<3>(0) = ax.axis() * ax.angle();
}

Eigen::MatrixXf PigSolver::getRegressedSkelTPose()
{
	int N = m_topo.joint_num;
	Eigen::MatrixXf skel = Eigen::MatrixXf::Zero(3, N);
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

std::vector<Eigen::Vector4f> PigSolver::projectBoxes()
{
	std::vector<Eigen::Vector4f> boxes; 
	for (int camid = 0; camid < m_cameras.size(); camid++)
	{
		double minx, miny, maxx, maxy;
		Eigen::MatrixXf points = m_cameras[camid].R * m_verticesFinal;
		points = points.colwise() + m_cameras[camid].T;
		points = m_cameras[camid].K * points; 
		for (int i = 0; i < m_vertexNum; i++)
		{
			float x = points(0, i) / points(2, i) * 1920;
			float y = points(1, i) / points(2, i) * 1080;
			if (i == 0)
			{
				minx = maxx = x; 
				miny = maxy = y;
			}
			else
			{
				minx = minx < x ? minx : x;
				maxx = maxx > x ? maxx : x;
				miny = miny < y ? miny : y;
				maxy = maxy > y ? maxy : y;
			}
		}
		if (maxx < 0 || maxy < 0 || minx >= 1920 || miny >= 1080)
		{
			boxes.emplace_back(Eigen::Vector4f::Zero()); 
		}
		else
		{
			minx = minx > 0 ? minx : 0;
			miny = miny > 0 ? miny : 0;
			maxx = maxx < 1920 ? maxx : 1920;
			maxy = maxy < 1080 ? maxy : 1080;
			boxes.emplace_back(Eigen::Vector4f(minx, miny, maxx, maxy));
		}
	}
	return boxes; 
}


float PigSolver::FitPoseToVerticesSameTopo(const int maxIterTime, const float terminal)
{
	Eigen::VectorXf V_target = Eigen::Map<Eigen::VectorXf>(m_targetVSameTopo.data(), 3 * m_vertexNum);

	int M = m_poseToOptimize.size();
	int N = m_topo.joint_num;

	float loss = 0;
	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();
		//std::stringstream ss; 
		//ss << "E:/debug_pig3/shapeiter/pose_" << iterTime << ".obj";
		//SaveObj(ss.str());

		Eigen::VectorXf r = Eigen::Map<Eigen::VectorXf>(m_verticesFinal.data(), 3 * m_vertexNum) - V_target;

		Eigen::VectorXf theta(3 + 3 * M);
		theta.segment<3>(0) = m_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
		}
		Eigen::VectorXf theta0 = theta;
		// solve
		Eigen::MatrixXf H_view;
		Eigen::VectorXf b_view;
		Eigen::MatrixXf J; // J_vert
		Eigen::MatrixXf J_joint;
		CalcPoseJacobiPartTheta(J_joint, J);
		Eigen::MatrixXf H1 = J.transpose() * J;
		Eigen::MatrixXf b1 = -J.transpose() * r;

		float lambda = 0.0001;
		float w1 = 1;
		float w_reg = 0.01;
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXf H_reg = DTD;  // reg term 
		Eigen::VectorXf b_reg = -theta; // reg term 

		Eigen::MatrixXf H = H1 * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXf b = b1 * w1 + b_reg * w_reg;

		Eigen::VectorXf delta = H.ldlt().solve(b);

		// update 
		m_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
		}
		// if(iterTime == 1) break; 
		loss = delta.norm();
		if (loss < terminal) break;
	}
	return loss;
}

float PigSolver::FitPoseToJointsSameTopo(Eigen::MatrixXf target)
{
	int maxIterTime = 100;
	float terminal = 0.0001f;
	TimerUtil::Timer<std::chrono::milliseconds> TT;

	int M = m_poseToOptimize.size();
	float loss = 0;
	int iterTime = 0;
	Eigen::VectorXf target_vec = Eigen::Map<Eigen::VectorXf>(target.data(), m_jointNum * 3);

	TT.Start();
	for (; iterTime < maxIterTime; iterTime++)
	{
		UpdateJoints();
		Eigen::VectorXf r = Eigen::Map<Eigen::VectorXf>(m_jointsFinal.data(), 3 * m_jointNum) - target_vec;

		Eigen::VectorXf theta(3 + 3 * M);
		theta.segment<3>(0) = m_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
		}
		Eigen::VectorXf theta0 = theta;
		// solve
		Eigen::MatrixXf H_view;
		Eigen::VectorXf b_view;
		Eigen::MatrixXf J_vert; // J_vert
		Eigen::MatrixXf J_joint;
		CalcPoseJacobiPartTheta(J_joint, J_vert, false);
		Eigen::MatrixXf H1 = J_joint.transpose() * J_joint;
		Eigen::MatrixXf b1 = -J_joint.transpose() * r;

		float lambda = 0.0001;
		float w1 = 1;
		float w_reg = 0.01;
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXf H_reg = DTD;  // reg term 
		Eigen::VectorXf b_reg = -theta; // reg term 

		Eigen::MatrixXf H = H1 * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXf b = b1 * w1 + b_reg * w_reg;

		Eigen::VectorXf delta = H.ldlt().solve(b);

		// update 
		m_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
		}
		// if(iterTime == 1) break; 
		loss = delta.norm();
		if (loss < terminal) break;
	}

	float time = TT.Elapsed();

	std::cout << "iter : " << iterTime << "  loss: " << loss << "  tpi: " << time / iterTime << std::endl;
	return loss;
}



void PigSolver::FitPoseToVerticesSameTopoLatent()
{
	int maxIterTime = 100;
	float terminal = 0.0001;
	Eigen::VectorXf V_target = Eigen::Map<Eigen::VectorXf>(m_targetVSameTopo.data(), 3 * m_vertexNum);

	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		std::cout << "iter: " << iterTime << std::endl;
		UpdateVertices();
		std::stringstream ss;
		ss << "G:/debug_pig4/poseiter/pose_" << iterTime << ".obj";
		SaveObj(ss.str());

		Eigen::VectorXf r = Eigen::Map<Eigen::VectorXf>(m_verticesFinal.data(), 3 * m_vertexNum) - V_target;

		Eigen::VectorXf theta(3 + 3 + 32);
		theta.segment<3>(0) = m_translation;
		theta.segment<3>(3) = m_poseParam.segment<3>(0);
		theta.segment<32>(6) = m_latentCode;

		Eigen::VectorXf theta0 = theta;

		Eigen::MatrixXf J_joint, J_vert;
		CalcPoseJacobiLatent(J_joint, J_vert);
		Eigen::MatrixXf H1 = J_vert.transpose() * J_vert;
		Eigen::MatrixXf b1 = -J_vert.transpose() * r;

		float lambda = 0.001;
		float w1 = 1;
		float w_reg = 0.01;
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(38, 38);
		Eigen::MatrixXf H_reg = DTD;  // reg term 
		Eigen::VectorXf b_reg = -theta; // reg term 

		Eigen::MatrixXf H = H1 * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXf b = b1 * w1 + b_reg * w_reg;

		Eigen::VectorXf delta = H.ldlt().solve(b);

		// update 
		m_translation += delta.segment<3>(0);
		m_poseParam.segment<3>(0) += delta.segment<3>(3);
		m_latentCode += delta.segment<32>(6);

		// if(iterTime == 1) break; 
		if (delta.norm() < terminal) break;
	}
}