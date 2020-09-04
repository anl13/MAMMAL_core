#include "pigsolverdevice.h"
#include <json/json.h> 

PigSolverDevice::PigSolverDevice(const std::string& _configFile)
	:PigModelDevice(_configFile)
{
	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(_configFile);
	if (!instream.is_open())
	{
		std::cout << "can not open " << _configFile << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}

	std::string topo_type = root["topo"].asString();
	m_skelTopo = getSkelTopoByType(topo_type);
	m_poseToOptimize.clear();
	for (const auto& c : root["pose_to_solve"])
	{
		m_poseToOptimize.push_back(c.asInt());
	}


	// read optim pair;
	m_skelCorr.clear();
	for (auto const&c : root["optimize_pair"])
	{
		CorrPair pair;
		pair.target = c[0].asInt();
		pair.type = c[1].asInt();
		pair.index = c[2].asInt();
		pair.weight = c[3].asDouble();
		m_skelCorr.push_back(pair);
	}

	instream.close();

	// pre-allocate device memory 
	m_initScale = false; 
	m_host_paramLines.resize(m_poseToOptimize.size() * 3 + 3);
	m_host_paramLines[0] = 0; 
	m_host_paramLines[1] = 1;
	m_host_paramLines[2] = 2; 
	for (int i = 0; i < m_poseToOptimize.size(); i++)
	{
		int id = m_poseToOptimize[i];
		for (int k = 0; k < 3; k++) m_host_paramLines[3 + 3 * i + k] = 3 + 3 * id + k;
	}
	m_device_paramLines.upload(m_host_paramLines); 

	int paramNum = m_host_paramLines.size(); 
	d_depth_renders.resize(10); 
	int H = WINDOW_HEIGHT;
	int W = WINDOW_WIDTH;
	for (int i = 0; i < 10; i++)
		cudaMalloc((void**)&d_depth_renders[i], H * W * sizeof(float)); 
	d_det_mask.create(H, W);
	d_det_sdf.create(H, W);
	d_rend_sdf.create(H, W); 
	d_det_gradx.create(H, W); 
	d_det_grady.create(H, W);
	d_const_distort_mask.create(H, W); 
	d_const_scene_mask.create(H, W); 
	d_J_vert.create(paramNum, 3 * m_vertexNum); 
	d_J_joint.create(paramNum, 3 * m_jointNum); 
	d_JT_sil.create(paramNum, m_vertexNum);
	d_J_joint_full.create(3 * m_jointNum + 3, 3 * m_jointNum);
	d_J_vert_full.create(3 * m_jointNum + 3, 3 * m_vertexNum);
	d_r_sil.create(m_vertexNum); 

	d_ATA_sil.create(paramNum, paramNum);
	d_ATb_sil.create(paramNum); 
}

PigSolverDevice::~PigSolverDevice()
{
	for (int i = 0; i < 10; i++)cudaFree(d_depth_renders[i]); 
	d_depth_renders.clear();
	d_det_mask.release();
	d_det_sdf.release();
	d_rend_sdf.release(); 
	d_det_gradx.release(); 
	d_det_grady.release();
	d_const_distort_mask.release(); 
	d_const_scene_mask.release(); 
	d_J_vert.release();
	d_J_joint.release();
	d_JT_sil.release(); 
	d_J_joint_full.release();
	d_J_vert_full.release(); 
	d_r_sil.release(); 
	d_ATA_sil.release();
	d_ATb_sil.release(); 
}

// Only point with more than 1 observations could be triangulated
void PigSolverDevice::directTriangulationHost()
{
	int N = m_skelTopo.joint_num;
	m_skel3d.resize(N, Eigen::Vector3f::Zero()); 
	for (int i = 0; i < N; i++)
	{
		Eigen::Vector3f X = Eigen::Vector3f::Zero(); // joint position to solve.  
		int validnum = 0;
		for (int k = 0; k < m_source.view_ids.size(); k++)
		{
			if (m_source.dets[k].keypoints[i](2) < m_skelTopo.kpt_conf_thresh[i]) continue;
			validnum++;
		}
		if (validnum < 2)
		{
			m_skel3d[i] = X;
			continue;
		}

		// usually, converge in 3 iteractions 
		for (int iter = 0; iter < 10; iter++)
		{
			Eigen::Matrix3f H1 = Eigen::Matrix3f::Zero();
			Eigen::Vector3f b1 = Eigen::Vector3f::Zero();
			for (int k = 0; k < m_source.view_ids.size(); k++)
			{
				int view = m_source.view_ids[k];
				Camera cam = m_cameras[view];
				Eigen::Vector3f keypoint = m_source.dets[k].keypoints[i];
				if (keypoint(2) < m_skelTopo.kpt_conf_thresh[i]) continue;
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

			}
		}
		m_skel3d[i] = X;
	}
}

// regress skel points from Surface model. 
std::vector<Eigen::Vector3f> PigSolverDevice::getRegressedSkel()
{
	int N = m_skelTopo.joint_num;
	std::vector<Eigen::Vector3f> skel(N, Eigen::Vector3f::Zero());
	for (int i = 0; i < m_skelCorr.size(); i++)
	{
		CorrPair P = m_skelCorr[i];
		int t = P.target;
		if (P.type == 1)
		{
			skel[t] += m_host_verticesPosed[P.index] * P.weight;
		}
		else if (P.type == 0)
		{
			skel[t] += m_host_jointsPosed[P.index] * P.weight;
		}
	}
	return skel;
}

// 1. compute scale 
// 2. align T
// 3. align R 
void PigSolverDevice::globalAlign()
{
	directTriangulationHost(); 
	if (m_initScale) return; 
	int N = m_skelTopo.joint_num;

	std::vector<float> weights(N, 0); 
	std::vector<Eigen::Vector3f> skelReg = getRegressedSkel();

	// step1: compute scale by averaging bone length. 
	std::vector<float> regBoneLens; 
	std::vector<float> triBoneLens;
	for (int bid = 0; bid < m_skelTopo.bones.size(); bid++)
	{
		int sid = m_skelTopo.bones[bid](0); 
		int eid = m_skelTopo.bones[bid](1); 
		if (m_skel3d[sid].norm() < 0.0001f || m_skel3d[eid].norm() < 0.0001f) continue; 
		float regLen = (skelReg[sid] - skelReg[eid]).norm(); 
		float triLen = (m_skel3d[sid] - m_skel3d[eid]).norm(); 
		regBoneLens.push_back(regLen);
		triBoneLens.push_back(triLen);
	}
	float a = 0;
	float b = 0; 
	for (int i = 0; i < regBoneLens.size(); i++)
	{
		a += triBoneLens[i] * regBoneLens[i];
		b += regBoneLens[i] * regBoneLens[i];
	}
	float alpha = a / b;
	m_host_scale = alpha * m_host_scale; 
	UpdateVertices(); 

	// step2: compute translation of root joint. (here, tail root is root joint) 
	skelReg = getRegressedSkel(); 
	Eigen::Vector3f regCenter = skelReg[18];
	Eigen::Vector3f triCenter = m_skel3d[18];
	m_host_translation = triCenter - regCenter; 
}

void PigSolverDevice::fitPoseToVSameTopo(
	const std::vector<Eigen::Vector3f> &_tv
)
{
	pcl::gpu::DeviceArray<Eigen::Vector3f> target_device; 
	target_device.upload(_tv); 
	int M = m_poseToOptimize.size(); 
	int maxIterTime = 100;
	float terminal = 0.0001; 

	Eigen::VectorXf theta0(3 + 3 * M); 
	theta0.segment<3>(0) = m_host_translation;
	for (int i = 0; i < M; i++) theta0.segment<3>(3 + 3 * i) = m_host_poseParam[m_poseToOptimize[i]];

	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices(); 
		Eigen::VectorXf theta(3 + 3 * M);
		theta.segment<3>(0) = m_host_translation;
		for (int i = 0; i < M; i++) theta.segment<3>(3 + 3 * i) = m_host_poseParam[m_poseToOptimize[i]];

		pcl::gpu::DeviceArray2D<float> AT_joint, AT_vert, ATA;
		pcl::gpu::DeviceArray<float> ATb; 
		calcPoseJacobiPartTheta_device(AT_joint, AT_vert); 
		computeATA_device(AT_vert, ATA);

		computeATb_device(AT_vert, m_device_verticesPosed, target_device, ATb);

		Eigen::MatrixXf ATA_eigen = Eigen::MatrixXf::Zero(3 + 3 * M, 3 + 3 * M); 
		Eigen::VectorXf ATb_eigen = Eigen::VectorXf::Zero(3 + 3 * M);
		ATA.download(ATA_eigen.data(), ATA_eigen.rows()*sizeof(float)); 

		ATb.download(ATb_eigen.data()); 

		float lambda = 0.0001;
		float w1 = 1;
		float w_reg = 0.01; 
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::VectorXf b_reg = -theta; 
		Eigen::MatrixXf H = ATA_eigen * w1 + DTD * w_reg + DTD * lambda;
		Eigen::VectorXf b = ATb_eigen * w1 + b_reg * w_reg; 
		Eigen::VectorXf delta = H.ldlt().solve(b); 
		
		m_host_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)m_host_poseParam[m_poseToOptimize[i]] += delta.segment<3>(3 + 3 * i);
		
		float grad_norm = delta.norm(); 
		if (grad_norm < terminal) break;
	}

}

std::vector<Eigen::Vector3f> 
PigSolverDevice::getRegressedSkel_host()
{
	int N = m_skelTopo.joint_num; 
	std::vector<Eigen::Vector3f> skels(N, Eigen::Vector3f::Zero()); 
	for (int i = 0; i < m_skelCorr.size(); i++)
	{
		const CorrPair& P = m_skelCorr[i];
		if (P.type == 1)
		{
			skels[P.target] += m_host_verticesPosed[P.index] * P.weight;
		}
		else
		{
			skels[P.target] += m_host_jointsPosed[P.index] * P.weight;
		}
	}
	return skels; 
}

void PigSolverDevice::calcSkelJacobiPartTheta_host(Eigen::MatrixXf& J)
{
	Eigen::MatrixXf jointJacobiPose, J_joint;
	int N = m_skelCorr.size();
	int M = m_poseToOptimize.size();
	J = Eigen::MatrixXf::Zero(3 * N, 3 + 3 * M);

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
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx) * m_host_localSE3[child].block<3, 3>(0, 0);
		}
	}

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
			affine = jointDerivativeId == 0 ? affine : (m_host_globalSE3[m_host_parents[jointDerivativeId]] * affine);

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

	J_joint.resize(3 * m_jointNum, 3 + 3 * M);
	for (int i = 0; i < m_jointNum; i++)
	{
		J_joint.block<3, 3>(3 * i, 0) = jointJacobiPose.block<3, 3>(3 * i, 0);
		for (int k = 0; k < M; k++)
		{
			int thetaidx = m_poseToOptimize[k];
			J_joint.block<3, 3>(3 * i, 3 + 3 * k) = jointJacobiPose.block<3, 3>(3 * i, 3 + 3 * thetaidx);
		}
	}

	Eigen::MatrixXf block = Eigen::MatrixXf::Zero(3, 3 + 3 * m_jointNum);
	for (int i = 0; i < N; i++)
	{
		CorrPair P = m_skelCorr[i];

		if (P.type == 0)
		{
			J.middleRows(3 * i, 3) =
				J_joint.middleRows(3 * P.index, 3);
			continue;
		}
		if (P.type == 1)
		{
			block.setZero();
			int vIdx = P.index;
			Eigen::Vector3f v0 = m_host_verticesDeformed[vIdx];
			block.setZero();
			for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
			{
				if (m_host_lbsweights(jIdx, vIdx) < 0.00001) continue;
				Eigen::Vector3f j0 = m_host_jointsDeformed[jIdx];
				block += (m_host_lbsweights(jIdx, vIdx) * jointJacobiPose.middleRows(3 * jIdx, 3));
				for (int pIdx = jIdx; pIdx > -1; pIdx = m_host_parents[pIdx]) // poseParamIdx to derive
				{
					Eigen::Matrix3f T = Eigen::Matrix3f::Zero();
					for (int axis_id = 0; axis_id < 3; axis_id++)
					{
						Eigen::Matrix3f dR1 = RP.block<3, 3>(9 * pIdx + 3 * axis_id, 0) * LP.block<3, 3>(3 * jIdx, 3 * pIdx);
						T.col(axis_id) = dR1 * (v0 - j0) * m_host_lbsweights(jIdx, vIdx);
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


void PigSolverDevice::calcPose2DTerm_host(
	const DetInstance& det, 
	const Camera& cam, 
	const std::vector<Eigen::Vector3f> & skel2d,
	const Eigen::MatrixXf& Jacobi3d,
	Eigen::MatrixXf& ATA,
	Eigen::VectorXf& ATb
)
{
	Eigen::Matrix3f R = cam.R;
	Eigen::Matrix3f K = cam.K;
	Eigen::Vector3f T = cam.T;

	int N = m_skelCorr.size();
	int M = m_poseToOptimize.size();
	Eigen::VectorXf r = Eigen::VectorXf::Zero(2 * N);
	Eigen::MatrixXf J;

	J = Eigen::MatrixXf::Zero(2 * N, 3 + 3 * M);

	for (int i = 0; i < N; i++)
	{
		const CorrPair& P = m_skelCorr[i];
		int t = P.target;
		if (det.keypoints[t](2) < m_skelTopo.kpt_conf_thresh[t]) continue;

		Eigen::Vector3f x_local = K * (R * skel2d[t] + T);
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
}

void PigSolverDevice::optimizePose()
{
	int maxIterTime = 50; 
	float terminal = 0.0001f; 
	
	int M = m_poseToOptimize.size();
	int paramNum = 3 + 3 * M;

	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();

		Eigen::VectorXf theta(paramNum);
		theta.segment<3>(0) = m_host_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_host_poseParam[jIdx];
			
		}

		// data term
		Eigen::MatrixXf ATA_data = Eigen::MatrixXf::Zero(paramNum, paramNum); // data term 
		Eigen::VectorXf ATb_data = Eigen::VectorXf::Zero(paramNum);  // data term 
		Calc2dJointProjectionTerm(m_source, ATA_data, ATb_data); 

		float lambda = 0.0005; // LM algorhtim
		float w_data = 1;
		float w_reg = 0.01;
		float w_temp = 0.0;
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(paramNum, paramNum);
		Eigen::MatrixXf ATA_reg = DTD;  // reg term 
		Eigen::VectorXf ATb_reg = -theta; // reg term 

		Eigen::MatrixXf ATA = ATA_data * w_data + ATA_reg * w_reg + DTD * lambda;
		Eigen::VectorXf ATb = ATb_data * w_data + ATb_reg * w_reg;

		Eigen::VectorXf delta = ATA.ldlt().solve(ATb);

		// update 

		m_host_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_host_poseParam[jIdx] += delta.segment<3>(3 + 3 * i);
		}
		
		if (delta.norm() < terminal) break;
	}
}

void PigSolverDevice::normalizeCamera()
{
	for (int i = 0; i < m_cameras.size(); i++)
	{
		m_cameras[i].NormalizeK();
	}
}

void PigSolverDevice::normalizeSource()
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

void PigSolverDevice::Calc2dJointProjectionTerm(
	const MatchedInstance& source,
	Eigen::MatrixXf& ATA_data, Eigen::VectorXf& ATb_data)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size(); 
	Eigen::MatrixXf poseJ3d;
	calcSkelJacobiPartTheta_host(poseJ3d);

	std::vector<Eigen::Vector3f> skel2d = getRegressedSkel_host();

	// data term
	ATA_data = Eigen::MatrixXf::Zero(paramNum, paramNum); // data term 
	ATb_data = Eigen::VectorXf::Zero(paramNum);  // data term 
	for (int k = 0; k < source.view_ids.size(); k++)
	{
		Eigen::MatrixXf H_view;
		Eigen::VectorXf b_view;
		int view = source.view_ids[k];
		calcPose2DTerm_host(source.dets[k], m_cameras[view], skel2d, poseJ3d, H_view, b_view);

		ATA_data += H_view;
		ATb_data += b_view;
	}
}

void PigSolverDevice::renderDepths()
{
	// render images 
	mp_renderEngine->clearAllObjs();
	RenderObjectMesh* p_model = new RenderObjectMesh();
	p_model->SetVertices(m_host_verticesPosed);
	p_model->SetFaces(m_host_facesVert);
	p_model->SetNormal(m_host_normalsFinal);
	p_model->SetColors(m_host_normalsFinal);
	mp_renderEngine->meshObjs.push_back(p_model);

	const auto& cameras = m_cameras;

	for (int view = 0; view < m_rois.size(); view++)
	{
		Camera cam = m_rois[view].cam;
		Eigen::Matrix3f R = cam.R.cast<float>();
		Eigen::Vector3f T = cam.T.cast<float>();
		mp_renderEngine->s_camViewer.SetExtrinsic(R, T);

		float * depth_device = mp_renderEngine->renderDepthDevice();
		cudaMemcpy(d_depth_renders[view], depth_device, WINDOW_WIDTH*WINDOW_HEIGHT * sizeof(float),
			cudaMemcpyDeviceToDevice);
	}
}

#define DEBUG_SIL
void PigSolverDevice::optimizePoseSilhouette(
	const std::vector<ROIdescripter>& rois,
	int maxIter)
{
	int iter = 0;

#ifdef DEBUG_SIL
	std::vector<cv::Mat> color_mask_dets;
	for (int view = 0; view < rois.size(); view++)
	{
		int camid = rois[view].viewid;
		cv::Mat img(cv::Size(1920, 1080), CV_8UC3); img.setTo(255);
		my_draw_mask(img, rois[view].mask_list, Eigen::Vector3i(255, 0, 0), 0);
		color_mask_dets.push_back(img);
	}
#endif 

	int M = m_poseToOptimize.size();
	int paramNum = 3 + 3 * M; 

	Eigen::VectorXf theta_last(paramNum);
	theta_last.segment<3>(0) = m_host_translation;
	for (int i = 0; i < M; i++)
	{
		int jIdx = m_poseToOptimize[i];
		theta_last.segment<3>(3 + 3 * i) = m_host_poseParam[jIdx];
	}


	for (; iter < maxIter; iter++)
	{
		UpdateVertices();
		UpdateNormalFinal();

		Eigen::VectorXf theta(paramNum);
		theta.segment<3>(0) = m_host_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_host_poseParam[jIdx];
		}

		// calc joint term 
		Eigen::MatrixXf ATA_data; 
		Eigen::VectorXf ATb_data; 
		Calc2dJointProjectionTerm(m_source, ATA_data, ATb_data); 

		renderDepths(); 


#ifdef DEBUG_SIL
		// test 
		//std::vector<cv::Mat> pseudos;
		//for (int view = 0; view < renders.size(); view++)
		//{
		//	cv::Mat colored = pseudoColor(renders[view]);
		//	pseudos.push_back(colored);
		//}
		//cv::Mat pack_pseudo;
		//packImgBlock(pseudos,pack_pseudo);
		//std::stringstream ss_pseudo;
		//ss_pseudo << "G:/pig_results/debug/" << std::setw(6) << std::setfill('0') << iter << "_depth.jpg";
		//cv::imwrite(ss_pseudo.str(), pack_pseudo); 


		//cv::Mat blended;
		//cv::Mat pack_det;
		//packImgBlock(color_mask_dets, pack_det);
		//blended = pack_render * 0.5 + pack_det * 0.5;
		//std::stringstream ss;
		//ss << "G:/pig_results/debug/" << std::setw(6) << std::setfill('0')
		//	<< iter << ".jpg";
		//cv::imwrite(ss.str(), blended);
#endif 
		// compute terms
		Eigen::MatrixXf ATA_sil;
		Eigen::VectorXf ATb_sil;
		CalcSilhouettePoseTerm(m_rois, d_depth_renders, ATA_sil, ATb_sil);

		float lambda = 0.005;
		float w_data = 0.01;
		float w_sil = 1;
		float w_reg = 1;
		float w_temp = 0;
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXf H_reg = DTD;  // reg term 
		Eigen::VectorXf b_reg = -theta; // reg term 
		Eigen::MatrixXf H_temp = DTD;
		Eigen::VectorXf b_temp = theta_last - theta;
		Eigen::MatrixXf H = ATA_sil * w_sil + H_reg * w_reg
			+ DTD * lambda + H_temp * w_temp
			+ ATA_data * w_data;
		Eigen::VectorXf b = ATb_sil * w_sil + b_reg * w_reg
			+ b_temp * w_temp
			+ ATb_data * w_data;
		Eigen::VectorXf delta = H.ldlt().solve(b);

		std::cout << "reg  term b: " << b_reg.norm() << std::endl;
		std::cout << "temp term b: " << b_temp.norm() << std::endl;

		// update 
		m_host_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_host_poseParam[jIdx] += delta.segment<3>(3 + 3 * i);
		}
	}
}

