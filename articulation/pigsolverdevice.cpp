#include "pigsolverdevice.h"
#include <json/json.h> 
#include "../utils/image_utils_gpu.h"
#include <cuda_profiler_api.h>
#include "../utils/geometry.h"

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


	m_poseToOptimize.clear();
	for (const auto& c : root["pose_to_solve"])
	{
		m_poseToOptimize.push_back(c.asInt());
	}

	m_valid_threshold = root["valid_threshold"].asFloat();
	m_lambda = root["lambda"].asFloat();
	m_w_data_term = root["data_term"].asFloat();
	m_w_sil_term = root["sil_term"].asFloat();
	m_w_reg_term = root["reg_term"].asFloat();
	m_w_temp_term = root["temp_term"].asFloat();
	m_w_floor_term = root["floor_term"].asFloat();
	m_kpt_track_dist = root["kpt_track_dist"].asFloat();
	m_w_anchor_term = root["anchor_term"].asFloat();
	m_use_gpu = root["use_gpu"].asBool(); 
	m_iou_thres = root["iou_thres"].asFloat(); 
	m_anchor_folder = root["anchor_folder"].asString();

	gt_scales.resize(4); 
	for (int i = 0; i < 4; i++)
	{
		gt_scales[i] = root["scales"][i].asFloat();
	}

	m_param_reg_weight.resize(m_poseToOptimize.size()*3+3, 1);
	for (auto const &c : root["reg_weights"])
	{
		int index = c[0].asInt(); 
		float wx = c[1].asFloat(); 
		float wy = c[2].asFloat(); 
		float wz = c[3].asFloat(); 
		for (int i = 0; i < m_poseToOptimize.size(); i++)
		{
			int jointid = m_poseToOptimize[i];
			if (jointid == index)
			{
				m_param_reg_weight[3 + 3 * i] = wx;
				m_param_reg_weight[3 + 3 * i + 1] = wy;
				m_param_reg_weight[3 + 3 * i + 2] = wz;
			}
		}
	}

	instream.close();
	m_visRegressorList.resize(m_jointNum);
	std::ifstream visRegFile(m_folder + "/vis_regressor.txt");
	if (visRegFile.is_open())
		for (int i = 0; i < m_jointNum; i++)
		{
			int id, num;
			visRegFile >> id >> num;
			if (num == 0)continue;
			m_visRegressorList[i].resize(num);
			for (int k = 0; k < num; k++)
			{
				visRegFile >> m_visRegressorList[i][k];
			}
		}
	visRegFile.close();

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
	d_depth_renders_interact.resize(10); 
	int H = WINDOW_HEIGHT;
	int W = WINDOW_WIDTH;
	for (int i = 0; i < 10; i++)
	{
		cudaMalloc((void**)&d_depth_renders[i], H * W * sizeof(float));
		cudaMalloc((void**)&d_depth_renders_interact[i], H * W * sizeof(float));
	}
	d_J_vert.create(paramNum, 3 * m_vertexNum); 
	d_J_joint.create(paramNum, 3 * m_jointNum); 
	
	d_J_joint_full.create(3 * m_jointNum + 3, 3 * m_jointNum);
	d_J_vert_full.create(3 * m_jointNum + 3, 3 * m_vertexNum);

	h_J_joint.resize(3 * m_jointNum, paramNum); 
	h_J_vert.resize(3 * m_vertexNum, paramNum); 

	d_RP.create(3, 9 * m_jointNum); 
	d_LP.create(3 * m_jointNum, 3 * m_jointNum); 

	d_ATA_sil.create(paramNum, paramNum);
	d_ATb_sil.create(paramNum); 

	init_backgrounds = false; 

	cudaMalloc((void**)&d_middle_mask, H/2*W /2* sizeof(uchar));

	if (m_use_gpu)
	{
		cudaMalloc((void**)&d_rend_sdf, H / 2 * W / 2 * sizeof(float));
		cudaMalloc((void**)&d_const_distort_mask, H * W * sizeof(uchar));
		d_const_scene_mask.resize(10);
		d_det_mask.resize(10);
		d_det_sdf.resize(10);
		d_det_gradx.resize(10);
		d_det_grady.resize(10);

		for (int i = 0; i < 10; i++)
		{
			cudaMalloc((void**)&d_const_scene_mask[i], H*W * sizeof(float));
			cudaMalloc((void**)&d_det_mask[i], H*W * sizeof(float));
			cudaMalloc((void**)&d_det_sdf[i], W / 2 * H / 2 * sizeof(float));
			cudaMalloc((void**)&d_det_gradx[i], W / 2 * H / 2 * sizeof(float));
			cudaMalloc((void**)&d_det_grady[i], W / 2 * H / 2 * sizeof(float));
		}
	}

	d_AT_sil.create(paramNum, m_vertexNum); 
	d_b_sil.create(m_vertexNum); 

	m_last_thetas.resize(paramNum); 
	m_last_thetas.segment<3>(0) = m_host_translation;
	for (int i = 0; i < m_poseToOptimize.size(); i++)
	{
		int jIdx = m_poseToOptimize[i];
		m_last_thetas.segment<3>(3 + 3 * i) = m_host_poseParam[jIdx];
	}

	m_param_temp_weight = Eigen::VectorXf::Ones(paramNum); 
	m_param_observe_num = Eigen::VectorXf::Zero(m_skelTopo.joint_num); 

	m_scaleCount = 0.f; 
	// read anchor data 
	m_host_translation.setZero();
	for (int i = 0; i < m_jointNum; i++)m_host_poseParam[i].setZero(); 
	UpdateVertices(); 

	m_det_confs.resize(m_skelTopo.joint_num, 0); 

	m_anchor_lib.load(m_anchor_folder);

	m_isReAssoc = false;
}

PigSolverDevice::~PigSolverDevice()
{
	for (int i = 0; i < 10; i++)
	{
		cudaFree(d_depth_renders_interact[i]);
		cudaFree(d_depth_renders[i]);
	}
	d_depth_renders.clear();
	d_depth_renders_interact.clear(); 
	
	d_J_vert.release();
	d_J_joint.release();
	d_J_joint_full.release();
	d_J_vert_full.release(); 
	d_ATA_sil.release();
	d_ATb_sil.release(); 
	d_RP.release(); 
	d_LP.release();

	cudaFree(d_middle_mask); 
	cudaFree(d_rend_sdf); 
	cudaFree(d_const_distort_mask); 

	if (m_use_gpu)
	{
		for (int i = 0; i < 10; i++)
		{
			cudaFree(d_const_scene_mask[i]);
			cudaFree(d_det_mask[i]);
			cudaFree(d_det_sdf[i]);
			cudaFree(d_det_gradx[i]);
			cudaFree(d_det_grady[i]);
		}
		d_const_scene_mask.clear();
		d_det_sdf.clear();
		d_det_gradx.clear();
		d_det_grady.clear();
		d_det_mask.clear();
	}
}

// Only point with more than 1 observations could be triangulated
std::vector<Eigen::Vector3f> PigSolverDevice::directTriangulationHost()
{
	for (int i = 0; i < m_skelTopo.joint_num; i++) m_det_confs[i] = 0; 
	int N = m_skelTopo.joint_num;
	std::vector<Eigen::Vector3f> skel; 
	skel.resize(N, Eigen::Vector3f::Zero()); 
	for (int i = 0; i < N; i++)
	{
		Eigen::Vector3f X = Eigen::Vector3f::Zero(); // joint position to solve.  
		int validnum = 0;
		for (int k = 0; k < m_source.view_ids.size(); k++)
		{
			if (m_source.dets[k].keypoints[i](2) < m_skelTopo.kpt_conf_thresh[i]) continue;
			validnum++;
		}
		m_det_confs[i] = validnum; 
		if (validnum < 2)
		{
			skel[i] = X;
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
		skel[i] = X;
	}
	return skel; 
}

float PigSolverDevice::computeScale()
{
	m_skel3d = directTriangulationHost();
	//if (m_initScale) return; 
	int N = m_skelTopo.joint_num;

	std::vector<Eigen::Vector3f> skelReg = getRegressedSkel_host();

	//m_host_scale = gt_scales[m_pig_id];
	//m_initScale = true;

	// step1: compute scale by averaging bone length. 
	std::vector<float> regBoneLens;
	std::vector<float> triBoneLens;
	std::vector<Eigen::Vector2i> bones = {
		/*{0,1}, {0,2}, {1,2}, {1,3}, {2,4}, {3,4}, {1,4}, {2,3},*/
		{0,3},{0,4},{3,4},
	{11,13}, {13,15}, {12,14},{14,16},{18,11}, {18,12}
	};
	for (int bid = 0; bid < bones.size(); bid++)
	{
		int sid = bones[bid](0);
		int eid = bones[bid](1);
		if (m_skel3d[sid].norm() < 0.0001f || m_skel3d[eid].norm() < 0.0001f) continue;
		float regLen = (skelReg[sid] - skelReg[eid]).norm();
		float triLen = (m_skel3d[sid] - m_skel3d[eid]).norm();
		if (bid < 3) triLen *= 1;
		regBoneLens.push_back(regLen);
		triBoneLens.push_back(triLen);
	}
	float a = 0;
	float b = 0;
	for (int i = 0; i < regBoneLens.size(); i++)
	{
		std::cout << " -- " << i <<  " -- " << triBoneLens[i] / regBoneLens[i] << std::endl; 
		a += triBoneLens[i] * regBoneLens[i];
		b += regBoneLens[i] * regBoneLens[i];
	}
	float alpha;
	if (a == 0 || b == 0) alpha = m_host_scale;
	else alpha = a / b;

	return alpha; 
}
// 1. compute scale 
// 2. align T
// 3. align R 
void PigSolverDevice::globalAlign()
{
	m_skel3d = directTriangulationHost(); 
	if (m_initScale) return;

	int N = m_skelTopo.joint_num;

	std::vector<float> weights(N, 0); 
	std::vector<Eigen::Vector3f> skelReg = getRegressedSkel_host();

	m_host_scale = gt_scales[m_pig_id];
	float alpha = computeScale(); 
	std::cout << "m_pig: " << m_pig_id << "  alpha: " << alpha << std::endl; 
	m_initScale = true;

	//std::cout << "pig: " << m_pig_id << " scale: " << alpha << std::endl;
	//// running average
	//if (!m_initScale)
	//{
	//	m_host_scale = (m_host_scale * m_scaleCount + alpha) / (m_scaleCount + 1);
	//	m_initScale = true; 
	//}
	//m_scaleCount += 1.f; 
	UpdateVertices();

	// step2: compute translation of root joint. (here, tail root is root joint) 
	skelReg = getRegressedSkel_host(); 
	Eigen::Vector3f regCenter = skelReg[18];
	Eigen::Vector3f triCenter = m_skel3d[18];
	if (triCenter.norm() > 0)
	{
		m_host_translation = triCenter - regCenter;
		UpdateVertices();
	}

	computeDepthWeight(); 
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

		pcl::gpu::DeviceArray2D<float> ATA;
		pcl::gpu::DeviceArray<float> ATb; 
		calcPoseJacobiPartTheta_device(d_J_joint, d_J_vert, true);

		computeATA_device(d_J_vert, ATA);

		computeATb_device(d_J_vert, m_device_verticesPosed, target_device, ATb);

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


void PigSolverDevice::fitPoseToJointSameTopo(
	const std::vector<Eigen::Vector3f> &_jointstarget
)
{
	int maxIterTime = 100;
	float terminal = 0.0001f;
	TimerUtil::Timer<std::chrono::milliseconds> TT;

	int M = m_poseToOptimize.size();
	float loss = 0;
	int iterTime = 0;

	TT.Start();
	for (; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();
		Eigen::VectorXf theta(3 + 3 * M);
		theta.segment<3>(0) = m_host_translation;
		for (int i = 0; i < M; i++) theta.segment<3>(3 + 3 * i) = m_host_poseParam[m_poseToOptimize[i]];

		calcPoseJacobiPartTheta_device(d_J_joint, d_J_vert, true);
		d_J_joint.download(h_J_joint.data(), 3 * m_jointNum * sizeof(float));

		

		Eigen::MatrixXf ATA_eigen = Eigen::MatrixXf::Zero(3 + 3 * M, 3 + 3 * M);
		Eigen::VectorXf ATb_eigen = Eigen::VectorXf::Zero(3 + 3 * M);
		calcJoint3DTerm_host(h_J_joint, _jointstarget, ATA_eigen, ATb_eigen);
		
		float lambda = 0.0001;
		float w1 = 1;
		float w_reg = 0.00001;
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(3 + 3 * M, 3 + 3 * M);
		
		Eigen::MatrixXf ATA_reg; 
		Eigen::VectorXf ATb_reg;
		CalcRegTerm(theta, ATA_reg, ATb_reg); 

		Eigen::MatrixXf ATA = ATA_eigen * w1 + ATA_reg * w_reg + DTD * lambda;
		Eigen::VectorXf ATb = ATb_eigen * w1 + ATb_reg * w_reg;
		Eigen::VectorXf delta = ATA.ldlt().solve(ATb);

		m_host_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)m_host_poseParam[m_poseToOptimize[i]] += delta.segment<3>(3 + 3 * i);
		loss = ATb.norm(); 
		float grad_norm = delta.norm();
		if (grad_norm < terminal) break;
	}

	float time = TT.Elapsed();

	//std::cout << "iter : " << iterTime << "  ATb: " << loss << "  tpi: " << time / iterTime << std::endl;
	return;
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
		if (jointId == 0)
		{
			rodriguesDerivative.block<3, 9>(0, 0) = EulerJacobiF(pose); 
			//rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiF(pose);

		}
		else 
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
	int camid,
	const std::vector<Eigen::Vector3f> & skel3d,
	const Eigen::MatrixXf& Jacobi3d,
	Eigen::MatrixXf& ATA,
	Eigen::VectorXf& ATb,
	float track_radius, 
	bool is_converge_detect)
{
	Camera& cam = m_cameras[camid];
	Eigen::Matrix3f R = cam.R;
	Eigen::Matrix3f K = cam.K;
	Eigen::Vector3f T = cam.T;
	K.row(0) /= 1920; 
	K.row(1) /= 1080;
	int N = m_skelCorr.size();
	int M = m_poseToOptimize.size();
	Eigen::VectorXf r = Eigen::VectorXf::Zero(2 * N);
	Eigen::MatrixXf J;

	J = Eigen::MatrixXf::Zero(2 * N, 3 + 3 * M);

	std::vector<Eigen::Vector3f> skels2d;
	project(cam, skel3d, skels2d);
	
	for (int i = 0; i < N; i++)
	{
		const CorrPair& P = m_skelCorr[i];
		int t = P.target;
		if (det.keypoints[t](2) < m_skelTopo.kpt_conf_thresh[t]) continue;
		if (is_converge_detect)
		{
			float dist = (skels2d[t].segment<2>(0) - det.keypoints[t].segment<2>(0)).norm();
			float weight = 1;
			if (m_depth_weight.size() > 0 && m_depth_weight[camid] > 0.1) weight = m_depth_weight[camid];
			if (dist > track_radius * weight) {
				continue;
			}
		}
		m_det_confs[t] += 1; 

		Eigen::Vector3f x_local = K * (R * skel3d[t] + T);
		x_local(0);
		x_local(1);
		Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
		D(0, 0) = 1 / x_local(2);
		D(1, 1) = 1 / x_local(2);
		D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
		D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));

		J.middleRows(2 * i, 2) = P.weight * D * K * R * Jacobi3d.middleRows(3 * i, 3);

		Eigen::Vector2f u;
		u(0) = x_local(0) / x_local(2);
		u(1) = x_local(1) / x_local(2);
		Eigen::Vector2f det_u;
		det_u(0) = det.keypoints[t](0) / 1920; 
		det_u(1) = det.keypoints[t](1) / 1080;
		r.segment<2>(2 * i) = P.weight * (u - det_u);
	}

	ATb = -J.transpose() * r;
	ATA = J.transpose() * J;
}

void PigSolverDevice::optimizePose()
{
	std::cout << "optimize pose " << m_pig_id << std::endl; 
	m_det_confs.resize(m_skelTopo.joint_num);
	for (int k = 0; k < m_skelTopo.joint_num; k++) m_det_confs[k] = 0;
	directTriangulationHost();
	int maxIterTime = 200; 
	float terminal = 0.0001f; 
	
	int M = m_poseToOptimize.size();
	int paramNum = 3 + 3 * M;

	m_host_scale = gt_scales[m_pig_id];

	float loss_2d, loss_reg, loss_temp;

	int iterTime = 0; 

	for (; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();

		Eigen::VectorXf theta(paramNum);
		theta.segment<3>(0) = m_host_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_host_poseParam[jIdx];
		}

		calcSkelJacobiPartTheta_host(h_J_skel); // order same to m_skelCorr
		calcPoseJacobiPartTheta_device(d_J_joint, d_J_vert);
		d_J_vert.download(h_J_vert.data(), 3 * m_vertexNum * sizeof(float));
		d_J_joint.download(h_J_joint.data(), 3 * m_jointNum * sizeof(float));

		// data term
		Eigen::MatrixXf ATA_data = Eigen::MatrixXf::Zero(paramNum, paramNum); // data term 
		Eigen::VectorXf ATb_data = Eigen::VectorXf::Zero(paramNum);  // data term 

		if(m_skelProjs.size() > 0)
		for (int k = 0; k < m_skelTopo.joint_num; k++) m_det_confs[k] = 0;
		Calc2dJointProjectionTerm(m_source, ATA_data, ATb_data, m_kpt_track_dist, false); 

		float w_data = m_w_data_term;
		float w_reg = m_w_reg_term;
		float w_temp = m_w_temp_term;
		float w_anchor = m_w_anchor_term; 
		float w_floor = m_w_floor_term; 

		Eigen::MatrixXf ATA_floor; 
		Eigen::VectorXf ATb_floor; 
		CalcJointFloorTerm(ATA_floor, ATb_floor); 
		
		Eigen::MatrixXf DTD;
		CalcLambdaTerm(DTD);
		
		Eigen::MatrixXf ATA_temp = Eigen::MatrixXf::Zero(paramNum,paramNum); 
		Eigen::VectorXf ATb_temp = Eigen::VectorXf::Zero(paramNum); 
		
		if (m_last_regressed_skel3d.size() > 0)
		{
			CalcJointTempTerm2(ATA_temp, ATb_temp, h_J_skel, m_last_regressed_skel3d);
		}

		Eigen::MatrixXf ATA_reg;
		Eigen::VectorXf ATb_reg;
		CalcRegTerm(theta, ATA_reg, ATb_reg, false); 

		Eigen::MatrixXf ATA = ATA_data * w_data + DTD 
			+ ATA_temp * w_temp + ATA_reg * w_reg
			+ ATA_floor * w_floor
			; 
		Eigen::VectorXf ATb = ATb_data * w_data + ATb_temp * w_temp
			+ ATb_reg * w_reg
			 + ATb_floor * w_floor
			;

		Eigen::VectorXf delta = ATA.ldlt().solve(ATb);

		// update 

		m_host_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_host_poseParam[jIdx] += delta.segment<3>(3 + 3 * i);
		}
		loss_2d = ATb_data.norm(); 
		loss_reg = ATb_reg.norm(); 
		loss_temp = ATb_temp.norm();
		if (delta.norm() < terminal) break;
	}
	std::cout << "--iter: " << iterTime << std::endl;
	std::cout << "--loss ATb_data: " << loss_2d << std::endl;
	std::cout << "--loss ATb_reg : " << loss_reg << std::endl; 
	std::cout << "--loss ATb_temp: " << loss_temp << std::endl; 
}


void PigSolverDevice::Calc2dJointProjectionTerm(
	const MatchedInstance& source,
	Eigen::MatrixXf& ATA_data, Eigen::VectorXf& ATb_data, 
	float track_radius, 
	bool with_depth_weight, 
	bool is_converge_detect)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size(); 

	std::vector<Eigen::Vector3f> skel3d = getRegressedSkel_host();

	// data term
	ATA_data = Eigen::MatrixXf::Zero(paramNum, paramNum); // data term 
	ATb_data = Eigen::VectorXf::Zero(paramNum);  // data term 
	for (int k = 0; k < source.view_ids.size(); k++)
	{
		Eigen::MatrixXf H_view;
		Eigen::VectorXf b_view;
		int camid = source.view_ids[k];
		calcPose2DTerm_host(source.dets[k], camid, skel3d, h_J_skel,
			H_view, b_view, track_radius, is_converge_detect);
		float weight = 1; 
		if (with_depth_weight)
		{
			if(m_depth_weight[camid] > 0.1)
				weight =  1 / m_depth_weight[camid];
		}
		ATA_data += H_view * weight;
		ATb_data += b_view * weight;
	}
}

void PigSolverDevice::renderDepths()
{
	// render images 
	mp_renderEngine->clearAllObjs();
	RenderObjectColor* p_model = new RenderObjectColor();
	p_model->SetVertices(m_host_verticesPosed);
	p_model->SetFaces(m_host_facesVert);
	p_model->SetNormal(m_host_normalsFinal);
	p_model->SetColor(Eigen::Vector3f(1.0f,0.0f,0.0f));
	mp_renderEngine->colorObjs.push_back(p_model);

	const auto& cameras = m_cameras;

	for (int camid = 0; camid < cameras.size(); camid++)
	{
		Camera cam = m_cameras[camid];
		mp_renderEngine->s_camViewer.SetExtrinsic(cam.R, cam.T);

		float * depth_device = mp_renderEngine->renderDepthDevice();
		cudaMemcpy(d_depth_renders[camid], depth_device, WINDOW_WIDTH*WINDOW_HEIGHT * sizeof(float),
			cudaMemcpyDeviceToDevice);
	}
	mp_renderEngine->clearAllObjs(); 
}


//#define DEBUG_SIL

void PigSolverDevice::optimizePoseSilhouette(
	int maxIter)
{
	int iter = 0;

#ifdef DEBUG_SIL
	std::vector<cv::Mat> color_mask_dets;
	for (int view = 0; view < m_rois.size(); view++)
	{
		int camid = m_rois[view].viewid;
		cv::Mat img(cv::Size(1920, 1080), CV_8UC3); img.setTo(255);
		my_draw_mask(img,m_rois[view].mask_list, Eigen::Vector3i(255, 0, 0), 0);
		color_mask_dets.push_back(img);
	}
#endif 
	int M = m_poseToOptimize.size();
	int paramNum = 3 + 3 * M; 

	m_host_scale = gt_scales[m_pig_id];

	generateDataForSilSolver(); // generate mask opencv 

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

		calcSkelJacobiPartTheta_host(h_J_skel); 


		// calc joint term 
		Eigen::MatrixXf ATA_data; 
		Eigen::VectorXf ATb_data; 
		Calc2dJointProjectionTerm(m_source, ATA_data, ATb_data, m_kpt_track_dist, false); 

		calcPoseJacobiPartTheta_device(d_J_joint, d_J_vert, false); // TODO: remove d_J_vert computation here. 

		renderDepths(); 

#ifdef DEBUG_SIL
		// test 
		std::vector<cv::Mat> pseudos;
		cv::Mat depth_cv(cv::Size(1920, 1080), CV_32FC1); 
		for (int view = 0; view < m_rois.size(); view++)
		{
			cudaMemcpy(depth_cv.data, d_depth_renders[view], 1920 * 1080 * sizeof(float), cudaMemcpyDeviceToHost);
			cv::Mat colored = pseudoColor(depth_cv);
			pseudos.push_back(colored);
		}
		cv::Mat pack_pseudo;
		packImgBlock(pseudos,pack_pseudo);
		std::stringstream ss_pseudo;
		ss_pseudo << "G:/pig_results/debug/" << std::setw(6) << std::setfill('0') << iter << "_depth.jpg";
		cv::imwrite(ss_pseudo.str(), pack_pseudo); 

		cv::Mat blended;
		cv::Mat pack_det;
		packImgBlock(color_mask_dets, pack_det);
		blended = pack_pseudo * 0.5 + pack_det * 0.5;
		std::stringstream ss;
		ss << "G:/pig_results/debug/blend/" << std::setw(6) << std::setfill('0')
			<< iter << "_blend.jpg";
		cv::imwrite(ss.str(), blended);
#endif 
		// compute terms

		Eigen::MatrixXf ATA_sil;
		Eigen::VectorXf ATb_sil;

		if(m_use_gpu)
		    CalcSilhouettePoseTerm(ATA_sil, ATb_sil);
		else 
		    CalcSilouettePoseTerm_cpu(ATA_sil, ATb_sil, iter); 

		//Eigen::MatrixXf ATA_sil2;
		//Eigen::VectorXf ATb_sil2;
		//CalcSilouettePoseTerm_cpu(ATA_sil2, ATb_sil2, iter); 
		//std::cout << "ATA_difference: " << (ATA_sil - ATA_sil2).norm() << std::endl; 
		//std::cout << "ATb_difference: " << (ATb_sil - ATb_sil2).norm() << std::endl;

		//std::cout << "ATA cpu: " << std::endl<< ATA_sil2.block<9, 9>(0, 0) << std::endl; 
		//std::cout << "ATA gpu: " << std::endl << ATA_sil.block<9, 9>(0, 0) << std::endl;
		//std::cout << "ATb cpu: " << std::endl << ATb_sil2.segment<9>(0).transpose() << std::endl;
		//std::cout << "ATb gpu: " << std::endl << ATb_sil.segment<9>(0).transpose() << std::endl;


		float lambda = m_lambda;
		float w_data = m_w_data_term;
		float w_sil = m_w_sil_term;
		float w_reg = m_w_reg_term;
		float w_temp = m_w_temp_term;
		float w_floor = m_w_floor_term; 

		Eigen::MatrixXf ATA_floor;
		Eigen::VectorXf ATb_floor;
		CalcJointFloorTerm(ATA_floor, ATb_floor);

		Eigen::MatrixXf DTD;
		CalcLambdaTerm(DTD);

		Eigen::MatrixXf ATA_temp = Eigen::MatrixXf::Zero(paramNum, paramNum);
		Eigen::VectorXf ATb_temp = Eigen::VectorXf::Zero(paramNum);

		if (m_last_regressed_skel3d.size() > 0)
		{
			CalcJointTempTerm2(ATA_temp, ATb_temp, h_J_skel, m_last_regressed_skel3d);
		}

		Eigen::MatrixXf ATA_reg;
		Eigen::VectorXf ATb_reg;
		CalcRegTerm(theta, ATA_reg, ATb_reg, false);

		Eigen::MatrixXf H = ATA_sil * w_sil + ATA_reg * w_reg
			+ DTD * lambda
			+ ATA_data * w_data;
		Eigen::VectorXf b = ATb_sil * w_sil + ATb_reg * w_reg
			+ ATb_data * w_data; 

		Eigen::VectorXf delta = H.ldlt().solve(b);

#ifdef SHOW_FITTING_INFO
		if (iter == maxIter - 1) {
			std::cout << "iter ...... " << iter << " ......" << std::endl;
			std::cout << "ATb data : " << (ATb_data * w_data).norm() << std::endl;
			std::cout << "ATb sil  : " << (ATb_sil * w_sil).norm() << std::endl;
			std::cout << "ATb temp : " << (ATb_temp*w_temp).norm() << std::endl;
			std::cout << "ATb reg  : " << (ATb_reg*w_reg).norm() << std::endl;
			std::cout << "ATb floor: " << (ATb_floor*w_floor).norm() << std::endl;
		}
#endif 
		// update 
		m_host_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_host_poseParam[jIdx] += delta.segment<3>(3 + 3 * i);
		}
	}
}

//#define DEBUG_SIL

void PigSolverDevice::CalcSilhouettePoseTerm(
	Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size();
	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);

#ifdef DEBUG_SIL
	std::vector<cv::Mat> chamfers_vis;
	std::vector<cv::Mat> chamfers_vis_det;
	std::vector<cv::Mat> gradx_vis;
	std::vector<cv::Mat> grady_vis;
	std::vector<cv::Mat> diff_vis;
	std::vector<cv::Mat> diff_xvis;
	std::vector<cv::Mat> diff_yvis;
#endif 
	for (int view = 0; view < m_viewids.size(); view++)
	{
		//if (m_valid_keypoint_ratio[view] < m_valid_threshold) {
		//	//std::cout << "view " << view << " is invalid. " << m_valid_keypoint_ratio[view] << std::endl;
		//	continue;
		//}
		int camid = m_viewids[view];
		// compute detection image data 

		convertDepthToMaskHalfSize_device(d_depth_renders[camid], d_middle_mask, 1920, 1080);
		sdf2d_device(d_middle_mask, d_rend_sdf, 960, 540);

#ifdef DEBUG_SIL
		cv::Mat tmp(cv::Size(960, 540), CV_32FC1);
		cudaMemcpy(tmp.data, d_rend_sdf, 960 * 540 * sizeof(float), cudaMemcpyDeviceToHost); 
		cv::Mat tmp_vis = visualizeSDF2d(tmp/2, 32); 
		chamfers_vis.push_back(tmp_vis);
		//cv::Mat tmpmask(cv::Size(960, 540), CV_8UC1); 
		//cudaMemcpy(tmpmask.data, d_middle_mask, 960 * 540 * sizeof(uchar), cudaMemcpyDeviceToHost); 
		
		cv::Mat tmp2(cv::Size(960, 540), CV_32FC1); 
		cudaMemcpy(tmp2.data, d_det_sdf[view], 960 * 540 * sizeof(float), cudaMemcpyDeviceToHost); 
		cv::Mat tmp2_vis = visualizeSDF2d(tmp2/2, 32);
		chamfers_vis_det.push_back(tmp2_vis); 
#endif 

		Camera cam = m_cameras[camid];
		Eigen::Matrix3f R = cam.R;
		Eigen::Matrix3f K = cam.K;
		Eigen::Vector3f T = cam.T;

		// approx IOU 
		std::vector<unsigned char> visibility(m_vertexNum, 0);
		check_visibility(d_depth_renders_interact[camid], 1920, 1080, m_device_verticesPosed,
			K, R, T, visibility);
		float total_overlay = 0;
		float total_visible = 0;
		//cv::Mat vis_test;
		//vis_test.create(cv::Size(1920, 1080), CV_8UC3); 
		//vis_test = m_rois[roiIdx].mask * 24;
		for (int i = 0; i < m_vertexNum; i++)
		{
			if (m_host_bodyParts[i] == TAIL) continue; // ignore tail and ear
			if (visibility[i] == 0) continue; //only consider visible parts 
			Eigen::Vector3f xlocal = m_host_verticesPosed[i];
			Eigen::Vector3f x_proj = K * (R * xlocal + T);
			int x = round(x_proj(0) / x_proj(2));
			int y = round(x_proj(1) / x_proj(2));
			if (x < 0 || x >= 1920 || y < 0 || y >= 1080)continue;
			if (c_const_distort_mask.at<uchar>(y, x) == 0) continue;
			if (c_const_scene_mask[camid].at<uchar>(y, x) > 0)continue;
			int m = m_rois[view].queryMask(xlocal);
			total_visible += 1;
			if (m != 1)
			{
				continue;
			}
			total_overlay += 1;
		}
		float iou = total_overlay / total_visible;
		if (iou < m_iou_thres)
		{
			continue;
		}

		//std::cout << "IN pigsolverdevice d_ATA_sil " << d_ATA_sil.rows() << " " << d_ATA_sil.cols() << std::endl; 
		//std::cout << "IN pigsolverdevice d_ATA_sil " << d_ATb_sil.size() << std::endl;
		setConstant2D_device(d_ATA_sil, 0);
		setConstant1D_device(d_ATb_sil, 0);

#ifdef DEBUG_SOLVER
		setConstant2D_device(d_AT_sil, 0); 
		setConstant1D_device(d_b_sil, 0); 
#endif 
		calcSilhouetteJacobi_device(K, R, T, d_depth_renders[camid],
			d_depth_renders_interact[camid],
			1 << m_pig_id, paramNum, view);

#ifdef DEBUG_SOLVER
		//// TODO: delete
		computeATA_device(d_AT_sil, d_ATA_sil); 
		computeATb_device(d_AT_sil, d_b_sil, d_ATb_sil); 
		Eigen::MatrixXf AT = Eigen::MatrixXf::Zero(m_vertexNum, paramNum);
		Eigen::VectorXf b = Eigen::VectorXf::Zero(m_vertexNum); 
		d_AT_sil.download(AT.data(), m_vertexNum * sizeof(float)); 
		d_b_sil.download(b.data()); 

		int visible = 0;
		for (int k = 0; k < b.rows(); k++)if (fabs(b(k)) > 0.00001)visible++;
		std::cout << "visible gpu: " << visible << std::endl; 
		//std::ofstream stream_cpu("G:/pig_results/debug/AT_gpu.txt");
		//stream_cpu << AT << std::endl;
		//stream_cpu.close();
#endif 

		Eigen::MatrixXf ATA_view = Eigen::MatrixXf::Zero(paramNum, paramNum);
		Eigen::VectorXf ATb_view = Eigen::VectorXf::Zero(paramNum);
		d_ATA_sil.download(ATA_view.data(), ATA_view.cols() * sizeof(float));
		d_ATb_sil.download(ATb_view.data());

		float weight;
		if (m_depth_weight.size() > 0 && m_depth_weight[camid] > 0.1)
		{
			weight = 1.f / m_depth_weight[camid];
		}
		else
		{
			weight = 1.0f; 
		}
		ATA += ATA_view * weight; 
		ATb += ATb_view * weight; 
	}

#ifdef DEBUG_SIL
	cv::Mat packP;
	packImgBlock(chamfers_vis, packP);
	cv::Mat packD;
	packImgBlock(chamfers_vis_det, packD);
	std::stringstream ssp;
	ssp << "G:/pig_results/debug/" << 0 << "_rend_sdf_gpu.jpg";
	cv::imwrite(ssp.str(), packP);
	std::stringstream ssd;
	ssd << "G:/pig_results/debug/" << 0 << "_det_sdf_gpu.jpg";
	cv::imwrite(ssd.str(), packD);

#endif 
}



void PigSolverDevice::CalcSilouettePoseTerm_cpu(
	Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, int iter)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size();
	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);
	calcPoseJacobiPartTheta_device(d_J_joint, d_J_vert);
	d_J_vert.download(h_J_vert.data(), 3 * m_vertexNum * sizeof(float)); 
	d_J_joint.download(h_J_joint.data(), 3 * m_jointNum * sizeof(float)); 

	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(m_vertexNum, paramNum);
	Eigen::VectorXf r = Eigen::VectorXf::Zero(m_vertexNum);

#ifdef DEBUG_SIL
	std::vector<cv::Mat> chamfers_vis;
	std::vector<cv::Mat> chamfers_vis_det;
	std::vector<cv::Mat> gradx_vis;
	std::vector<cv::Mat> grady_vis;
	std::vector<cv::Mat> diff_vis;
	std::vector<cv::Mat> diff_xvis;
	std::vector<cv::Mat> diff_yvis;
#endif 

	for (int roiIdx = 0; roiIdx < m_rois.size(); roiIdx++)
	{
		if (m_rois[roiIdx].valid < m_valid_threshold) {
			//std::cout << "view " << roiIdx << " is invalid. " << m_rois[roiIdx].valid << std::endl;
			continue;
		}
		A.setZero(); 
		r.setZero(); 

		cv::Mat P;
		int camid = m_source.view_ids[roiIdx];
		computeSDF2d_device(d_depth_renders[camid],d_middle_mask, P, 1920,1080);


#ifdef DEBUG_SIL
		chamfers_vis.emplace_back(visualizeSDF2d(P));

		cv::Mat chamfer_vis = visualizeSDF2d(m_rois[roiIdx].chamfer);

		chamfers_vis_det.push_back(chamfer_vis);
		diff_vis.emplace_back(visualizeSDF2d(P - m_rois[roiIdx].chamfer, 32));
#endif 
		Camera cam = m_rois[roiIdx].cam;
		Eigen::Matrix3f R = cam.R;
		Eigen::Matrix3f K = cam.K;
		Eigen::Vector3f T = cam.T;

		float wc = 0.01;

		std::vector<unsigned char> visibility(m_vertexNum, 0); 
		check_visibility(d_depth_renders_interact[camid], 1920, 1080, m_device_verticesPosed,
			K, R, T, visibility);

		// approx IOU 
		float total_overlay = 0;
		float total_visible = 0; 
		//cv::Mat vis_test;
		//vis_test.create(cv::Size(1920, 1080), CV_8UC3); 
		//vis_test = m_rois[roiIdx].mask * 24;
		for (int i = 0; i < m_vertexNum; i++)
		{
			if (m_host_bodyParts[i] == TAIL) continue; // ignore tail and ear
			if (visibility[i] == 0) continue; //only consider visible parts 
			Eigen::Vector3f xlocal = m_host_verticesPosed[i];
			Eigen::Vector3f x_proj = K * (R * xlocal + T);
			int x = round(x_proj(0) / x_proj(2));
			int y = round(x_proj(1) / x_proj(2));
			if (x < 0 || x >= 1920 || y < 0 || y >= 1080)continue;
			if (c_const_distort_mask.at<uchar>(y, x) == 0) continue;
			if (c_const_scene_mask[camid].at<uchar>(y, x) > 0)continue;
			int m = m_rois[roiIdx].queryMask(xlocal);
			total_visible += 1;
			//cv::circle(vis_test, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);

			if (m!=1)
			{
				continue;
			}
			total_overlay += 1;
		}

		float iou = total_overlay / total_visible; 
		if (iou < m_iou_thres)
		{
			continue; 
		}

		int visible = 0; 
		for (int i = 0; i < m_vertexNum; i++)
		{
			if (m_host_bodyParts[i] == TAIL || m_host_bodyParts[i] == L_EAR || m_host_bodyParts[i] == R_EAR) continue;

			if (visibility[i] == 0) continue; 
			Eigen::Vector3f x0 = m_host_verticesPosed[i];
			// check visibiltiy 
		
			Eigen::Vector3f x_local = K * (R * x0 + T);
			int x = round(x_local(0) / x_local(2)); 
			int y = round(x_local(1) / x_local(2)); 
			if (x < 0 || x >= 1920 || y < 0 || y >= 1080)continue;

			if (c_const_distort_mask.at<uchar>(y, x) == 0) continue; 
			if (c_const_scene_mask[camid].at<uchar>(y, x) > 0)continue; 

			int m = m_rois[roiIdx].queryMask(x0);
			// TODO: 20200602 check occlusion
			if (m == 2 || m == 3) continue;
			// TODO: 20200501 use mask to check visibility 
			float d = m_rois[roiIdx].queryChamfer(x0);
			if (d < -9999) continue;

			float p = queryPixel(P, x0, m_rois[roiIdx].cam);
			if (p > 10) continue; // only consider contours for loss 

			float ddx = queryPixel(m_rois[roiIdx].gradx, x0, cam);
			float ddy = queryPixel(m_rois[roiIdx].grady, x0, cam);

			Eigen::MatrixXf block2d = Eigen::MatrixXf::Zero(2, paramNum);
			Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
			D(0, 0) = 1 / x_local(2);
			D(1, 1) = 1 / x_local(2);
			D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
			D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));
			block2d = D * K * R * h_J_vert.middleRows(3 * i, 3);
			r(i) = (p - d);
			A.row(i) =  (block2d.row(0) * (ddx)+block2d.row(1) * (ddy));
			visible++;
		}
		A = wc * A;
		r = wc * r;

#ifdef DEBUG_SOLVER
		// TODO: delete
		std::ofstream stream_cpu("G:/pig_results/debug/AT_cpu.txt");
		stream_cpu << A << std::endl; 
		stream_cpu.close(); 
		std::cout << "visible cpu: " << visible << std::endl; 
#endif 
		float weight;
		if(m_depth_weight.size() > 0 && m_depth_weight[camid] > 0.1) weight= 1/m_depth_weight[camid];
		else weight = 1; 
		ATA += A.transpose() * A * weight;
		ATb += A.transpose() * r * weight;

	}
#ifdef DEBUG_SIL
	cv::Mat packP;
	packImgBlock(chamfers_vis, packP);
	cv::Mat packD;
	packImgBlock(chamfers_vis_det, packD);
	std::stringstream ssp;
	ssp << "G:/pig_results/debug/" << iter << "_rend_sdf_cpu.jpg";
	cv::imwrite(ssp.str(), packP);
	std::stringstream ssd;
	ssd << "G:/pig_results/debug/" << iter << "_det_sdf_cpu.jpg";
	cv::imwrite(ssd.str(), packD);

	cv::Mat packdiff; packImgBlock(diff_vis, packdiff);
	std::stringstream ssdiff;
	ssdiff << "G:/pig_results/debug/diff_" << iter << ".jpg";
	cv::imwrite(ssdiff.str(), packdiff);

#endif 
}


void PigSolverDevice::CalcPoseJacobiFullTheta_cpu(Eigen::MatrixXf& jointJacobiPose, Eigen::MatrixXf& J_vert,
	bool with_vert)
{
	// calculate delta rodrigues
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
			LP.block<3, 3>(3 * child, 3 * jIdx) = LP.block<3, 3>(3 * father, 3 * jIdx)
				* m_host_localSE3[child].block<3, 3>(0, 0); 
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

	if (!with_vert) return;
	J_vert = Eigen::MatrixXf::Zero(3 * m_vertexNum, 3 + 3 * m_jointNum);
	Eigen::MatrixXf block = Eigen::MatrixXf::Zero(3, 3 + 3 * m_jointNum);
	for (int vIdx = 0; vIdx < m_vertexNum; vIdx++)
	{
		Eigen::Vector3f v0 = m_host_verticesDeformed[vIdx];
		block.setZero();
		for (int jIdx = 0; jIdx < m_jointNum; jIdx++)
		{
			if (m_host_lbsweights(jIdx, vIdx) < 0.00001) continue;
			block += (m_host_lbsweights(jIdx, vIdx) * jointJacobiPose.middleRows(3 * jIdx, 3));
			Eigen::Vector3f j0 = m_host_jointsDeformed[jIdx];
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
		J_vert.middleRows(3 * vIdx, 3) = block;
	}
}

void PigSolverDevice::CalcPoseJacobiPartTheta_cpu(Eigen::MatrixXf& J_joint, Eigen::MatrixXf& J_vert,
	bool with_vert)
{
	Eigen::MatrixXf J_jointfull, J_vertfull;
	CalcPoseJacobiFullTheta_cpu(J_jointfull, J_vertfull, with_vert);
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
		if (with_vert)
			J_vert.middleCols(3 + 3 * i, 3) = J_vertfull.middleCols(3 + 3 * thetaid, 3);
	}
}




void PigSolverDevice::calcPoseJacobiFullTheta_device(
	pcl::gpu::DeviceArray2D<float> &J_joint,
	pcl::gpu::DeviceArray2D<float> &J_vert, 
	bool with_vert
)
{
	int cpucols = 3 * m_jointNum + 3; // theta dimension 
	if (J_joint.empty())
		J_joint.create(cpucols, m_jointNum * 3); // J_joint_cpu.T, with same storage array  
	if (J_vert.empty())
		J_vert.create(cpucols, m_vertexNum * 3); // J_vert_cpu.T 
	setConstant2D_device(J_joint, 0);
	setConstant2D_device(J_vert, 0);

	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * m_jointNum);
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3f& pose = m_host_poseParam[jointId];
		if (jointId == 0)
		{
			rodriguesDerivative.block<3, 9>(0, 0) = EulerJacobiF(pose);
			//rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiF(pose);

		}
		else {
			rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiF(pose);
		}
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

	J_joint.upload(jointJacobiPose.data(), (3 * m_jointNum) * sizeof(float), cpucols, 3 * m_jointNum);
	m_device_jointsDeformed.upload(m_host_jointsDeformed);
	m_device_verticesDeformed.upload(m_host_verticesDeformed);
	d_RP.upload(RP.data(), 9 * m_jointNum * sizeof(float), 3, 9 * m_jointNum);
	d_LP.upload(LP.data(), 3 * m_jointNum * sizeof(float), 3 * m_jointNum, 3 * m_jointNum);

	if(with_vert)
		calcPoseJacobiFullTheta_V_device(J_vert, J_joint, d_RP, d_LP); 
}


void PigSolverDevice::generateDataForSilSolver()
{
	int W = 1920;
	int H = 1080;
	m_det_masks_binary.resize(m_cameras.size()); 
	m_viewids = m_source.view_ids;
	m_mask_areas.resize(m_viewids.size(), 0); 
	m_valid_keypoint_ratio.resize(m_viewids.size(), 0);

	for (int view = 0; view < m_viewids.size(); view++)
	{
		cv::Mat amask(cv::Size(W, H), CV_8UC1);
		my_draw_mask_gray(amask,
			m_source.dets[view].mask, 1);
		m_mask_areas[view] = cv::countNonZero(amask);
		cv::resize(amask, m_det_masks_binary[view], cv::Size(W / 2, H / 2));

		int camid = m_viewids[view];
		int idcode = 1 << m_pig_id; 
		m_valid_keypoint_ratio[view] = checkKeypointsMaskOverlay(
			m_det_masks[camid], m_source.dets[view].keypoints, idcode
		);

		if (m_use_gpu)
		{
			cudaMemcpy(d_middle_mask, m_det_masks_binary[view].data, 960 * 540 * sizeof(uchar), cudaMemcpyHostToDevice);
			sdf2d_device(d_middle_mask, d_det_sdf[view], 960, 540);
			sobel_device(d_det_sdf[view], d_det_gradx[view], d_det_grady[view], 960, 540);

			cudaMemcpy(d_det_mask[view], m_det_masks[camid].data, 1920 * 1080 * sizeof(uchar), cudaMemcpyHostToDevice);
		}
	}

	m_param_observe_num.setZero(); 
	for (int view = 0; view < m_viewids.size(); view++)
	{
		for (int jid = 0; jid < m_skelTopo.joint_num; jid++)
		{
			if (m_source.dets[view].keypoints[jid](2) >
				m_skelTopo.kpt_conf_thresh[jid]) m_param_observe_num(jid)++;
		}
	}
	std::vector<std::pair<int, int> > observe_optim_map = {
		{5, 14},
	{6, 6},
	{7, 15}, 
	{8, 7},
	{9,16},
	{10,8},
	{11,55},
	{12,39},
	{13,56},
	{14,40},
	{15,57},
	{16,41}
	};
	m_param_temp_weight.setOnes(); 
	for (int i = 0; i < observe_optim_map.size(); i++)
	{
		int skelid = observe_optim_map[i].first;
		int jointid = observe_optim_map[i].second;
		if (m_param_observe_num(skelid) >= 3) continue; 
		else {
			for (int k = 0; k < m_poseToOptimize.size(); k++)
			{
				if (m_poseToOptimize[k] != jointid)continue;
				else {
					m_param_temp_weight.segment<3>(3 + 3 * k) 
						*= (6 / (m_param_observe_num(skelid) + 0.5));
					break; 
				}
			}
		}
	}
	
}


// =======================
// Scene Constraint term: 
// A pig mush be on the ground, thus v(2)> 0 for all verticse. 
// Considering that we cant formulate soft tissue deformation, 
// we only constrain legs with this term, called `floor term`
// Instead of peanalize every surface point, I only penalize 
// joints for simplicity. 
// By AN Liang, 2020/Sept/17
// =======================
void PigSolverDevice::CalcJointFloorTerm(
	Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb
)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size(); 
	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum); 

	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(m_jointNum, paramNum); 
	Eigen::VectorXf b = Eigen::VectorXf::Zero(m_jointNum); 

	// assume we have J_joint_part_theta
	d_J_joint.download(h_J_joint.data(), m_jointNum * 3 * sizeof(float));
	for (int jid = 0; jid < m_jointNum; jid++)
	{
		Eigen::Vector3f joint = m_host_jointsPosed[jid];
		//if (jid == 2 && joint(2) < 0.13)
		//{
		//	A.row(jid) = h_J_joint.row(3 * jid + 2);
		//	b(jid) = joint(2) - 0.1;
		//	continue;
		//}
		if (joint(2) > 0) continue; 
		else
		{
			A.row(jid) = h_J_joint.row(3 * jid + 2);
			b(jid) = joint(2);
		}
	}
	//A.middleCols(0, 6).setZero();
	//b.segment<6>(0).setZero();
	ATA = A.transpose() * A; 
	ATb = -A.transpose() * b; 
}

void PigSolverDevice::CalcJointBidirectFloorTerm(
	Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, std::vector<bool> foot_contact
)
{
	assert(foot_contact.size() == 4); 
	int paramNum = 3 + 3 * m_poseToOptimize.size();
	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);

	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(m_jointNum, paramNum);
	Eigen::VectorXf b = Eigen::VectorXf::Zero(m_jointNum);

	// assume we have J_joint_part_theta
	d_J_joint.download(h_J_joint.data(), m_jointNum * 3 * sizeof(float));
	for (int jid = 0; jid < m_jointNum; jid++)
	{
		Eigen::Vector3f joint = m_host_jointsPosed[jid];
		//if (jid == 2 && joint(2) < 0.13)
		//{
		//	A.row(jid) = h_J_joint.row(3 * jid + 2);
		//	b(jid) = joint(2) - 0.1;
		//	continue;
		//}
		if (joint(2) > 0)
		{
			if ((jid == 9 && foot_contact[0]) ||
				(jid == 10 && foot_contact[1]) ||
				(jid == 15 && foot_contact[2]) ||
				(jid == 16 && foot_contact[3])
				)
			{
				A.row(jid) = h_J_joint.row(3 * jid + 2);
				b(jid) = joint(2);
			}
		}
		else
		{
			A.row(jid) = h_J_joint.row(3 * jid + 2);
			b(jid) = joint(2);
		}
	}

	//std::vector<int> active = { 6,14,39,55 };
	A.middleCols(0, 6).setZero();
	b.segment<6>(0).setZero(); 
	ATA = A.transpose() * A;
	ATb = -A.transpose() * b;
}



void PigSolverDevice::CalcJointTempTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, 
	const Eigen::VectorXf& last_theta, const Eigen::VectorXf& theta)
{
	int paramNum = 3 * m_poseToOptimize.size() + 3; 
	ATA = Eigen::MatrixXf::Identity(paramNum, paramNum);
	ATb = last_theta - theta;
	for (int i = 0; i < paramNum; i++)
	{
		ATA(i, i) *= m_param_temp_weight(i);
		ATb(i) *= m_param_temp_weight(i);
	}
}

std::vector<float> PigSolverDevice::computeValidObservation()
{
	std::vector<float> obs(m_skelTopo.joint_num, 0);
	for (int view = 0; view < m_source.view_ids.size(); view++)
	{
		for (int i = 0; i < m_skelTopo.joint_num; i++)
		{
			if (m_source.dets[view].keypoints[i](2) > m_skelTopo.kpt_conf_thresh[i])
				obs[i] += 1; 
		}
	}
	return obs; 
}

void PigSolverDevice::CalcJointTempTerm2(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, const Eigen::MatrixXf& skelJ,
	const std::vector<Eigen::Vector3f>& last_regressed_skel)
{
	int paramNum = 3 * m_poseToOptimize.size() + 3;
	int N = m_skelCorr.size(); 
	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(3 * N, paramNum);
	Eigen::VectorXf r = Eigen::VectorXf::Zero(3 * N);
	A = skelJ; 
	std::vector<Eigen::Vector3f> skel = getRegressedSkel_host(); 
	std::vector<float> obs = computeValidObservation(); 
	for (int i = 0; i < N; i++)
	{
		int t = m_skelCorr[i].target; 
		float w = m_skelCorr[i].weight;
		r.segment<3>(3 * i) = skel[t] - last_regressed_skel[t];
		if (obs[i] >= 1) {
			A.middleRows(3 * i, 3) *= 0.01;
			r.segment<3>(3 * i, 3) *= 0.01; 
		}
	}
	ATb = -A.transpose() * r; 
	ATA = A.transpose() * A; 
}

void PigSolverDevice::postProcessing()
{
	m_skel3d = getRegressedSkel_host(); 
	m_last_regressed_skel3d = m_skel3d;
	if (m_skelProjs.size() == 0)
	{
		m_skelProjs.resize(m_cameras.size()); 
		for (int i = 0; i < m_cameras.size(); i++)
		{
			m_skelProjs[i].resize(m_skelTopo.joint_num, Eigen::Vector3f::Zero()); 
		}
	}

	for (int view = 0; view < m_cameras.size(); view++)
	{
		project(m_cameras[view], m_skel3d, m_skelProjs[view]);
	}

	int paramNum = 3 + 3 * m_poseToOptimize.size(); 
	m_last_thetas.resize(paramNum);
	m_last_thetas.segment<3>(0) = m_host_translation;
	for (int i = 0; i < m_poseToOptimize.size(); i++)
	{
		int jIdx = m_poseToOptimize[i];
		m_last_thetas.segment<3>(3 + 3 * i) = m_host_poseParam[jIdx];
	}


	m_depth_weight.resize(m_cameras.size(), 0);
	Eigen::Vector3f center = m_host_jointsPosed[2];

	for (int camid = 0; camid < m_cameras.size(); camid++)
	{		
		Eigen::Vector3f center_local = m_cameras[camid].R * center + m_cameras[camid].T;
		float depth = center_local(2);
		if (depth <= 0) m_depth_weight[camid] = 0.0f;
		else {
			m_depth_weight[camid] = 2 / (depth + 0.1);
		}
	}
}

void PigSolverDevice::calcSkel3DTerm_host(
	const Eigen::MatrixXf& Jacobi3d,
	const std::vector<Eigen::Vector3f>& skel3d, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size(); 
	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum); 
	ATb = Eigen::VectorXf::Zero(paramNum); 
	int N = m_skelCorr.size();
	std::vector<Eigen::Vector3f> skel = getRegressedSkel_host(); 

	Eigen::VectorXf r = Eigen::VectorXf::Zero(3 * N);
	for (int i = 0; i < N; i++)
	{
		const CorrPair& P = m_skelCorr[i];
		int t = P.target;
		if (skel3d[t].norm() == 0) continue;

		r.segment<3>(3 * i) = P.weight * (skel[t] - skel3d[t]);
	}
	ATA = Jacobi3d.transpose() * Jacobi3d;
	ATb = -Jacobi3d.transpose() * r; 
}

void PigSolverDevice::calcJoint3DTerm_host(
	const Eigen::MatrixXf& Jacobi3d,
	const std::vector<Eigen::Vector3f>& joints, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size();
	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);
	int N = joints.size();

	Eigen::VectorXf r = Eigen::VectorXf::Zero(3 * N);
	for (int i = 0; i < N; i++)
	{
		r.segment<3>(3 * i) = (m_host_jointsPosed[i] - joints[i]);
	}
	ATA = Jacobi3d.transpose() * Jacobi3d;
	ATb = -Jacobi3d.transpose() * r;
}


void PigSolverDevice::CalcRegTerm(
	const Eigen::VectorXf& theta,
	Eigen::MatrixXf& ATA,
	Eigen::VectorXf& ATb,
	bool adaptive_weight
)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size(); 
	ATA = Eigen::MatrixXf::Identity(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);
	ATb = -theta; 
	ATb.segment<6>(0).setZero(); 

	if (!adaptive_weight) return;
	if (m_det_confs[0] >= 2 && m_det_confs[1] >= 2 && m_det_confs[2] >= 2
		&& m_det_confs[3] >= 2 && m_det_confs[4] >= 2)
	{
		std::vector<int> ignore = { 19,20,21,1,2}; // 21,22,23
		for (const int & k : ignore)
		{
			ATA.middleRows(3 + 3 * k, 3) *= 1;
			ATb.segment<3>(3 + 3 * k) *= 1;
		}
	}

	float w = 1;

	if (m_det_confs[5] >= 1 && m_det_confs[7] >= 1 && m_det_confs[9] >= 1)
	{
		std::vector<int> ignore = { 8,9,10 };
		float sum = m_det_confs[6] + m_det_confs[8] + m_det_confs[10];

		for (const int & k : ignore)
		{
			ATA.middleRows(3 + 3 * k, 3) *= (w / sqrtf(sum));
			ATb.segment<3>(3 + 3 * k) *= (w / sqrtf(sum));
		}
	}
	if (m_det_confs[6] >= 1 && m_det_confs[8] >= 1 && m_det_confs[10] >= 1)
	{
		std::vector<int> ignore = { 4,5,6 };
		float sum = m_det_confs[6] + m_det_confs[8] + m_det_confs[10];

		for (const int & k : ignore)
		{
			ATA.middleRows(3 + 3 * k, 3) *= (w / sqrtf(sum));
			ATb.segment<3>(3 + 3 * k) *= (w / sqrtf(sum));
		}
	}

	if (m_det_confs[11] >= 1 && m_det_confs[13] >= 1 && m_det_confs[15] >= 1)
	{
		std::vector<int> ignore = { 16,17,18 };
		float sum = m_det_confs[11] + m_det_confs[13] + m_det_confs[15];

		for (const int & k : ignore)
		{
			ATA.middleRows(3 + 3 * k, 3) *= (w / sqrtf(sum));
			ATb.segment<3>(3 + 3 * k) *= (w / sqrtf(sum));
		}
	}
	if (m_det_confs[12] >= 1 && m_det_confs[14] >= 1 && m_det_confs[16] >= 1)
	{
		std::vector<int> ignore = { 12,13,14 };
		float sum = m_det_confs[12] + m_det_confs[14] + m_det_confs[16];
		for (const int & k : ignore)
		{
			ATA.middleRows(3 + 3 * k, 3) *= (w / sqrtf(sum));
			ATb.segment<3>(3 + 3 * k) *= (w / sqrtf(sum));
		}
	}
}

void PigSolverDevice::getTheta(Eigen::VectorXf& theta)
{
	int M = m_poseToOptimize.size();
	int paramNum = 3 + 3 * M;
	theta.resize(paramNum);
	theta.setZero();
	theta.segment<3>(0) = m_host_translation;
	for (int i = 0; i < M; i++)
	{
		int jIdx = m_poseToOptimize[i];
		theta.segment<3>(3 + 3 * i) = m_host_poseParam[jIdx];
	}
}

void PigSolverDevice::setTheta(const Eigen::VectorXf& theta)
{
	int M = m_poseToOptimize.size();
	int paramNum = 3 + 3 * M;
	m_host_translation = theta.segment<3>(0);
	for (int i = 0; i < M; i++)
	{
		int jIdx = m_poseToOptimize[i];
		m_host_poseParam[jIdx] = theta.segment<3>(3 + 3 * i);
	}
}

void PigSolverDevice::computeDepthWeight()
{
	m_depth_weight.resize(m_cameras.size(), 0);
	Eigen::Vector3f center = m_host_jointsPosed[2];

	for (int camid = 0; camid < m_cameras.size(); camid++)
	{
		Eigen::Vector3f center_local = m_cameras[camid].R * center + m_cameras[camid].T;
		float depth = center_local(2);
		if (depth <= 0) m_depth_weight[camid] = 0.0f;
		else {
			m_depth_weight[camid] = 2 / (depth + 0.01);
		}
	}
}

std::vector<float> PigSolverDevice::regressSkelVisibility(int camid)
{
	renderDepths();
	std::vector<unsigned char> visibility(m_vertexNum, 0);
	Eigen::Matrix3f K = m_cameras[camid].K;
	Eigen::Matrix3f R = m_cameras[camid].R; 
	Eigen::Vector3f T = m_cameras[camid].T; 
	//check_visibility(d_depth_renders_interact[camid], 1920, 1080, m_device_verticesPosed,
	//	K, R, T, visibility);
	check_visibility(d_depth_renders[camid], 1920, 1080, m_device_verticesPosed,
		K, R, T, visibility); // use single depth because other' estimation is not trustable either.
	std::vector<float> skel_vis(m_skelTopo.joint_num, 0); 
	auto skel = getRegressedSkel_host(); 
	for (int i = 0; i < m_skelCorr.size(); i++)
	{
		Eigen::Vector3f X = skel[i];
		Eigen::Vector3f x_proj = K * (R * X + T);
		int x = round(x_proj(0) / x_proj(2));
		int y = round(x_proj(1) / x_proj(2));
		if (x < 0 || x >= 1920 || y < 0 || y >= 1080)continue;

		// we alwayes need to consider false fitting results
		//if (c_const_distort_mask.at<uchar>(y, x) == 0) continue;
		//if (c_const_scene_mask[camid].at<uchar>(y, x) > 0)continue;

		int t = m_skelCorr[i].target; 
		int type = m_skelCorr[i].type; 

		for (int k = 0; k < m_visRegressorList[t].size(); k++)
		{
			skel_vis[t] += visibility[m_visRegressorList[t][k]];
		}
		skel_vis[t] /= m_visRegressorList[t].size();
		
	}
	
	return skel_vis; 
}

void PigSolverDevice::computeAllSkelVisibility()
{
	m_skel_vis.resize(m_cameras.size()); 
	for (int camid = 0; camid < m_cameras.size(); camid++)
	{
		m_skel_vis[camid] = regressSkelVisibility(camid); 
	}
}