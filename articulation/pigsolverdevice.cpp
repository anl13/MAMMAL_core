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

	// some initial state 
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