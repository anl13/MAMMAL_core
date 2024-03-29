#include "pigsolverdevice.h"
#include "../utils/geometry.h"
#include "../utils/Hungarian.h"

//#define VISUAL_LOSS

cv::Mat PigSolverDevice::debug_source_visualize()
{
	std::vector<cv::Mat> m_imgdet; 
	cloneImgs(m_rawimgs, m_imgdet); 
	for (int i = 0; i < m_source.view_ids.size(); i++)
	{
		int camid = m_source.view_ids[i];
		drawSkelDebug(m_imgdet[camid], m_source.dets[i].keypoints, m_skelTopo);
		//my_draw_box(m_imgdet[camid], m_source.dets[i].box, m_CM[m_pig_id]);
		//my_draw_mask(m_imgsDetect[camid], m_matched[id].dets[i].mask, m_CM[id], 0.5);
	}

	std::vector<cv::Mat> crop_list(12);
	for (int k = 0; k < crop_list.size(); k++)
	{
		crop_list[k] = cv::Mat(cv::Size(256, 256), CV_8UC3);
		crop_list[k].setTo(cv::Scalar(255, 255, 255)); 
	}
	for (int i = 0; i < m_source.dets.size(); i++)
	{
		Eigen::Vector4f box = m_source.dets[i].box;
		Eigen::Vector4f box2 = expand_box(box, 0.2);
		int camid = m_source.view_ids[i];
		cv::Mat raw_img = m_imgdet[camid];
		cv::Rect2i roi(box2[0], box2[1], box2[2] - box2[0], box2[3] - box2[1]);
		cv::Mat img = raw_img(roi);
		cv::Mat img2 = resizeAndPadding(img, 256, 256);
		crop_list[camid] = img2; 
	}
	cv::Mat output;

	packImgBlock(crop_list, output);

	return output;
}


void AnchorPoseLib::load(std::string folder)
{
	std::string pig_config = "D:/Projects/animal_calib/articulation/artist_config_sym.json";
	PigModelDevice driver(pig_config);
	SkelTopology topo = driver.GetTopo();
	anchors.clear();
	int pose_id = 0;
	while (1) {
		std::stringstream filess;
		filess << folder << "/pose" << pose_id << ".txt";
		std::ifstream input(filess.str());
		if (!input.is_open())break;
		AnchorPoseType anchor;
		anchor.pose = driver.GetPose();
		for (int k = 0; k < 3; k++) input >> anchor.translation(k);
		for (int k = 0; k < 62; k++)
		{
			input >> anchor.pose[k](0) >>
				anchor.pose[k](1) >> anchor.pose[k](2);
		}
		input >> anchor.scale;
		input.close();
		anchor.scale = 1;
		anchor.translation(0) = 0;
		anchor.translation(1) = 0;
		anchor.pose[0](0) = 0;
		driver.SetPose(anchor.pose);
		driver.SetTranslation(anchor.translation);
		driver.SetScale(anchor.scale);
		driver.UpdateVertices();
		anchor.joint_positions_62 = driver.GetJoints();
		anchor.joint_positions_23 = driver.getRegressedSkel_host();

		anchors.push_back(anchor);
		pose_id++;
	};
}

void PigSolverDevice::getThetaAnchor(Eigen::VectorXf& theta)
{
	theta.resize(3);
	theta(0) = m_host_translation(0);
	theta(1) = m_host_translation(1);
	theta(2) = m_host_poseParam[0](0);
}

void PigSolverDevice::setThetaAnchor(const Eigen::VectorXf& theta)
{
	int paramNum = 3;
	m_host_translation(0) = theta(0);
	m_host_translation(1) = theta(1);
	m_host_poseParam[0](0) = theta(2);
}


// calc anchor pose term 
void PigSolverDevice::CalcSkelProjectionTermAnchor(
	const MatchedInstance& source,
	Eigen::MatrixXf& ATA_data, Eigen::VectorXf& ATb_data,
	bool with_depth_weight)
{
	int paramNum = 3;

	std::vector<Eigen::Vector3f> skel2d = getRegressedSkel_host();

	Eigen::MatrixXf J3D; 
	J3D.resize(h_J_skel.rows(), 3); 
	J3D.middleCols(0, 2) = h_J_skel.middleCols(0, 2); 
	J3D.col(2) = h_J_skel.col(3);
	// data term
	ATA_data = Eigen::MatrixXf::Zero(paramNum, paramNum); // data term 
	ATb_data = Eigen::VectorXf::Zero(paramNum);  // data term 
	for (int k = 0; k < source.view_ids.size(); k++)
	{
		Eigen::MatrixXf H_view;
		Eigen::VectorXf b_view;
		int camid = source.view_ids[k];
		calcSkel2DTermAnchor_host(source.dets[k], camid, skel2d, J3D, H_view, b_view);
		float weight = 1;
		if (with_depth_weight)
		{
			weight = m_depth_weight[camid];
		}
		ATA_data += H_view * weight;
		ATb_data += b_view * weight;
	}
}

void PigSolverDevice::calcSkel2DTermAnchor_host(
	const DetInstance& det,
	int camid,
	const std::vector<Eigen::Vector3f> & skel2d,
	const Eigen::MatrixXf& Jacobi3d,
	Eigen::MatrixXf& ATA,
	Eigen::VectorXf& ATb
)
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

	J = Eigen::MatrixXf::Zero(2 * N, 3);

	for (int i = 0; i < N; i++)
	{
		const CorrPair& P = m_skelCorr[i];
		int t = P.target;
		if (det.keypoints[t](2) < m_skelTopo.kpt_conf_thresh[t]) continue;

		Eigen::Vector3f x_local = K * (R * skel2d[t] + T);
		x_local(0);
		x_local(1);
		Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
		D(0, 0) = 1 / x_local(2);
		D(1, 1) = 1 / x_local(2);
		D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
		D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));

		float dw = 1;
		if (m_skelProjs.size() > 0)
			if(m_depth_weight.size() == m_cameras.size())
				dw = m_depth_weight[camid];
			

		J.middleRows(2 * i, 2) = P.weight * D * K * R * Jacobi3d.middleRows(3 * i, 3) / dw;

		Eigen::Vector2f u;
		u(0) = x_local(0) / x_local(2);
		u(1) = x_local(1) / x_local(2);
		Eigen::Vector2f det_u;
		det_u(0) = det.keypoints[t](0) / 1920;
		det_u(1) = det.keypoints[t](1) / 1080;
		r.segment<2>(2 * i) = P.weight * (u - det_u) / dw;
	}

	ATb = -J.transpose() * r;
	ATA = J.transpose() * J;
}


void PigSolverDevice::optimizeAnchor(int anchor_id)
{
	m_det_confs.resize(m_skelTopo.joint_num);
	for (int k = 0; k < m_skelTopo.joint_num; k++) m_det_confs[k] = 0;
	directTriangulationHost();

	int maxIterTime = 200;

	AnchorPoseType A = m_anchor_lib.anchors[anchor_id];
	m_host_translation = A.translation;
	// manually define scale 
	m_host_translation *= m_host_scale;
	m_host_poseParam = A.pose;

	int M = m_poseToOptimize.size();
	int paramNum = 3;
	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();

		Eigen::VectorXf theta(3);
		getThetaAnchor(theta);

		calcSkelJacobiPartTheta_host(h_J_skel); // order same to m_skelCorr
		//calcPoseJacobiPartTheta_device(d_J_joint, d_J_vert);
		//d_J_vert.download(h_J_vert.data(), 3 * m_vertexNum * sizeof(float));
		//d_J_joint.download(h_J_joint.data(), 3 * m_jointNum * sizeof(float));

		Eigen::MatrixXf ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
		Eigen::VectorXf ATb = Eigen::VectorXf::Zero(paramNum);
		Eigen::MatrixXf ATA_data = Eigen::MatrixXf::Zero(paramNum, paramNum);
		Eigen::VectorXf ATb_data = Eigen::VectorXf::Zero(paramNum);
		CalcSkelProjectionTermAnchor(m_source, ATA_data, ATb_data, false);

		ATA += ATA_data;
		ATb += ATb_data;
		Eigen::VectorXf delta = ATA.ldlt().solve(ATb);
		m_host_translation(0) += delta(0);
		m_host_translation(1) += delta(1);
		m_host_poseParam[0](0) += delta(2);
		if (delta.norm() < 0.00001) break;
	}
}

int PigSolverDevice::searchAnchorSpace()
{
	std::vector<float> anchor_errors(m_anchor_lib.anchors.size(), 0);
	std::vector<float> anchor_mask_errors(m_anchor_lib.anchors.size(), 0); 
	for (int anchor_id = 0; anchor_id < m_anchor_lib.anchors.size(); anchor_id++)
	{
		optimizeAnchor(anchor_id); 
		anchor_errors[anchor_id] = evaluate_error(); 
		anchor_mask_errors[anchor_id] = evaluate_mask_error(); 

		std::cout << "anchor " << std::setw(2) << anchor_id << " : " << anchor_errors[anchor_id] 
			<< " , " << anchor_mask_errors[anchor_id] << std::endl; 
	}

	int min_id = -1; 
	int min_loss = 100000;
	for (int i = 0; i < anchor_errors.size(); i++)
	{
		float err = anchor_errors[i] + anchor_mask_errors[i] * 50;
		//float err = anchor_errors[i] + anchor_mask_errors[i] * 5; 
		if (err < min_loss)
		{
			min_loss = err;
			min_id = i;
		}
	}
	m_anchor_id = min_id;
	return min_id; 
}

float PigSolverDevice::evaluate_error()
{
	// joint error 
	std::vector<float> joint_2d_errors(m_skelTopo.joint_num, 0); 
	std::vector<int> joint_2d_valid(m_skelTopo.joint_num, 0); 

	int N = m_skelCorr.size(); 
	std::vector<Eigen::Vector3f> skel3d = getRegressedSkel_host(); 
	for (int viewid = 0; viewid < m_source.view_ids.size(); viewid++)
	{
		int camid = m_source.view_ids[viewid];
		for (int i = 0; i < N; i++)
		{
			int t = m_skelCorr[i].target;
			if (m_source.dets[viewid].keypoints[t](2) < m_skelTopo.kpt_conf_thresh[t]) continue;
			Camera cam = m_cameras[camid];
			Eigen::Vector3f joint2d = cam.K * (cam.R * skel3d[t] + cam.T);
			joint2d = joint2d / joint2d(2); 
			joint_2d_errors[t] += (joint2d.segment<2>(0) - m_source.dets[viewid].keypoints[t].segment<2>(0)).norm();
			joint_2d_valid[t] += 1;
		}
	}
	float joint_2d_error_avg = 0; 
	for (int i = 0; i < joint_2d_errors.size(); i++)
	{
		if(joint_2d_valid[i] > 0)
		joint_2d_error_avg += joint_2d_errors[i] / joint_2d_valid[i];
	}
	return joint_2d_error_avg; 
}

void PigSolverDevice::CalcLambdaTerm(Eigen::MatrixXf& ATA)
{
	int M = m_poseToOptimize.size(); 
	int paramNum = 3 + 3 * M; 
	ATA = Eigen::MatrixXf::Identity(paramNum, paramNum); 
	// "pose_to_solve": [ 0, 2, 4, 5, 6, 7, 8, 13, 14, 15, 16, 38, 39, 40, 41, 54, 55, 56, 57, 21, 22, 23],
	//std::vector<float> rot_weights = { // size same to M 
	//	//0.0001, 0.01, 0.01,
	//	0.0001, 0.0001, 0.0001,
	//	0.0001, 0.0001, 0.0001, 0.0001,
	//	0.0001, 0.0001, 0.0001, 0.0001,
	//	0.0001, 0.0001, 0.0001, 0.0001,
	//	0.0001, 0.0001, 0.0001, 0.0001,
	//	0.0001, 0.0001,0.0001
	//	//2,2,2
	//};
	//for (int i = 0; i < rot_weights.size(); i++)
	//{
	//	ATA(3 + 3 * i, 3 + 3 * i) = rot_weights[i];
	//	ATA(3 + 3 * i + 1, 3 + 3 * i + 1) = rot_weights[i];
	//	ATA(3 + 3 * i + 2, 3 + 3 * i + 2) = rot_weights[i];
	//}
}



void PigSolverDevice::calcAnchorTerm_host(int anchorid,
	const Eigen::VectorXf& theta, 
	Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size(); 
	ATA = Eigen::MatrixXf::Identity(paramNum, paramNum); 
	ATb = Eigen::VectorXf::Zero(paramNum); 

	const AnchorPoseType& A = m_anchor_lib.anchors[anchorid];
	
	ATb(2) = A.translation(2) - theta(2);
	ATb(4) = A.pose[0](1) - theta(4);
	ATb(5) = A.pose[0](2) - theta(5); 
	for (int i = 1; i < m_poseToOptimize.size(); i++)
	{
		int jid = m_poseToOptimize[i];
		float weight = 1; 

		ATb.segment<3>(3 + 3 * i) = A.pose[jid] - theta.segment<3>(3+3*i); 
	}

	float all_conf = 0; 
	for (int i = 0; i < m_det_confs.size(); i++)
	{
		//std::cout << " i : " << i << " , " << m_det_confs[i] << std::endl; 
		all_conf += m_det_confs[i];
	}
	//std::cout << "pig " << m_pig_id << "  " << all_conf << std::endl; 

	float weight;
	if (all_conf > 130) weight = 0.001; 
	else weight = 0.1; 
	if (m_det_confs[0] >= 2 && m_det_confs[1] >= 2 && m_det_confs[2] >= 2
		&& m_det_confs[3] >= 2 && m_det_confs[4] >= 2)
	{
		std::vector<int> ignore = {19,20,21};
		for (const int & k : ignore)
		{
			ATA.middleRows<3>(3 + 3 * k) *= 0.001;
			ATb.segment<3>(3 + 3 * k) *= 0.001;
		}
	}

	if (m_det_confs[5] >= 3 && m_det_confs[7] >= 3 && m_det_confs[9] >= 3)
	{
		std::vector<int> ignore = {7,8,9,10 };

		float w_leg = weight; 
		if (m_det_confs[5] + m_det_confs[7] + m_det_confs[9] > 18) w_leg *= 0.1;
		for (const int & k : ignore)
		{
			ATA.middleRows<3>(3 + 3 * k) *= w_leg;
			ATb.segment<3>(3 + 3 * k) *= w_leg;
		}
	}
	if (m_det_confs[6] >= 3 && m_det_confs[8] >= 3 && m_det_confs[10] >= 3)
	{
		std::vector<int> ignore = { 3,4,5,6 };
		float w_leg = weight;
		if (m_det_confs[6] + m_det_confs[8] + m_det_confs[10] > 18) w_leg *= 0.1;
		for (const int & k : ignore)
		{
			ATA.middleRows<3>(3 + 3 * k) *= w_leg;
			ATb.segment<3>(3 + 3 * k) *= w_leg;
		}
	}
	if (m_det_confs[11] >= 3 && m_det_confs[13] >= 3 && m_det_confs[15] >= 3)
	{
		std::vector<int> ignore = { 15,16,17,18 };
		float w_leg = weight;
		if (m_det_confs[11] + m_det_confs[13] + m_det_confs[15] > 18) w_leg *= 0.1;
		for (const int & k : ignore)
		{
			ATA.middleRows<3>(3 + 3 * k) *= w_leg;
			ATb.segment<3>(3 + 3 * k) *= w_leg;
		}
	}
	if (m_det_confs[12] >= 3 && m_det_confs[14] >= 3 && m_det_confs[16] >= 3)
	{
		std::vector<int> ignore = { 11,12,13,14 };
		float w_leg = weight;
		if (m_det_confs[12] + m_det_confs[14] + m_det_confs[16] > 18) w_leg *= 0.1;
		for (const int & k : ignore)
		{
			ATA.middleRows<3>(3 + 3 * k) *= w_leg;
			ATb.segment<3>(3 + 3 * k) *= w_leg;
		}
	}
	if (m_det_confs[20] >= 2 && m_det_confs[18] >= 2 && m_det_confs[0] >= 2)
	{
		ATb.segment<3>(3) *= 0.0;
		ATA.middleRows(3, 3) *= 0.0;
		ATb.segment<3>(0) *= 0.0;
		ATA.middleRows(0, 3) *= 0.0 ;
	}
}

// This jacobi is 
void PigSolverDevice::calcAnchorTermHeight_host(
	int anchorid, const Eigen::MatrixXf& Jacobi3d, 
	Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb
)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size();
	ATA = Eigen::MatrixXf::Identity(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);
	int N = m_skelCorr.size(); 
	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(N, paramNum); 
	Eigen::VectorXf b = Eigen::VectorXf::Zero(N); 
	std::vector<Eigen::Vector3f> skel = getRegressedSkel_host(); 
	for (int i = 0; i < N; i++)
	{
		A.row(i) = Jacobi3d.row(3 * i + 2) * 5; 
		int t = m_skelCorr[i].target; 
		b(i) = m_host_scale * m_anchor_lib.anchors[anchorid].joint_positions_23[t](2) - skel[t](2); 
		b(i) = b(i) * 5; 
	}
	for (int i = 0; i < N; i++)
	{
		int t = m_skelCorr[i].target;
		if (m_det_confs[t] >= 1) {
			b(i) = 0; 
			continue;
		}
	}
	ATA = A.transpose() * A; 
	ATb = A.transpose() * b; 
}


void PigSolverDevice::CalcAnchorRegTerm(const Eigen::VectorXf& theta, 
	Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, 
	int anchor_id, bool adaptive_weight)
{
	Eigen::VectorXf theta0 = Eigen::VectorXf::Zero(theta.rows());
	for (int i = 0; i < m_poseToOptimize.size(); i++)
	{
		int jid = m_poseToOptimize[i];
		theta0.segment<3>(3 + 3 * i) = m_anchor_lib.anchors[anchor_id].pose[jid];
	}
	int paramNum = 3 + 3 * m_poseToOptimize.size();
	ATA = Eigen::MatrixXf::Identity(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);
	ATb = theta0-theta;
	ATb.segment<6>(0).setZero();

	if (!adaptive_weight) return;
	if (m_det_confs[0] >= 2 && m_det_confs[1] >= 2 && m_det_confs[2] >= 2
		&& m_det_confs[3] >= 2 && m_det_confs[4] >= 2)
	{
		std::vector<int> ignore = { 19,20,21,1,2 }; // 21,22,23
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

float PigSolverDevice::evaluate_mask_error()
{
	float mask_error = 0; 

	renderDepths();

	for (int viewid = 0; viewid < m_rois.size(); viewid++)
	{
		int camid = m_source.view_ids[viewid];
		Camera cam = m_cameras[camid];
		
		std::vector<unsigned char> visibility(m_vertexNum, 0);
		check_visibility(d_depth_renders[camid], 1920, 1080, m_device_verticesPosed,
			cam.K, cam.R, cam.T, visibility);

		
		int valid_view = 0; 
		float all_chamfers = 0;

		for (int i = 0; i < m_vertexNum; i++)
		{
			if (m_host_bodyParts[i] == TAIL || m_host_bodyParts[i] == L_EAR || 
				m_host_bodyParts[i] == R_EAR) continue; // ignore tail and ear
			if (visibility[i] == 0) continue; //only consider visible parts 
			Eigen::Vector3f xlocal =  m_host_verticesPosed[i] ; 
			int m = m_rois[viewid].queryMask(xlocal); 
			float d = m_rois[viewid].queryChamfer(xlocal); 

			if (d < -9999) {
				continue;
			}
			if (m == 2 || m == 3 || m < -1) {

				continue;
			}
			
			if (d >= 0) {
				all_chamfers += 0;
			}
			else
			{
				all_chamfers += fabsf(d); 

			}
			valid_view += 1; 

		}
		mask_error += all_chamfers / valid_view; 
	}
	return mask_error; 
}

float PigSolverDevice::approxIOU(int viewid)
{
	// assume we already have d_depth_interact
	float total_overlay = 0;
	float total_visible = 0; 

	int camid = m_source.view_ids[viewid];
	Camera cam = m_cameras[camid];

	std::vector<unsigned char> visibility(m_vertexNum, 0);
	check_visibility(d_depth_renders_interact[camid], 1920, 1080, m_device_verticesPosed,
		cam.K, cam.R, cam.T, visibility);

	for (int i = 0; i < m_vertexNum; i++)
	{
		if (m_host_bodyParts[i] == TAIL) continue; // ignore tail and ear
		if (visibility[i] == 0) continue; //only consider visible parts 
		Eigen::Vector3f xlocal = m_host_verticesPosed[i];
		int m = m_rois[viewid].queryMask(xlocal);
		total_visible += 1; 
		if (m == 2 || m == 3 || m < -1) 
		{
			continue;
		}

		total_overlay += 1; 

	}
	
	return total_overlay / total_visible;
}

float ComputeTempWeight(float height)
{
	float a = 0.15 - height;
	if (a < 0) return 0;
	if (a > 0.1) a = 0.1;
	a = a / 0.1; 
	a = pow(a, 8);
	return a * 10;
}



// in use
float PigSolverDevice::optimizePoseSilWithAnchorOneStep(int iter)
{
#ifdef DEBUG_SIL
	std::vector<cv::Mat> color_mask_dets;
	for (int view = 0; view < m_rois.size(); view++)
	{
		int camid = m_rois[view].viewid;
		cv::Mat img(cv::Size(1920, 1080), CV_8UC3); img.setTo(255);
		my_draw_mask(img, m_rois[view].mask_list, Eigen::Vector3i(255, 0, 0), 0);
		color_mask_dets.push_back(img);
	}
#endif 
	int paramNum = 3 + 3 * m_poseToOptimize.size();
	int M = m_poseToOptimize.size();
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
	float track_radius = m_params.m_kpt_track_dist - iter * 10;
	track_radius = track_radius > 30 ? track_radius : 30;
	bool is_converge_radius = false;
	if (iter > 20) is_converge_radius = true;
	if (m_isReAssoc)
	{
		//std::cout << "pig: " << m_pig_id << "  iter: " << iter << std::endl; 
		Calc2dSkelProjectionTermReassoc(ATA_data, ATb_data, false);
	}
	else
	{
		Calc2dJointProjectionTerm(m_source, ATA_data, ATb_data, track_radius, false, is_converge_radius);
	}
	calcPoseJacobiPartTheta_device(d_J_joint, d_J_vert, false); // TODO: remove d_J_vert computation here. 
	d_J_joint.download(h_J_joint.data(), 3 * m_jointNum * sizeof(float));
	d_J_vert.download(h_J_vert.data(), 3 * m_vertexNum * sizeof(float)); 
	// compute terms

	Eigen::MatrixXf ATA_3d = Eigen::MatrixXf::Zero(paramNum, paramNum); 
	Eigen::VectorXf ATb_3d = Eigen::VectorXf::Zero(paramNum);
	auto skel_tri = directTriangulationHost(); 
	auto skel_reg = getRegressedSkel_host(); 
	calcSkel3DTerm_host(h_J_skel, skel_tri, ATA_3d, ATb_3d); 

	float lambda = m_params.m_lambda;
	float w_data = m_params.m_w_data_term;
	float w_sil; 
	if (iter < m_params.m_sil_step) w_sil = 0; 
	else w_sil = m_params.m_w_sil_term;
	float w_reg = m_params.m_w_reg_term;
	float w_temp = m_params.m_w_temp_term;
	float w_floor; 
	if (iter > 10) w_floor = m_params.m_w_floor_term * 10; 
	else  w_floor = m_params.m_w_floor_term;
	float w_collision; 
	if (iter < m_params.m_collision_step) 
		w_collision = 0; 
	else
		w_collision = m_params.m_w_collision_term; 

	float w_anchor = m_params.m_w_anchor_term; 

	Eigen::MatrixXf ATA_sil = Eigen::MatrixXf::Zero(paramNum, paramNum);
	Eigen::VectorXf ATb_sil = Eigen::VectorXf::Zero(paramNum);
	if (w_sil > 0)
	{
		renderDepths();
		if (m_use_gpu)
			CalcSilhouettePoseTerm(ATA_sil, ATb_sil, iter);
		else
			CalcSilouettePoseTerm_cpu(ATA_sil, ATb_sil, iter);
	}

	Eigen::MatrixXf ATA_floor;
	Eigen::VectorXf ATb_floor;
	CalcJointFloorTerm(ATA_floor, ATb_floor);

	Eigen::MatrixXf ATA_anchor = Eigen::MatrixXf::Zero(paramNum, paramNum);
	Eigen::VectorXf ATb_anchor = Eigen::VectorXf::Zero(paramNum);
	Eigen::MatrixXf ATA_anchor_h = Eigen::MatrixXf::Zero(paramNum, paramNum);
	Eigen::VectorXf ATb_anchor_h = Eigen::VectorXf::Zero(paramNum); 

	if (w_anchor > 0)
	{
		calcAnchorTerm_host(m_anchor_id, theta, ATA_anchor, ATb_anchor);
		calcAnchorTermHeight_host(m_anchor_id, h_J_skel, ATA_anchor_h, ATb_anchor_h);
		ATA_anchor += ATA_anchor_h;
		ATb_anchor += ATb_anchor_h;
	}

	Eigen::MatrixXf DTD;
	CalcLambdaTerm(DTD);

	Eigen::MatrixXf ATA_temp = Eigen::MatrixXf::Zero(paramNum, paramNum);
	Eigen::VectorXf ATb_temp = Eigen::VectorXf::Zero(paramNum);
	if (m_last_regressed_skel3d.size() > 0)
	{
		CalcJointTempTerm2(ATA_temp, ATb_temp, h_J_skel, m_last_regressed_skel3d);
	}

	float height = getAvgHeight(); 
	if (m_params.m_use_height_enhanced_temp)
	{
		w_temp = m_params.m_w_temp_term * (1 + ComputeTempWeight(height));
	}
	else
	{
		w_temp = m_params.m_w_temp_term; 
	}

	Eigen::MatrixXf ATA_reg;
	Eigen::VectorXf ATb_reg;

	if(m_params.m_use_bodyonly_reg)
		CalcRegTermBodyOnly(theta, ATA_reg, ATb_reg); 
	else {
		CalcRegTerm(theta, ATA_reg, ATb_reg, false);
	}

#ifdef USE_SIFT
	Eigen::MatrixXf ATA_sift; 
	Eigen::VectorXf ATb_sift; 
	CalcSIFTTerm(m_siftCorrs, ATA_sift, ATb_sift); 
#endif 

	float w_on_floor = m_params.m_w_on_floor_term;
	Eigen::MatrixXf ATA_on_floor; 
	Eigen::VectorXf ATb_on_floor; 
	CalcJointOnFloorTerm(ATA_on_floor, ATb_on_floor);

	Eigen::MatrixXf ATA_col, ATA_col2; 
	Eigen::VectorXf ATb_col, ATb_col2; 
	CalcCollisionJointTerm_cpu(h_J_joint, ATA_col, ATb_col); 

	CalcCollisionSurfaceTerm_cpu(h_J_vert, ATA_col2, ATb_col2);
	ATA_col += ATA_col2; 
	ATb_col += ATb_col2; 

	Eigen::MatrixXf H = ATA_sil * w_sil + ATA_reg * w_reg
		+ DTD * lambda
		+ ATA_data * w_data + ATA_anchor * w_anchor
		+ ATA_floor * w_floor
		+ ATA_temp * w_temp
		+ ATA_on_floor * w_on_floor
#ifdef USE_SIFT
		+ ATA_sift * w_sift
#endif 
		+ATA_col * w_collision
		; 
	Eigen::VectorXf b = ATb_sil * w_sil + ATb_reg * w_reg
		+ ATb_data * w_data + ATb_anchor * w_anchor
		+ ATb_floor * w_floor
		+ ATb_temp * w_temp
		+ ATb_on_floor * w_on_floor
#ifdef USE_SIFT
		+ ATb_sift * w_sift
#endif 
		+ ATb_col * w_collision
		; 

	Eigen::VectorXf delta = H.ldlt().solve(b);

	// update 
	m_host_translation += delta.segment<3>(0);
	for (int i = 0; i < M; i++)
	{
		int jIdx = m_poseToOptimize[i];
		m_host_poseParam[jIdx] += delta.segment<3>(3 + 3 * i);
	}

#ifdef VISUAL_LOSS
	std::cout << "pig " << m_pig_id << " iter: " << iter << std::endl; 
	std::cout << "   ATb_data: " << ATb_data.norm() << std::endl; 
	std::cout << "   ATb_sil : " << ATb_sil.norm() << std::endl; 
	std::cout << "   ATb_reg : " << ATb_reg.norm() << std::endl; 
	std::cout << "   ATb_anchor: " << ATb_anchor.norm() << std::endl; 
	std::cout << "   ATb_floor: " << ATb_floor.norm() << std::endl; 
	std::cout << "   ATb_temp: " << ATb_temp.norm() << std::endl; 
	std::cout << "   ATb_col: " << ATb_col.norm() << std::endl;
	std::cout << "      delta : " << delta.norm() << std::endl; 
	std::cout << "      b.norm: " << b.norm() << std::endl;
#endif 
#ifdef USE_SIFT
	std::cout << "ATb_sift: " << ATb_sift.norm() << std::endl; 
#endif 
	return delta.norm(); 
}

void PigSolverDevice::Calc2dSkelProjectionTermReassoc(
	Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, bool with_depth_weight)
{
	int paramNum = 3 + 3 * m_poseToOptimize.size(); 
	std::vector<Eigen::Vector3f> skel3d = getRegressedSkel_host();
	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum); 
	ATb = Eigen::VectorXf::Zero(paramNum); 
	for (int camid = 0; camid < m_cameras.size(); camid++)
	{
		Eigen::MatrixXf H;
		Eigen::VectorXf b; 
		Calc2DSkelTermReassoc_host(m_keypoints_reassociated[camid], skel3d, camid, h_J_skel, H, b);
		float weight = 1; 
		if (with_depth_weight)
		{
			if (m_depth_weight[camid] > 0.1)
				weight = 1 / m_depth_weight[camid];
		}

		ATA += H * weight; 
		ATb += b * weight; 

		//std::cout << "cam " << camid << " b.norm: " << b.norm() << "  w: " << weight << std::endl;
	}
}


void PigSolverDevice::Calc2DSkelTermReassoc_host(const std::vector<Eigen::Vector3f>& skel_det, const std::vector<Eigen::Vector3f>& skel3d, int camid,
	const Eigen::MatrixXf& Jacobi3d, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	const Camera& cam = m_cameras[camid];
	Eigen::Matrix3f R = cam.R; 
	Eigen::Matrix3f K = cam.K; 
	Eigen::Vector3f T = cam.T; 
	K.row(0) /= 1920; 
	K.row(1) /= 1080;
	int N = m_skelCorr.size();
	int M = m_poseToOptimize.size(); 
	int paramNum = 3 + 3 * M; 
	Eigen::VectorXf r = Eigen::VectorXf::Zero(2 * N); 
	Eigen::MatrixXf J = Eigen::MatrixXf::Zero(2 * N, 3 + 3 * M); 
	
	std::vector<Eigen::Vector3f> skels2d; 
	project(cam, skel3d, skels2d); 
	for (int i = 0; i < N; i++)
	{
		const CorrPair& P = m_skelCorr[i];
		int t = P.target;
		if (skel_det[t](2) == 0) continue;
		//if (camid == 3) // top view do not trust leg
		//{
		//	if (t >= 5 && t <= 16) continue; 
		//}
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
		det_u(0) = skel_det[t](0) / 1920;
		det_u(1) = skel_det[t](1) / 1080;
		r.segment<2>(2 * i) = P.weight * (u - det_u);
	}

	ATb = -J.transpose() * r;
	ATA = J.transpose() * J;
}

void PigSolverDevice::projectSkels()
{
	m_skelProjs.resize(m_cameras.size()); 
	auto skel = getRegressedSkel_host();
	for (int camid = 0; camid < m_cameras.size(); camid++)
	{
		m_skelProjs[camid].resize(m_skelTopo.joint_num, Eigen::Vector3f::Zero());
		for (int kpt_id = 0; kpt_id < m_skelTopo.joint_num; kpt_id++)
		{
			if (skel[kpt_id].norm() == 0) continue;
			Eigen::Vector3f p = skel[kpt_id];
			m_skelProjs[camid][kpt_id] = project(m_cameras[camid], p);
		}
	}
}

Eigen::Vector4f estimateBox(const std::vector<Eigen::Vector3f>& keypoints)
{
	float minx = 1920; 
	float miny = 1080;
	float maxx = 0; 
	float maxy = 0;
	for(int i = 0; i < keypoints.size(); i++)
	{
		if (keypoints[i](2) == 0) continue; 
		minx = minx > keypoints[i](0) ? keypoints[i](0) : minx; 
		maxx = maxx > keypoints[i](0) ? maxx : keypoints[i](0);
		miny = miny > keypoints[i](1) ? keypoints[i](1) : miny; 
		maxy = maxy > keypoints[i](1) ? maxy : keypoints[i](1); 
	}
	minx = minx - 50 > 0 ? minx - 50 : 0;
	miny = miny - 50 > 0 ? miny - 50 : 0; 
	maxx = maxx + 50 < 1919 ? maxx + 50 : 1919;
	maxy = maxy + 50 < 1079 ? maxy + 50 : 1079;

	Eigen::Vector4f box(minx, miny, maxx, maxy); 
	return box; 
}

bool isValidBox(const Eigen::Vector4f& box)
{
	if (box(0) >= box(2)) return false;
	if (box(1) >= box(3)) return false;
	return true; 
}

cv::Mat PigSolverDevice::debug_vis_reassoc_swap()
{
	std::vector<Eigen::Vector3i> m_CM;
	getColorMap("anliang_rgb", m_CM);
	std::vector<cv::Mat> m_imgdet;
	cloneImgs(m_rawimgs, m_imgdet);
	for (int camid = 0; camid < m_cameras.size(); camid++)
	{
		drawSkelDebug(m_imgdet[camid], m_keypoints_reassociated[camid], m_skelTopo);
	}

	std::vector<cv::Mat> crop_list(12);
	for (int k = 0; k < crop_list.size(); k++)
	{
		crop_list[k] = cv::Mat(cv::Size(256, 256), CV_8UC3);
		crop_list[k].setTo(cv::Scalar(255, 255, 255));
	}
	for (int camid = 0; camid < m_cameras.size(); camid++)
	{
		Eigen::Vector4f box = estimateBox(m_keypoints_reassociated[camid]);
		if (!isValidBox(box))continue; 
		cv::Mat raw_img = m_imgdet[camid];
		cv::Rect2i roi(box[0], box[1], box[2] - box[0], box[3] - box[1]);
		cv::Mat img = raw_img(roi);
		cv::Mat img2 = resizeAndPadding(img, 256, 256);
		crop_list[camid] = img2;
	}
	cv::Mat output;
	packImgBlock(crop_list, output);
	return output;
}