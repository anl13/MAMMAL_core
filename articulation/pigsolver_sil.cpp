#include "pigsolver.h"

#define DEBUG_SIL 

void PigSolver::optimizePoseSilhouette(int maxIter)
{
	int iter = 0;

#ifdef DEBUG_SIL
	std::vector<cv::Mat> color_mask_dets;
	std::vector<cv::Mat> raw_ims;
	for (int view = 0; view < m_rois.size(); view++)
	{
		int camid = m_rois[view].viewid;
		raw_ims.push_back(m_rawImgs[camid]);
		cv::Mat img(cv::Size(1920, 1080), CV_8UC3); img.setTo(255);
		my_draw_mask(img, m_rois[view].mask_list, Eigen::Vector3i(255, 0, 0), 0);
		color_mask_dets.push_back(img);
	}
#endif 

	int M = m_poseToOptimize.size();
	for (; iter < maxIter; iter++)
	{
		std::cout << "ITER: " << iter << std::endl;

		UpdateVertices();
#ifdef DEBUG_SIL
		std::stringstream ss_obj;
		ss_obj << "G:/pig_results/debug/sil_" << iter << ".obj";
		SaveObj(ss_obj.str());
#endif 

		// calc joint term 
		Eigen::MatrixXf poseJ3d;
		CalcSkelJacobiPartThetaByPairs(poseJ3d);
		Eigen::MatrixXf skel = getRegressedSkelbyPairs();
		Eigen::MatrixXf H1 = Eigen::MatrixXf::Zero(3 + 3 * M, 3 + 3 * M); // data term 
		Eigen::VectorXf b1 = Eigen::VectorXf::Zero(3 + 3 * M);  // data term 
		for (int k = 0; k < m_source.view_ids.size(); k++)
		{
			Eigen::MatrixXf H_view;
			Eigen::VectorXf b_view;
			CalcPose2DTermByPairs(k, skel, poseJ3d, H_view, b_view);
			H1 += H_view;
			b1 += b_view;
		}

		UpdateNormalFinal(); 
		// render images 
		mp_renderEngine->meshObjs.clear();
		RenderObjectMesh* p_model = new RenderObjectMesh(); 
		p_model->SetVertices(m_verticesFinal); 
		p_model->SetFaces(m_facesVert); 
		p_model->SetNormal(m_normalFinal); 
		p_model->SetColors(m_normalFinal); 
		mp_renderEngine->meshObjs.push_back(p_model); 

		cv::Mat depth_img; 
		depth_img.create(cv::Size(1920, 1080), CV_32FC1); 
		const auto& cameras = m_cameras;
		std::vector<cv::Mat> renders;
		std::vector<cv::Mat> rend_bgr;
		for (int view = 0; view < m_rois.size(); view++)
		{
			auto cam = m_rois[view].cam;
			Eigen::Matrix3f R = cam.R.cast<float>();
			Eigen::Vector3f T = cam.T.cast<float>();
			mp_renderEngine->s_camViewer.SetExtrinsic(R, T); 

			float * depth_device = mp_renderEngine->renderDepthDevice();
			cudaMemcpy(depth_img.data, depth_device, depth_img.cols * depth_img.rows * sizeof(float),
				cudaMemcpyDeviceToHost);
			
			renders.push_back(depth_img);
			rend_bgr.emplace_back(fromDepthToColorMask(depth_img));
		}

#ifdef DEBUG_SIL
		// test 
		cv::Mat pack_render;
		packImgBlock(rend_bgr, pack_render);
		cv::Mat rawpack;
		packImgBlock(raw_ims, rawpack);
		cv::Mat blended;
		//blended = overlay_renders(rawpack, pack_render, 0);
		cv::Mat pack_det;
		packImgBlock(color_mask_dets, pack_det);
		blended = pack_render * 0.5 + pack_det * 0.5;
		std::stringstream ss;
		ss << "G:/pig_results/debug/" << std::setw(6) << std::setfill('0')
			<< iter << ".jpg";
		cv::imwrite(ss.str(), blended);
#endif 
		// compute terms
		Eigen::VectorXf theta(3 + 3 * M);
		theta.segment<3>(0) = m_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
		}

		Eigen::MatrixXf ATA;
		Eigen::VectorXf ATb;
		CalcSilhouettePoseTerm(renders, ATA, ATb, iter);
		float lambda = 0.005;
		float w_joint = 0.01;
		float w1 = 1;
		float w_reg = 1;
		float w_temp = 0;
		Eigen::MatrixXf DTD = Eigen::MatrixXf::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXf H_reg = DTD;  // reg term 
		Eigen::VectorXf b_reg = -theta; // reg term 
		Eigen::MatrixXf H_temp = DTD;
		Eigen::VectorXf b_temp = theta_last - theta;
		Eigen::MatrixXf H = ATA * w1 + H_reg * w_reg
			+ DTD * lambda + H_temp * w_temp
			+ H1 * w_joint;
		Eigen::VectorXf b = ATb * w1 + b_reg * w_reg
			+ b_temp * w_temp
			+ b1 * w_joint;
		Eigen::VectorXf delta = H.ldlt().solve(b);

		std::cout << "reg  term b: " << b_reg.norm() << std::endl;
		std::cout << "temp term b: " << b_temp.norm() << std::endl;
		// update 
		m_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
		}
	}

	theta_last.segment<3>(0) = m_translation;
	for (int i = 0; i < M; i++)
	{
		int jIdx = m_poseToOptimize[i];
		theta_last.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
	}
}


void PigSolver::CalcSilhouettePoseTerm(
	const std::vector<cv::Mat>& depths,
	Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, int iter)
{
	int M = 3 + 3 * m_poseToOptimize.size();
	ATA = Eigen::MatrixXf::Zero(M, M);
	ATb = Eigen::VectorXf::Zero(M);
	Eigen::MatrixXf J_joint, J_vert;
	CalcPoseJacobiPartTheta(J_joint, J_vert, true);

	//// visualize 
	//std::vector<cv::Mat> chamfers_vis; 
	//std::vector<cv::Mat> chamfers_vis_det; 
	//std::vector<cv::Mat> gradx_vis;
	//std::vector<cv::Mat> grady_vis;
	//std::vector<cv::Mat> diff_vis; 
	//std::vector<cv::Mat> diff_xvis; 
	//std::vector<cv::Mat> diff_yvis;

	float total_r = 0;

	std::cout << "m_rois.size() : " << m_rois.size() << std::endl; 
	for (int roiIdx = 0; roiIdx < m_rois.size(); roiIdx++)
	{
		if (m_rois[roiIdx].valid < 0.6) {
			std::cout << "view " << roiIdx << " is invalid. " << m_rois[roiIdx].valid << std::endl;
			continue;
		}
		Eigen::MatrixXf A = Eigen::MatrixXf::Zero(m_vertexNum, M);
		Eigen::VectorXf r = Eigen::VectorXf::Zero(m_vertexNum);
		cv::Mat P = computeSDF2dFromDepthf(depths[roiIdx]);
		//chamfers_vis.emplace_back(visualizeSDF2d(P));
		//chamfers_vis_det.emplace_back(visualizeSDF2d(m_rois[roiIdx].chamfer));
		//diff_vis.emplace_back(visualizeSDF2d(P - m_rois[roiIdx].chamfer, 32));

		cv::Mat Pdx, Pdy;
		computeGradient(P, Pdx, Pdy);
		//gradx_vis.emplace_back(visualizeSDF2d(Pdx, 32));
		//grady_vis.emplace_back(visualizeSDF2d(Pdy, 32));
		//diff_xvis.emplace_back(visualizeSDF2d(Pdx - m_rois[roiIdx].gradx, 32));
		//diff_yvis.emplace_back(visualizeSDF2d(Pdy - m_rois[roiIdx].grady, 32));

		auto cam = m_rois[roiIdx].cam;
		Eigen::Matrix3f R = cam.R;
		Eigen::Matrix3f K = cam.K;
		Eigen::Vector3f T = cam.T;

		//float wc = 200.0 / m_rois[roiIdx].area;
		float wc = 0.01; 
		std::cout << "wc " << roiIdx << " : " << wc << std::endl; 
		for (int i = 0; i < m_vertexNum; i++)
		{
			//float w;
			//if (m_bodyParts[i] == R_F_LEG || m_bodyParts[i] == R_B_LEG ||
			//	m_bodyParts[i] == L_F_LEG || m_bodyParts[i] == L_B_LEG) w = 5;
			//else w = 1;
			if(m_bodyParts[i] == TAIL || m_bodyParts[i]==L_EAR || m_bodyParts[i] == R_EAR) continue; 
			float w = 1;

			Eigen::Vector3f x0 = m_verticesFinal.col(i);
			// check visibiltiy 
			Camera & cam = m_rois[roiIdx].cam;
			float depth_value = queryPixel(depths[roiIdx], x0, cam);
			Eigen::Vector3f x0_local = cam.R * x0 + cam.T;
			bool visible;
			if (abs(x0_local(2) - depth_value) < 0.02) visible = true;
			else visible = false;
			if (!visible) continue;

			int m = m_rois[roiIdx].queryMask(x0);
			// TODO: 20200602 check occlusion
			if (m == 2 || m == 3) continue;
			// TODO: 20200501 use mask to check visibility 
			float d = m_rois[roiIdx].queryChamfer(x0);
			if (d < -9999) continue;
			float ddx = queryPixel(m_rois[roiIdx].gradx, x0, m_rois[roiIdx].cam);
			float ddy = queryPixel(m_rois[roiIdx].grady, x0, m_rois[roiIdx].cam);
			float p = queryPixel(P, x0, m_rois[roiIdx].cam);
			float pdx = queryPixel(Pdx, x0, m_rois[roiIdx].cam);
			float pdy = queryPixel(Pdy, x0, m_rois[roiIdx].cam);
			if (p > 10) continue; // only consider contours for loss 

			Eigen::MatrixXf block2d = Eigen::MatrixXf::Zero(2, M);
			Eigen::Vector3f x_local = K * (R * x0 + T);
			Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
			D(0, 0) = 1 / x_local(2);
			D(1, 1) = 1 / x_local(2);
			D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
			D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));
			block2d = D * K * R * J_vert.middleRows(3 * i, 3);
			r(i) = w * (p - d);
			A.row(i) = w * (block2d.row(0) * (ddx)+block2d.row(1) * (ddy));
		}
		A = wc * A;
		r = wc * r;
		//std::cout << "r.norm() : " << r.norm() << std::endl;
		total_r += r.norm();
		ATA += A.transpose() * A;
		ATb += A.transpose() * r;
	}

	std::cout << "sil  term b: " << total_r << std::endl;
	//cv::Mat packP;
	//packImgBlock(chamfers_vis, packP);
	//cv::Mat packD;
	//packImgBlock(chamfers_vis_det, packD);
	//std::stringstream ssp;
	//ssp << "E:/debug_pig3/iters_p/" << iter << ".jpg";
	//cv::imwrite(ssp.str(), packP);
	//std::stringstream ssd; 
	//ssd << "E:/debug_pig3/iters_d/" << iter << ".jpg";
	//cv::imwrite(ssd.str(), packD);
	//cv::Mat packX, packY;
	//packImgBlock(gradx_vis, packX);
	//packImgBlock(grady_vis, packY);
	//std::stringstream ssx, ssy;
	//ssx << "E:/debug_pig3/iters_p/gradx_" << iter << ".jpg";
	//ssy << "E:/debug_pig3/iters_p/grady_" << iter << ".jpg";
	//cv::imwrite(ssx.str(), packX);
	//cv::imwrite(ssy.str(), packY);

	//cv::Mat packdiff; packImgBlock(diff_vis, packdiff);
	//std::stringstream ssdiff;
	//ssdiff << "E:/debug_pig3/diff/diff_" << iter << ".jpg";
	//cv::imwrite(ssdiff.str(), packdiff);

	//cv::Mat packdiffx; packImgBlock(diff_xvis, packdiffx);
	//std::stringstream ssdifx;
	//ssdifx << "E:/debug_pig3/diff/diffx_" << iter << ".jpg";
	//cv::imwrite(ssdifx.str(), packdiffx);

	//cv::Mat packdiffy; packImgBlock(diff_yvis, packdiffy);
	//std::stringstream ssdify;
	//ssdify << "E:/debug_pig3/diff/diffy_" << iter << ".jpg";
	//cv::imwrite(ssdify.str(), packdiffy);
}

