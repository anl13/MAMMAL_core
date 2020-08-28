#include "pigsolver.h"


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
		//cv::Mat img = m_rawImgs[camid].clone();
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
		ss_obj << "E:/pig_results/debug/sil_" << iter << ".obj";
		SaveObj(ss_obj.str());
#endif 

		// calc joint term 
		Eigen::MatrixXd poseJ3d;
		CalcSkelJacobiPartThetaByPairs(poseJ3d);
		Eigen::MatrixXd skel = getRegressedSkelbyPairs();
		Eigen::MatrixXd H1 = Eigen::MatrixXd::Zero(3 + 3 * M, 3 + 3 * M); // data term 
		Eigen::VectorXd b1 = Eigen::VectorXd::Zero(3 + 3 * M);  // data term 
		for (int k = 0; k < m_source.view_ids.size(); k++)
		{
			Eigen::MatrixXd H_view;
			Eigen::VectorXd b_view;
			CalcPose2DTermByPairs(k, skel, poseJ3d, H_view, b_view);
			H1 += H_view;
			b1 += b_view;
		}

		// render images 
		Model m3c;
		m3c.vertices = m_verticesFinal;
		m3c.faces = m_facesVert;
		ObjModel m4c;
		convert3CTo4C(m3c, m4c);

		animal_offscreen->SetIndices(m4c.indices);
		animal_offscreen->SetBuffer("positions", m4c.vertices);

		cv::Mat rendered_img(1920, 1080, CV_32FC4);
		std::vector<cv::Mat> rendered_imgs;
		rendered_imgs.push_back(rendered_img);
		const auto& cameras = m_cameras;
		std::vector<cv::Mat> renders;
		std::vector<cv::Mat> rend_bgr;
		for (int view = 0; view < m_rois.size(); view++)
		{
			auto cam = m_rois[view].cam;
			Eigen::Matrix3f R = cam.R.cast<float>();
			Eigen::Vector3f T = cam.T.cast<float>();

			animal_offscreen->_SetViewByCameraRT(cam.R, cam.T);
			Eigen::Matrix4f camview = Eigen::Matrix4f::Identity();
			camview.block<3, 3>(0, 0) = cam.R.cast<float>();
			camview.block<3, 1>(0, 3) = cam.T.cast<float>();
			nanogui::Matrix4f camview_nano = eigen2nanoM4f(camview);
			animal_offscreen->SetUniform("view", camview_nano);
			animal_offscreen->DrawOffscreen();
			animal_offscreen->DownloadRenderingResults(rendered_imgs);
			std::vector<cv::Mat> channels(4);
			cv::split(rendered_imgs[0], channels);
			cv::Mat depth = -channels[2];
			renders.push_back(depth);
			rend_bgr.emplace_back(fromDeptyToColorMask(depth));
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
		ss << "E:/pig_results/debug/" << std::setw(6) << std::setfill('0')
			<< iter << ".jpg";
		cv::imwrite(ss.str(), blended);
#endif 
		// compute terms
		Eigen::VectorXd theta(3 + 3 * M);
		theta.segment<3>(0) = m_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
		}

		Eigen::MatrixXd ATA;
		Eigen::VectorXd ATb;
		CalcSilhouettePoseTerm(renders, ATA, ATb, iter);
		double lambda = 0.005;
		double w_joint = 0.01;
		double w1 = 1;
		double w_reg = 1;
		double w_temp = 0;
		Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXd H_reg = DTD;  // reg term 
		Eigen::VectorXd b_reg = -theta; // reg term 
		Eigen::MatrixXd H_temp = DTD;
		Eigen::VectorXd b_temp = theta_last - theta;
		Eigen::MatrixXd H = ATA * w1 + H_reg * w_reg
			+ DTD * lambda + H_temp * w_temp
			+ H1 * w_joint;
		Eigen::VectorXd b = ATb * w1 + b_reg * w_reg
			+ b_temp * w_temp
			+ b1 * w_joint;
		Eigen::VectorXd delta = H.ldlt().solve(b);

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
	Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb, int iter)
{
	int M = 3 + 3 * m_poseToOptimize.size();
	ATA = Eigen::MatrixXd::Zero(M, M);
	ATb = Eigen::VectorXd::Zero(M);
	Eigen::MatrixXd J_joint, J_vert;
	CalcPoseJacobiPartTheta(J_joint, J_vert);

	//// visualize 
	//std::vector<cv::Mat> chamfers_vis; 
	//std::vector<cv::Mat> chamfers_vis_det; 
	//std::vector<cv::Mat> gradx_vis;
	//std::vector<cv::Mat> grady_vis;
	//std::vector<cv::Mat> diff_vis; 
	//std::vector<cv::Mat> diff_xvis; 
	//std::vector<cv::Mat> diff_yvis;

	double total_r = 0;

	for (int roiIdx = 0; roiIdx < m_rois.size(); roiIdx++)
	{
		if (m_rois[roiIdx].valid < 0.6) {
			std::cout << "view " << roiIdx << " is invalid. " << m_rois[roiIdx].valid << std::endl;
			continue;
		}
		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m_vertexNum, M);
		Eigen::VectorXd r = Eigen::VectorXd::Zero(m_vertexNum);
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
		Eigen::Matrix3d R = cam.R;
		Eigen::Matrix3d K = cam.K;
		Eigen::Vector3d T = cam.T;

		double wc = 200.0 / m_rois[roiIdx].area;
		//std::cout << "wc " << roiIdx << " : " << wc << std::endl; 
		for (int i = 0; i < m_vertexNum; i++)
		{
			double w;
			if (m_bodyParts[i] == R_F_LEG || m_bodyParts[i] == R_B_LEG ||
				m_bodyParts[i] == L_F_LEG || m_bodyParts[i] == L_B_LEG) w = 5;
			else w = 1;
			Eigen::Vector3d x0 = m_verticesFinal.col(i);
			// check visibiltiy 
			Camera & cam = m_rois[roiIdx].cam;
			float depth_value = queryPixel(depths[roiIdx], x0, cam);
			Eigen::Vector3d x0_local = cam.R * x0 + cam.T;
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

			Eigen::MatrixXd block2d = Eigen::MatrixXd::Zero(2, M);
			Eigen::Vector3d x_local = K * (R * x0 + T);
			Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2, 3);
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

