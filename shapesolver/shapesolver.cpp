#include "shapesolver.h"
#include "../utils/image_utils.h"
#include "../utils/timer_util.h"

ShapeSolver::ShapeSolver(const std::string& _configFile) : PigSolver(_configFile)
{

}

ShapeSolver::~ShapeSolver()
{

}

void ShapeSolver::CalcPoseTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	Eigen::MatrixXf J;
	Eigen::MatrixXf _Jjoint;
	CalcPoseJacobiPartTheta(_Jjoint, J, true);
	int M = J.cols();
	//Eigen::VectorXd b = Eigen::VectorXd::Zero(m_vertexNum);
	Eigen::VectorXf b = Eigen::VectorXf::Zero(m_vertexNum * 3);
	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(m_vertexNum * 3, M);
	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++)
	{
		const int tIdx = m_corr[sIdx];
		const float wd = m_wDeform[sIdx];
		if (tIdx == -1 || wd < FLT_EPSILON) continue;
		const auto tn = m_tarModel->normals.col(tIdx);
		const auto iv = m_verticesFinal.col(sIdx);
		const auto tv = m_tarModel->vertices.col(tIdx);
		//A.middleRows(sIdx, 1) =
		//	tn.transpose() * J.middleRows(3 * sIdx, 3);
		//b(sIdx) = wd * (tn.dot(tv - iv));
		A.middleRows(sIdx * 3, 3) = J.middleRows(3 * sIdx, 3);
		b.segment<3>(sIdx * 3) = tv - iv;
	}
	Eigen::MatrixXf AT = A.transpose();
	ATA = AT * A;
	ATb = AT * b;

	std::cout << "pose r: " << b.norm() << std::endl;
}

void ShapeSolver::CalcDvDSe3()
{
	int N = m_vertexNum; 
	int M = mp_nodeGraph->nodeIdx.size(); 
	dv_dse3.resize(3 * N, 6 * M); 
	dv_dse3.setZero(); 
	dvs_dse3.resize(3 * N, 6 * M); 
	dvs_dse3.setZero(); 
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> globalAffineNormalized = m_globalAffine;
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		globalAffineNormalized.block<3, 1>(0, jointId * 4 + 3) -= (m_globalAffine.block<3, 3>(0, jointId * 4)*m_jointsShaped.col(jointId));
	}

	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++) 
	{
		//if (m_bodyParts[sIdx] != HEAD) continue; 
		const auto iv = m_verticesFinal.col(sIdx);
		const auto iv1 = m_verticesDeformed.col(sIdx); 
		Eigen::Matrix<float, 4, 4, Eigen::ColMajor> globalAffineAverage;
		Eigen::Map<Eigen::VectorXf>(globalAffineAverage.data(), 16)
			= Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::ColMajor>>
			(globalAffineNormalized.data(), 16, m_jointNum) * (m_lbsweights.col(sIdx));

		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++)
		{
			const int ni = mp_nodeGraph->knn(i, sIdx);// nodeindex
			const float w = mp_nodeGraph->weight(i, sIdx); // node weight
			if (w < FLT_EPSILON || ni == -1)
				continue;

			int ni_vid = mp_nodeGraph->nodeIdx(ni); 
			//if (m_bodyParts[ni_vid] != HEAD) continue; 

			const int col = 6 * ni;
			Eigen::MatrixXf dv0 = Eigen::MatrixXf::Zero(3, 6);
			dv0(0, 1) = iv.z();
			dv0(0, 2) = -iv.y();
			dv0(1, 0) = -iv.z();
			dv0(1, 2) = iv.x();
			dv0(2, 0) = iv.y();
			dv0(2, 1) = -iv.x();
			dv0.middleCols(3, 3) = Eigen::Matrix3f::Identity();
			dv0 = globalAffineAverage.block<3, 3>(0, 0) * dv0;

			dv_dse3.block<3, 6>(3 * sIdx, 6 * ni) = dv0 * w;

			Eigen::MatrixXf dv1 = Eigen::MatrixXf::Zero(3, 6); 
			dv1(0, 1) = iv1.z();
			dv1(0, 2) = -iv1.y();
			dv1(1, 0) = -iv1.z();
			dv1(1, 2) = iv1.x();
			dv1(2, 0) = iv1.y();
			dv1(2, 1) = -iv1.x();
			dv1.middleCols(3, 3) = Eigen::Matrix3f::Identity();
			dvs_dse3.block<3, 6>(3 * sIdx, 6 * ni) = dv1 * w;
		}
	}
}

//void ShapeSolver::CalcShapeTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
//{
//	Eigen::MatrixXf jointJ, J;
//	CalcShapeJacobi(jointJ, J);
//	int M = J.cols();
//	//Eigen::VectorXd b = Eigen::VectorXd::Zero(m_vertexNum);
//	Eigen::VectorXf b = Eigen::VectorXf::Zero(m_vertexNum * 3);
//	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(m_vertexNum * 3, M);
//	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++)
//	{
//		const int tIdx = m_corr[sIdx];
//		const float wd = m_wDeform[sIdx];
//		if (tIdx == -1 || wd < FLT_EPSILON) continue;
//		const auto tn = m_tarModel->normals.col(tIdx);
//		const auto iv = m_verticesFinal.col(sIdx);
//		const auto tv = m_tarModel->vertices.col(tIdx);
//		//A.middleRows(sIdx, 1) =
//		//	tn.transpose() * J.middleRows(3 * sIdx, 3);
//		//b(sIdx) = wd * (tn.dot(tv - iv));
//		A.middleRows(sIdx * 3, 3) = J.middleRows(3 * sIdx, 3);
//		b.segment<3>(sIdx * 3) = tv - iv;
//	}
//	Eigen::MatrixXf AT = A.transpose();
//	ATA = AT * A;
//	ATb = AT * b;
//}

void ShapeSolver::CalcDeformTerm(
	Eigen::MatrixXf& ATA,
	Eigen::VectorXf& ATb
)
{
	if (m_wDeform.size() != m_vertexNum)
		m_wDeform = Eigen::VectorXf::Ones(m_vertexNum);

	std::vector<Eigen::Triplet<float>> triplets;
	Eigen::VectorXf b = Eigen::VectorXf::Zero(m_vertexNum);

	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++) {
		//if (m_bodyParts[sIdx] != HEAD) continue; 

		const int tIdx = m_corr[sIdx];
		float wd = m_wDeform(sIdx);

		if (tIdx == -1 || wd < FLT_EPSILON)
			continue;

		const auto tn = m_tarModel->normals.col(tIdx);
		const auto iv = m_verticesFinal.col(sIdx);
		const auto tv = m_tarModel->vertices.col(tIdx);


		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++)
		{
			const int ni = mp_nodeGraph->knn(i, sIdx);
			int ni_vid = mp_nodeGraph->nodeIdx(ni);
			//if (m_bodyParts[ni_vid] != HEAD) continue; 

			if (ni == -1) continue; 

			const int col = 6 * ni;
			Eigen::MatrixXf dv0;
			dv0.resize(3, 6); 
			dv0 = dv_dse3.block<3, 6>(3 * sIdx, 6 * ni); 

			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col, wd * (tn.x()*dv0(0, 0) + tn.y()*dv0(1, 0) + tn.z()*dv0(2, 0)))); // alpha_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 1, wd * (tn.x()*dv0(0, 1) + tn.y()*dv0(1, 1) + tn.z()*dv0(2, 1)))); // beta_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 2, wd * (tn.x()*dv0(0, 2) + tn.y()*dv0(1, 2) + tn.z()*dv0(2, 2)))); // gamma_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 3, wd * (tn.x()*dv0(0, 3) + tn.y()*dv0(1, 3) + tn.z()*dv0(2, 3)))); // tx_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 4, wd * (tn.x()*dv0(0, 4) + tn.y()*dv0(1, 4) + tn.z()*dv0(2, 4)))); // ty_ni
			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 5, wd * (tn.x()*dv0(0, 5) + tn.y()*dv0(1, 5) + tn.z()*dv0(2, 5)))); // tz_ni
		}
		b[sIdx] = wd * (tn.dot(iv - tv));
	}
	Eigen::SparseMatrix<float> A(m_vertexNum, mp_nodeGraph->nodeIdx.size() * 6);
	A.setFromTriplets(triplets.begin(), triplets.end());
	
	ATA = A.transpose() * A;
	ATb = -A.transpose() * b;
	std::cout << "deform r: " << b.norm() << std::endl;
}

#define DEBUG_SIL 

void ShapeSolver::CalcDeformTerm_sil(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	int nodeNum = mp_nodeGraph->nodeIdx.size();
	int paramNum = 6 * nodeNum;
	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);

	UpdateNormalFinal();

	mp_renderEngine->meshObjs.clear();
	RenderObjectMesh* p_model = new RenderObjectMesh();
	p_model->SetVertices(m_verticesFinal);
	p_model->SetFaces(m_facesVert);
	p_model->SetNormal(m_normalFinal);
	p_model->SetColors(m_normalFinal);
	mp_renderEngine->meshObjs.push_back(p_model);
	cv::Mat depth_img;
	depth_img.create(cv::Size(1920, 1080), CV_32FC1);
	const  auto & cameras = m_cameras;
	std::vector<cv::Mat> renders;
	for (int view = 0; view < m_rois.size(); view++)
	{
		Camera cam = m_rois[view].cam;
		mp_renderEngine->s_camViewer.SetExtrinsic(cam.R, cam.T);
		float * depth_device = mp_renderEngine->renderDepthDevice();
		cudaMemcpy(depth_img.data, depth_device, depth_img.cols * depth_img.rows * sizeof(float),
			cudaMemcpyDeviceToHost);
		renders.push_back(depth_img.clone());
	}
	std::vector<Eigen::Triplet<float>> triplets;


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
		int camid = m_source.view_ids[roiIdx];
		if (!in_list(camid, usedviews)) continue; 
		std::cout << "camid: " << camid << std::endl; 
		Eigen::MatrixXf A = Eigen::MatrixXf::Zero(m_vertexNum, paramNum);
		Eigen::VectorXf r = Eigen::VectorXf::Zero(m_vertexNum);
		cv::Mat P = computeSDF2dFromDepthf(renders[roiIdx]);
		Camera cam = m_rois[roiIdx].cam;
		Eigen::Matrix3f R = cam.R;
		Eigen::Matrix3f K = cam.K;
		Eigen::Vector3f T = cam.T;


#ifdef DEBUG_SIL
		chamfers_vis.emplace_back(visualizeSDF2d(P));

		cv::Mat chamfer_vis = visualizeSDF2d(m_rois[roiIdx].chamfer);

		chamfers_vis_det.push_back(chamfer_vis);
		diff_vis.emplace_back(visualizeSDF2d(P - m_rois[roiIdx].chamfer, 32));
#endif 

		int visible_num = 0; 
		for (int i = 0; i < m_vertexNum; i++)
		{
			if (m_bodyParts[i] == TAIL || m_bodyParts[i] == L_EAR
				|| m_bodyParts[i] == R_EAR) continue;
			Eigen::Vector3f x0 = m_verticesFinal.col(i);
			float depth_value = queryPixel(renders[roiIdx], x0, cam);
			Eigen::Vector3f x_local = K * (R * x0 + T);
			bool visible;
			if (abs(x_local(2) - depth_value) < 0.02) visible = true;
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
			if (p > 50) continue;

			visible_num += 1; 

			Eigen::MatrixXf block2d;
			Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
			D(0, 0) = 1 / x_local(2);
			D(1, 1) = 1 / x_local(2);
			D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
			D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));
			block2d = D * K * R * dv_dse3.middleRows(3 * i, 3);

			r(i) = p - d;
			A.row(i) = block2d.row(0) * ddx + block2d.row(1) * ddy;
		}
		std::cout << " ---visible num: " << visible_num << std::endl; 
		ATA += 0.0001 * A.transpose() * A;
		ATb += 0.0001 * A.transpose() * r;
	}
#ifdef DEBUG_SIL
	cv::Mat packP;
	packImgBlock(chamfers_vis, packP);
	cv::Mat packD;
	packImgBlock(chamfers_vis_det, packD);
	std::stringstream ssp;
	ssp << "G:/pig_results/" << 0 << "_rend_sdf_cpu.jpg";
	cv::imwrite(ssp.str(), packP);
	std::stringstream ssd;
	ssd << "G:/pig_results/" << 0 << "_det_sdf_cpu.jpg";
	cv::imwrite(ssd.str(), packD);

	cv::Mat packdiff; packImgBlock(diff_vis, packdiff);
	std::stringstream ssdiff;
	ssdiff << "G:/pig_results/diff_" << 0 << ".jpg";
	cv::imwrite(ssdiff.str(), packdiff);

#endif 
	std::cout << "ATb deform: " << ATb.norm() << std::endl; 
}

void ShapeSolver::CalcSymTerm(
	Eigen::MatrixXf& ATA,
	Eigen::VectorXf& ATb
)
{
	int symNum = m_symNum;
	int nodeNum = mp_nodeGraph->nodeIdx.size(); 
	/*symNum = 1; */
	Eigen::VectorXf b = Eigen::VectorXf::Zero(3 * m_vertexNum);
	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(3 * m_vertexNum, 6 * nodeNum);
	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++)
	{
		//if (m_bodyParts[sIdx] != HEAD) continue; 

		const int tIdx = m_symIdx[sIdx][0];
		float tW = m_symweights[sIdx][0];

		if (tIdx < 0 || tW <= 0) continue;
		if (tIdx == sIdx)
		{
			// do something 
			continue;
		}
		// else 
		const auto iv = m_verticesDeformed.col(sIdx);
		const auto tv = m_verticesDeformed.col(tIdx);

		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++) // 4
		{
			const int ni = mp_nodeGraph->knn(i, sIdx);
			const float wi = mp_nodeGraph->weight(i, sIdx);
			if (wi < FLT_EPSILON || ni == -1)
				continue;

			int coli = 6 * ni;
			int row = 3 * sIdx;
			A.block<3,6>(row,coli) = dvs_dse3.block<3,6>(row,coli)*tW;
			//triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 1, iv.z() * tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 2, -iv.y()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 3, 1 * tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 0, -iv.z()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 2, iv.x()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 4, 1 * tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 0, iv.y()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 1, -iv.x()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 5, 1 * tW));
		}

		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++)
		{
			const int ni = mp_nodeGraph->knn(i, tIdx);
			const float wi = mp_nodeGraph->weight(i, tIdx);
			if (wi < FLT_EPSILON || ni < 0) continue;
			int coli = 6 * ni;
			int row = 3 * sIdx;
			A.block<3, 6>(row, coli) += -dvs_dse3.block<3, 6>(row, coli)*tW; 
			A.block<1, 6>(row + 1, coli) += dvs_dse3.block<1, 6>(row + 1, coli)*tW;
			//triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 1, -tv.z()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 2, tv.y()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 3, -1 * tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 0, -tv.z()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 2, tv.x()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 4, 1 * tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 0, -tv.y()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 1, tv.x()* tW));
			//triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 5, -1 * tW));
		}

		// note that, artist model is symmetric to y axis
		Eigen::Vector3f r = Eigen::Vector3f::Zero();
		r(0) = iv(0) - tv(0);
		r(1) = iv(1) + tv(1);
		r(2) = iv(2) - tv(2);
		b.segment<3>(3 * sIdx) = r * tW;
	}

	ATA = A.transpose() * A; 
	ATb = -A.transpose() * b;

	float r = b.norm();
	std::cout << "total sym r: " << r << std::endl;
}

void ShapeSolver::CalcSmthTerm(
	Eigen::MatrixXf& ATA,
	Eigen::VectorXf& ATb
)
{
	std::vector<Eigen::Triplet<float>> triplets;
	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(3 * mp_nodeGraph->nodeNet.size(), 6 * mp_nodeGraph->nodeIdx.size()); 
	Eigen::VectorXf b = Eigen::VectorXf::Zero(3 * mp_nodeGraph->nodeNet.size());
	for (int ni = 0; ni < mp_nodeGraph->nodeIdx.size(); ni++) {
		const auto Ti = m_warpField.block<3, 4>(0, 4 * ni);
		for (int j = 0; j < mp_nodeGraph->nodeNet.rows(); j++) {
			const int nj = mp_nodeGraph->nodeNet(j, ni);
			if (nj == -1)
				continue;
			// sv: node position 
			const auto sv = m_verticesShaped.col(mp_nodeGraph->nodeIdx[nj]);
			const auto Tj = m_warpField.block<3, 4>(0, 4 * nj);
			const Eigen::Vector3f r = Ti * sv.homogeneous();
			const Eigen::Vector3f s = Tj * sv.homogeneous();
			const int row = 3 * (ni * int(mp_nodeGraph->nodeNet.rows()) + j);
			const int coli = 6 * ni;
			const int colj = 6 * nj;



			// 1st row
			triplets.emplace_back(Eigen::Triplet<float>(row, coli + 1, r.z()));
			triplets.emplace_back(Eigen::Triplet<float>(row, coli + 2, -r.y()));
			triplets.emplace_back(Eigen::Triplet<float>(row, coli + 3, 1));
			triplets.emplace_back(Eigen::Triplet<float>(row, colj + 1, -s.z()));
			triplets.emplace_back(Eigen::Triplet<float>(row, colj + 2, s.y()));
			triplets.emplace_back(Eigen::Triplet<float>(row, colj + 3, -1));

			// 2nd row
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, coli + 0, -r.z()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, coli + 2, r.x()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, coli + 4, 1));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, colj + 0, s.z()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, colj + 2, -s.x()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 1, colj + 4, -1));

			// 3rd row
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, coli + 0, r.y()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, coli + 1, -r.x()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, coli + 5, 1));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, colj + 0, -s.y()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, colj + 1, s.x()));
			triplets.emplace_back(Eigen::Triplet<float>(row + 2, colj + 5, -1));

			// bs
			b.segment<3>(row) = s - r;
		}
	}

	//Eigen::SparseMatrix<float> A(3 * mp_nodeGraph->nodeNet.size(), 6 * mp_nodeGraph->nodeIdx.size());
	//A.setFromTriplets(triplets.begin(), triplets.end());
	const auto AT = A.transpose();
	ATA =  AT * A;
	ATb =  AT * b;

	std::cout << "smth: " << b.norm() << std::endl; 
}

void ShapeSolver::CalcPointTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	int nodeNum = mp_nodeGraph->nodeIdx.size(); 
	ATA.resize(nodeNum * 6, nodeNum * 6); 
	ATA.setZero();
	ATb.resize(nodeNum * 6); 

	float total_r = 0;
	for (int i = 0; i < m_optimPairs.size(); i++)
	{
		if (m_optimPairs[i].type == 1)
		{
			int target = m_optimPairs[i].target;
			int vid = m_optimPairs[i].index;
			for (int k = 0; k < m_source.view_ids.size(); k++)
			{
				int camid = m_source.view_ids[k];
				if (!in_list(camid, usedviews)) continue;
				Camera cam = m_cameras[camid];
				const DetInstance& det = m_source.dets[k];
				Eigen::MatrixXf J = Eigen::MatrixXf::Zero(2, 6 * nodeNum); 
				Eigen::Vector3f x_local = cam.K * (cam.R* m_verticesFinal.col(vid) + cam.T);
				Eigen::MatrixXf D = Eigen::MatrixXf::Zero(2, 3);
				D(0, 0) = 1 / x_local(2);
				D(1, 1) = 1 / x_local(2);
				D(0, 2) = -x_local(0) / (x_local(2) * x_local(2));
				D(1, 2) = -x_local(1) / (x_local(2) * x_local(2));
				J = D * cam.K * cam.R * dv_dse3.middleRows<3>(3 * vid);
				Eigen::Vector2f u;
				u(0) = x_local(0) / x_local(2); 
				u(1) = x_local(1) / x_local(2); 
				Eigen::Vector2f r = u - det.keypoints[target].segment<2>(0);
				//std::cout << u.transpose() << ", " << det.keypoints[target].segment<2>(0).transpose() << std::endl; 
				//std::cout << "r: " << r << std::endl; 
				//std::cout << "K: " << cam.K << std::endl;
				ATA += J.transpose()* J; 
				ATb += -J.transpose() * r; 
				total_r += r.norm(); 
			}
		}
	}
	std::cout << "point r: " << total_r << std::endl; 
}

void ShapeSolver::CalcPointTerm3D(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	int nodeNum = mp_nodeGraph->nodeIdx.size();
	ATA.resize(nodeNum * 6, nodeNum * 6);
	ATA.setZero();
	ATb.resize(nodeNum * 6);
	CalcZ(); 
	float total_r = 0;
	for (int i = 0; i < m_optimPairs.size(); i++)
	{
		if (m_optimPairs[i].type == 1)
		{
			int target = m_optimPairs[i].target;
			int vid = m_optimPairs[i].index;
			
			Eigen::Vector3f targetpoint = Z.col(target); // skel 
			Eigen::MatrixXf J = dv_dse3.middleRows<3>(3 * vid);
			ATA += J.transpose() * J; 
			Eigen::Vector3f diff = (m_verticesFinal.col(vid) - targetpoint);
			
			ATb += -J.transpose() * diff;
			total_r += diff.transpose() * diff; 
		}
	}
	total_r = sqrtf(total_r); 
	std::cout << "point r: " << total_r << std::endl;
}

void ShapeSolver::CalcLaplacianTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	std::cout << "L3.cols: " << L3.cols() << " dvs_dse3.rows: " << dvs_dse3.rows() << std::endl; 

	Eigen::MatrixXf A = L3 * dvs_dse3; 
	Eigen::MatrixXf Delta = m_verticesDeformed * L;
	Eigen::MatrixXf diff = Delta - Delta0;
	Eigen::VectorXf b = Eigen::Map<Eigen::VectorXf>( diff.data(), 3*m_vertexNum);

	ATA = A.transpose() * A;
	ATb = -A.transpose() * b; 

	std::cout << "lap: " << b.norm() << std::endl; 
}


void ShapeSolver::setTargetModel(std::shared_ptr<MeshEigen> _targetModel)
{
	m_tarModel = _targetModel;
	m_tarTree = std::make_shared<KDTree<float>>(m_tarModel->vertices);
	m_corr = Eigen::VectorXi::Constant(m_vertexNum, -1);
	m_wDeform = Eigen::VectorXf::Ones(m_vertexNum);

	m_srcModel = std::make_shared<MeshEigen>();
	m_srcModel->vertices = m_verticesFinal;
	m_srcModel->faces = m_facesVert;
	m_srcModel->CalcNormal();
}

void ShapeSolver::CalcEarTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	int nodeNum = mp_nodeGraph->nodeIdx.size();
	ATA.resize(nodeNum * 6, nodeNum * 6);
	ATA.setZero();
	ATb.resize(nodeNum * 6);
	float total_r = 0;
	for (int i = 0; i < m_vertexNum; i++)
	{
		if (m_bodyParts[i] == L_EAR || m_bodyParts[i] == R_EAR || m_bodyParts[i] == NECK)
		{
			Eigen::MatrixXf J = dvs_dse3.middleRows<3>(3 * i);
			ATA += J.transpose() * J;
			Eigen::Vector3f diff = (m_verticesDeformed.col(i) - m_verticesShaped.col(i));

			ATb += -J.transpose() * diff;
			total_r += diff.transpose() * diff;
		}
	}
	total_r = sqrtf(total_r);
	std::cout << "ear r: " << total_r << std::endl;
}

void ShapeSolver::findCorr()
{
	m_iterModel.faces = m_facesVert;
	m_iterModel.vertices = m_verticesFinal;
	m_iterModel.CalcNormal();
	const float cosine = std::cosf(m_maxAngle);
	m_corr.setConstant(-1);
	for (int sIdx = 0; sIdx < m_iterModel.vertices.cols(); sIdx++) 
	{
		if (m_wDeform[sIdx] < FLT_EPSILON)
			continue;
		if (m_bodyParts[sIdx] == TAIL) continue; 
		if (m_bodyParts[sIdx] == L_EAR) continue; 
		if (m_bodyParts[sIdx] == R_EAR) continue; 
		const Eigen::Vector3f v = m_iterModel.vertices.col(sIdx);
		const std::vector<std::pair<float, size_t>> nbors = m_tarTree->KNNSearch(v, 256);
		for (const auto& nbor : nbors) {
			if (nbor.first < m_maxDist) {
				if (m_iterModel.normals.col(sIdx).dot(m_tarModel->normals.col(nbor.second)) > cosine) {
					m_corr[sIdx] = int(nbor.second);
					break;
				}
			}
		}
	}

	std::cout << "corr:" << (m_corr.array() >= 0).count() << std::endl;
}

void ShapeSolver::updateWarpField()
{
#pragma omp parallel for
	for (int ni = 0; ni < m_deltaTwist.cols(); ni++)
		m_warpField.middleCols(4 * ni, 4) = Twist(m_deltaTwist.col(ni))* m_warpField.middleCols(4 * ni, 4);
}

void ShapeSolver::solveNonrigidDeform(int maxIterTime, float updateThresh)
{
	m_deltaTwist.resize(6, mp_nodeGraph->nodeIdx.size());

	Eigen::MatrixXf ATA, ATAs, ATAsym;
	Eigen::VectorXf ATb, ATbs, ATbsym;
	Eigen::MatrixXf ATA_sil; 
	Eigen::VectorXf ATb_sil; 
	Eigen::MatrixXf ATA_ear; 
	Eigen::VectorXf ATb_ear;
	int paramNum = 6 * mp_nodeGraph->nodeIdx.size(); 

	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);

	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		std::cout << "iterTime: " << iterTime << std::endl; 
		for (int obid = 0; obid < obs.size(); obid++)
		{
			SetPose(obs[obid].pose);
			SetScale(obs[obid].scale);
			SetTranslation(obs[obid].translation);
			UpdateVertices();
			//m_iterModel.vertices = m_verticesFinal;
			//m_iterModel.CalcNormal();
			//findCorr();

			CalcDvDSe3();
			m_rois = obs[obid].rois; 
			m_source = obs[obid].source; 
			usedviews = obs[obid].usedViews;

			CalcDeformTerm_sil(ATA_sil, ATb_sil);

			ATA += ATA_sil * w_sil;
			ATb += ATb_sil * w_sil;
		}
		CalcSymTerm(ATAsym, ATbsym);
		ATA += ATAsym * m_wSym;
		ATb += ATbsym * m_wSym;

		CalcSmthTerm(ATAs, ATbs);
		ATA += ATAs * m_wSmth;
		ATb += ATbs * m_wSmth;

		//Eigen::MatrixXf ATA_lap; 
		//Eigen::VectorXf ATb_lap;
		//CalcLaplacianTerm(ATA_lap, ATb_lap); 
		//ATA += ATA_lap * w_lap; 
		//ATb += ATb_lap * w_lap; 


		Eigen::MatrixXf ATAr(paramNum, paramNum);
		ATAr.setIdentity();

		ATA += ATAr * m_wRegular;

		Eigen::Map<Eigen::VectorXf>(m_deltaTwist.data(), m_deltaTwist.size()) = ATA.ldlt().solve(ATb);

		if (m_deltaTwist.norm() < updateThresh)
			break;

		// debug 
		std::cout << "delta twist: " << m_deltaTwist.norm() << std::endl;
		updateWarpField();
		//updateIterModel();
	}
}

void ShapeSolver::updateIterModel()
{
	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++) {
		Eigen::Matrix4f T = Eigen::Matrix4f::Zero();

		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++) {
			const int ni = mp_nodeGraph->knn(i, sIdx);
			if (ni != -1)
				T += mp_nodeGraph->weight(i, sIdx) * m_warpField.middleCols(4 * ni, 4);
		}

		m_iterModel.vertices.col(sIdx) = T.topLeftCorner(3, 4) * m_srcModel->vertices.col(sIdx).homogeneous();
		m_iterModel.normals.col(sIdx) = T.topLeftCorner(3, 3) * m_srcModel->normals.col(sIdx);
	}
}

void ShapeSolver::solveNonrigidDeformVHull(int maxIterTime, float updateThresh)
{
	m_deltaTwist.resize(6, mp_nodeGraph->nodeIdx.size());

	Eigen::MatrixXf ATA, ATAs, ATAsym;
	Eigen::VectorXf ATb, ATbs, ATbsym;
	Eigen::MatrixXf ATA_sil;
	Eigen::VectorXf ATb_sil;
	Eigen::MatrixXf ATA_ear;
	Eigen::VectorXf ATb_ear;
	int paramNum = 6 * mp_nodeGraph->nodeIdx.size();

	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);

	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		std::cout << "iterTime: " << iterTime << std::endl;

		CalcDvDSe3();

		CalcDeformTerm(ATA_sil, ATb_sil); 

		ATA += ATA_sil; 
		ATb += ATb_sil; 

		CalcSymTerm(ATAsym, ATbsym);
		ATA += ATAsym * m_wSym;
		ATb += ATbsym * m_wSym;

		CalcSmthTerm(ATAs, ATbs);
		ATA += ATAs * m_wSmth;
		ATb += ATbs * m_wSmth;

		//Eigen::MatrixXf ATA_lap; 
		//Eigen::VectorXf ATb_lap;
		//CalcLaplacianTerm(ATA_lap, ATb_lap); 
		//ATA += ATA_lap * w_lap; 
		//ATb += ATb_lap * w_lap; 


		Eigen::MatrixXf ATAr(paramNum, paramNum);
		ATAr.setIdentity();

		ATA += ATAr * m_wRegular;

		Eigen::Map<Eigen::VectorXf>(m_deltaTwist.data(), m_deltaTwist.size()) = ATA.ldlt().solve(ATb);

		if (m_deltaTwist.norm() < updateThresh)
			break;

		// debug 
		std::cout << "delta twist: " << m_deltaTwist.norm() << std::endl;
		updateWarpField();
		updateIterModel();
	}
}


void ShapeSolver::totalSolveProcedure()
{
	initLaplacian(); 
	for (int i = 0; i < 30; i++)
	{
		// step1: 
		std::cout << "step: " << i << " ************************ " << std::endl;
		solveNonrigidDeform(1, 1e-5);

		std::stringstream ss;
		ss << "H:/pig_results_shape/shapeiter/shape_" << i << ".obj";
		SaveObj(ss.str());
	}

	//for (int i = 0; i < 5; i++)
	//{
	//	std::cout << "Second phase: " << i << std::endl;
	//	// step1: 
	//	m_wSmth = 1.0;
	//	m_wRegular = 0.1;
	//	m_maxDist = 0.2;
	//	m_wSym = 1;
	//	w_point = 10;
	//	solveNonrigidDeform(1, 1e-5);
	//	std::stringstream ss;
	//	ss << "H:/pig_results_shape/shapeiter/shape_2_" << i << ".obj";
	//	SaveObj(ss.str());

	//}
}


void ShapeSolver::computeVolume()
{
	CalcZ();
	m_V.center = Z.col(20).cast<float>();
	m_V.computeVolumeFromRoi(m_rois);
	m_V.getSurface();
	std::stringstream ss;
	ss << "H:/pig_results_shape/tmp.xyz";
	m_V.saveXYZFileWithNormal(ss.str());
	std::stringstream cmd;
	cmd << "D:/Projects/animal_calib/PoissonRecon.x64.exe --in " << ss.str() << " --out " << ss.str() << ".ply";
	const std::string cmd_str = cmd.str();
	const char* cmd_cstr = cmd_str.c_str();
	system(cmd_cstr);
}

//// solve pose and shape 
//// to fit visualhull model
//void ShapeSolver::solvePoseAndShape(int maxIterTime)
//{
//	float terminal = 0.01;
//
//	int M = m_poseToOptimize.size();
//	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
//	{
//		UpdateVertices();
//		m_iterModel.vertices = m_verticesFinal;
//		m_iterModel.CalcNormal();
//		findCorr();
//		Eigen::MatrixXf ATA;
//		Eigen::VectorXf ATb;
//		CalcPoseTerm(ATA, ATb);
//
//		Eigen::VectorXf theta(3 + 3 * M);
//		theta.segment<3>(0) = m_translation;
//		for (int i = 0; i < M; i++)
//		{
//			int jIdx = m_poseToOptimize[i];
//			theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
//		}
//
//		Eigen::MatrixXf poseJ3d;
//		CalcSkelJacobiPartThetaByPairs(poseJ3d);
//		Eigen::MatrixXf skel = getRegressedSkelbyPairs();
//		Eigen::MatrixXf Hjoint = Eigen::MatrixXf::Zero(3 + 3 * M, 3 + 3 * M); // data term 
//		Eigen::VectorXf bjoint = Eigen::VectorXf::Zero(3 + 3 * M);
//		for (int k = 0; k < m_source.view_ids.size(); k++)
//		{
//			Eigen::MatrixXf H_view;
//			Eigen::VectorXf b_view;
//			CalcPose2DTermByPairs(k, skel, poseJ3d, H_view, b_view);
//			Hjoint += H_view;
//			bjoint += b_view;
//		}
//
//		// solve
//		double lambda = 0.0005;
//		double w1 = 1;
//		double w_reg = 0.001;
//		Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(3 + 3 * M, 3 + 3 * M);
//		Eigen::MatrixXd H_reg = DTD;  // reg term 
//		Eigen::VectorXd b_reg = -theta; // reg term 
//
//		Eigen::MatrixXd H =
//			Hjoint + ATA * w1 + H_reg * w_reg + DTD * lambda;
//		Eigen::VectorXd b =
//			bjoint + ATb * w1 + b_reg * w_reg;
//
//		Eigen::VectorXd delta = H.ldlt().solve(b);
//
//		// update 
//		m_translation += delta.segment<3>(0);
//		for (int i = 0; i < M; i++)
//		{
//			int jIdx = m_poseToOptimize[i];
//			m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
//		}
//		if (delta.norm() < terminal) break;
//	}
//
//	//
//	//for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
//	//{
//	//	UpdateVertices();
//	//	UpdateVertices();
//	//	m_iterModel.vertices = m_verticesFinal;
//	//	m_iterModel.CalcNormal();
//	//	findCorr();
//	//	Eigen::MatrixXd ATA;
//	//	Eigen::VectorXd ATb;
//	//	CalcShapeTerm(ATA, ATb);
//
//	//	Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(m_shapeNum, m_shapeNum);  // Leveberg Marquart
//	//	Eigen::MatrixXd H_reg = DTD;
//	//	Eigen::VectorXd b_reg = -m_shapeParam;
//	//	double lambda = 0.001;
//	//	double w1 = 1;
//	//	double w_reg = 0.01;
//	//	Eigen::MatrixXd H = ATA * w1 + H_reg * w_reg + DTD * lambda;
//	//	Eigen::VectorXd b = ATb * w1 + b_reg * w_reg;
//
//	//	Eigen::VectorXd delta = H.ldlt().solve(b);
//	//	m_shapeParam = m_shapeParam + delta; 
//
//	//	if (delta.norm() < terminal) break;
//	//}
//}

void ShapeSolver::initLaplacian()
{
	D.resize(m_vertexNum); 
	Delta0.resize(3, m_vertexNum); 
	A.resize(m_vertexNum, m_vertexNum); 
	L.resize(m_vertexNum, m_vertexNum); 
	D.setZero();
	Delta0.setZero(); 
	A.setZero(); 
	L.setZero();

	std::vector<std::vector<int> > neighbours;
	neighbours.resize(m_vertexNum); 
	for (int i = 0; i < m_faceNum; i++)
	{
		Eigen::Vector3u f = m_facesVert.col(i);
		for (int k = 0; k < 3; k++)
		{
			int a = f(k);
			int b = f((k + 1) % 3);
			if (!in_list(b, neighbours[a])) neighbours[a].push_back(b); 
			if (!in_list(a, neighbours[b])) neighbours[b].push_back(a); 
		}
	}
	for (int i = 0; i < m_vertexNum; i++)
	{
		D(i) = neighbours[i].size(); 
	}
	
	for (int i = 0; i < m_vertexNum; i++)
	{
		for (int j = 0; j < neighbours[i].size(); j++)
		{
			int k = neighbours[i][j];
			A(i, k) = 1; 
		}
	}
	Eigen::MatrixXf B = A; 
	for (int i = 0; i < m_vertexNum; i++)
	{
		B.row(i) = A.row(i) / D(i); 
	}
	L = Eigen::MatrixXf::Identity(m_vertexNum, m_vertexNum) - B; 
	Delta0 = m_verticesShaped * L; 

	L3.resize(3 * m_vertexNum, 3 * m_vertexNum);
	L3.setZero(); 
	for (int i = 0; i < m_vertexNum; i++)
	{
		for (int j = 0; j < neighbours[i].size(); j++)
		{
			int k = neighbours[i][j];
			L3.block<3, 3>(3 * i, 3 * k) = Eigen::Matrix3f::Identity() / D(i); 
		}
	}
	L3 = -L3; 
	for (int i = 0; i < 3 * m_vertexNum; i++)L3(i, i) = 1; 
}