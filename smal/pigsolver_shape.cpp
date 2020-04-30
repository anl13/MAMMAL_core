#include "pigsolver.h"
#include "../utils/image_utils.h"

void PigSolver::feedData(
	const ROIdescripter& _roi,
	const BodyState& _body
)
{
	m_rois.push_back(_roi); 
	m_bodies.push_back(_body);
}

void PigSolver::feedRender(
	const cv::Mat& _render
)
{
	m_renders.push_back(_render);
}

#if 0
void PigSolver::iterateStep(int iter)
{
	double D = 0.001; // 1 mm per iter 
	// prepare chamfer 
	vector<cv::Mat> currentChamfers; 
	for (int i = 0; i < m_renders.size(); i++)
	{
		cv::Mat chamfer = get_dist_trans(m_renders[i]);
		currentChamfers.push_back(chamfer); 
	}

	cv::Mat packChamfer;
	packImgBlock(currentChamfers, packChamfer); 
	packChamfer = reverseChamfer(packChamfer);
	cv::Mat visPackChamfer = vis_float_image(packChamfer); 
	std::stringstream ss;
	ss << "E:/debug_pig2/chamfer/source_" << iter << ".png";
	cv::imwrite(ss.str(), visPackChamfer); 

	// currently, assu me only one frame used 
	//m_poseParam = m_bodies[0].pose;
	//m_translation = m_bodies[0].trans;
	//UpdateVertices();
	// compute gradient
	Eigen::VectorXd gradient; 
	gradient.resize(m_vertexNum); 
	gradient.setZero(); 

	cv::Mat packTargetChamfer; 
	std::vector<cv::Mat> targets; 
	for (int oid = 0; oid < m_rois.size(); oid++)
	{
		ROIdescripter& roi = m_rois[oid];
		cv::Mat&       chamfer = currentChamfers[oid];
		targets.push_back(roi.chamfer);
		for (int i = 0; i < m_vertexNum; i++)
		{
			if (m_bodyParts[i] != MAIN_BODY) {
				continue; // only optimize main body
			}
			double g = 0; 
			Eigen::Vector3d p = m_verticesFinal.col(i);
			double f = queryPixel(chamfer, p, roi.cam);
			double t = roi.queryChamfer(p); 
			if (t < 1)
			{
				g = D; 
			}
			else
			{
				//if (f < 5 && f < t)
				//{
				//	g = -D/2; 
				//}
			}
			gradient(i) += g; 
		}
	}
	packImgBlock(targets, packTargetChamfer); 
	packTargetChamfer = reverseChamfer(packTargetChamfer); 
	std::stringstream ss1; 
	ss1 << "E:/debug_pig/chamfer/target_" << iter << ".png";
	cv::Mat visTarget = vis_float_image(packTargetChamfer); 
	cv::imwrite(ss1.str(), visTarget); 

	m_deform -= gradient; 

	double r = gradient.norm(); 
	double r1 = m_deform.norm();

	std::cout << "r: " << r << std::endl; 
	std::cout << "d: " << r1 << std::endl;
}

void PigSolver::NaiveNodeDeformStep(int iter)
{
	double D = 0.001; // 1 mm per iter 
					  // prepare chamfer 
	vector<cv::Mat> currentChamfers;
	for (int i = 0; i < m_renders.size(); i++)
	{
		cv::Mat chamfer = get_dist_trans(m_renders[i]);
		currentChamfers.push_back(chamfer);
	}
	// compute gradient
	UpdateNormals(); 
	std::vector<cv::Mat> targets;
	for (int oid = 0; oid < m_rois.size(); oid++)
	{
		ROIdescripter& roi = m_rois[oid];
		cv::Mat&       chamfer = currentChamfers[oid];
		targets.push_back(roi.chamfer);
		for (int i = 0; i < mp_nodeGraph->nodeIdx.size(); i++)
		{
			int vid = mp_nodeGraph->nodeIdx[i];
			if (m_bodyParts[vid] != MAIN_BODY) {
				continue; // only optimize main body
			}
			double g = 0;
			Eigen::Vector3d p = m_verticesFinal.col(vid);
			Eigen::Vector3d n = m_normalShaped.col(vid);
			double f = queryPixel(chamfer, p, roi.cam);
			double t = roi.queryChamfer(p);
			if (t < 1)
			{
				m_warpField.block<3, 1>(0, 4 * i + 3) -= n*D;
			}
			else
			{
				//if (f < 5 && f < t)
				//{
				//	g = -D/2; 
				//}
			}
		}
	}
}

#endif 

void PigSolver::clearData()
{
	m_bodies.clear(); 
	m_renders.clear(); 
}

void PigSolver::optimizeShapeToBoneLength(int maxIter, double terminal)
{
	Eigen::MatrixXd JSkel;
	CalcShapeJacobiToSkel(JSkel);
	std::vector<std::pair<int, int> > rigid_bones =
	{
		{ 0,1 },{ 0,2 },{ 1,2 },{ 1,3 },{ 2,4 },{ 3,4 },
	{ 5,6 },{ 5,7 },{ 7,9 },{ 6,8 },{ 8,10 },{ 20,21 },
	{ 21,11 },{ 21,12 },{ 11,12 },{ 11,13 },{ 13,15 },
	{ 12,14 },{ 14,16 }
	};
	Eigen::MatrixXd ATA(m_shapeNum, m_shapeNum);
	Eigen::VectorXd ATb(m_shapeNum);
	CalcZ(); 

	for (int iter = 0; iter < maxIter; iter++) 
	{
		Eigen::MatrixXd tpose = getRegressedSkelTPose(); 
		ATA.setZero();
		ATb.setZero();
		for (int i = 1; i < rigid_bones.size(); i++) 
		{
			int a = rigid_bones[i].first;
			int b = rigid_bones[i].second;
			if (Z.col(a).norm() < 0.00001 || Z.col(b).norm() < 0.00001) continue;
			if (tpose.col(a).norm() < 0.00001 || tpose.col(b).norm() < 0.00001)continue;
			Eigen::Vector3d unit_vec = tpose.col(a) - tpose.col(b); 
			unit_vec.normalize();
			double length = (Z.col(a) - Z.col(b)).norm();
			Eigen::Vector3d source = unit_vec * length; 
			Eigen::Vector3d target = Z.col(a) - Z.col(b);
			if (length > 0.5) continue; 
			double ww = length;
			Eigen::Vector3d residual = - source + target;
			Eigen::MatrixXd jacobi = JSkel.middleRows(3 * a, 3) - JSkel.middleRows(3*b,3);
			ATA += ww * jacobi.transpose() * jacobi;
			ATb += ww * jacobi.transpose() * residual;
		}

		// prior
		ATA += 0.001 * Eigen::MatrixXd::Identity(m_shapeNum,m_shapeNum);
		ATb -= 0.001 * m_shapeParam;

		Eigen::VectorXd delta = ATA.ldlt().solve(ATb);
		m_shapeParam += delta;

		if (delta.norm() < terminal)
			break;
	}
}
#if 0
void PigSolver::naiveNodeDeform()
{
	GLFWwindow* windowPtr = mp_renderer->s_windowPtr;
	InitNodeAndWarpField();

	int iter = 0;
	for (; iter < 10; iter++)
	{
		UpdateVertices();
		mp_renderer->colorObjs.clear();
		mp_renderer->texObjs.clear();

		const auto& faces = m_facesVert;
		const Eigen::MatrixXf vs = m_verticesFinal.cast<float>();

		RenderObjectColor* pig_render = new RenderObjectColor();
		pig_render->SetFaces(faces);
		pig_render->SetVertices(vs);
		Eigen::Vector3f color(1.0, 1.0, 1.0);
		pig_render->SetColor(color);
		mp_renderer->colorObjs.push_back(pig_render);

		const auto& cameras = m_cameras;
		std::vector<cv::Mat> renders;
		for (int view = 0; view < m_source.view_ids.size(); view++)
		{
			int camid = m_source.view_ids[view];
			Eigen::Matrix3f R = cameras[camid].R.cast<float>();
			Eigen::Vector3f T = cameras[camid].T.cast<float>();
			mp_renderer->s_camViewer.SetExtrinsic(R, T);
			mp_renderer->Draw();
			cv::Mat capture = mp_renderer->GetImage();
			feedRender(capture);
			renders.push_back(capture);
		}

		// debug
		cv::Mat pack_render;
		packImgBlock(renders, pack_render);
		std::stringstream ss;
		ss << "E:/debug_pig2/shapeiter/" << std::setw(6)
			<< iter << ".jpg";
		cv::imwrite(ss.str(), pack_render);

		std::cout << RED_TEXT("iter:") << iter << std::endl;

	    iterateStep(iter);

		//NaiveNodeDeformStep(iter);
		clearData();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	}
}
#endif 
void PigSolver::CalcPoseTerm(Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb)
{
	Eigen::MatrixXd J;
	Eigen::MatrixXd _Jjoint;
	CalcPoseJacobiPartTheta(_Jjoint, J);
	int M = J.cols();
	//Eigen::VectorXd b = Eigen::VectorXd::Zero(m_vertexNum);
	Eigen::VectorXd b = Eigen::VectorXd::Zero(m_vertexNum * 3);
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m_vertexNum * 3, M);
	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++)
	{
		const int tIdx = m_corr[sIdx];
		const double wd = m_wDeform[sIdx];
		if (tIdx == -1 || wd < DBL_EPSILON) continue; 
		const auto tn = m_tarModel->normals.col(tIdx);
		const auto iv = m_verticesFinal.col(sIdx);
		const auto tv = m_tarModel->vertices.col(tIdx);
		//A.middleRows(sIdx, 1) =
		//	tn.transpose() * J.middleRows(3 * sIdx, 3);
		//b(sIdx) = wd * (tn.dot(tv - iv));
		A.middleRows(sIdx * 3, 3) = J.middleRows(3 * sIdx, 3);
		b.segment<3>(sIdx * 3) = tv - iv; 
	}
	Eigen::MatrixXd AT = A.transpose(); 
	ATA = AT * A;
	ATb = AT * b;
}

void PigSolver::CalcShapeTerm(Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb)
{
	Eigen::MatrixXd jointJ, J;
	CalcShapeJacobi(jointJ, J);
	int M = J.cols();
	//Eigen::VectorXd b = Eigen::VectorXd::Zero(m_vertexNum);
	Eigen::VectorXd b = Eigen::VectorXd::Zero(m_vertexNum * 3);
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m_vertexNum * 3, M);
	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++)
	{
		const int tIdx = m_corr[sIdx];
		const double wd = m_wDeform[sIdx];
		if (tIdx == -1 || wd < DBL_EPSILON) continue;
		const auto tn = m_tarModel->normals.col(tIdx);
		const auto iv = m_verticesFinal.col(sIdx);
		const auto tv = m_tarModel->vertices.col(tIdx);
		//A.middleRows(sIdx, 1) =
		//	tn.transpose() * J.middleRows(3 * sIdx, 3);
		//b(sIdx) = wd * (tn.dot(tv - iv));
		A.middleRows(sIdx * 3, 3) = J.middleRows(3 * sIdx, 3);
		b.segment<3>(sIdx * 3) = tv - iv;
	}
	Eigen::MatrixXd AT = A.transpose();
	ATA = AT * A;
	ATb = AT * b;
}

void PigSolver::CalcDeformTerm(
	Eigen::SparseMatrix<double>& ATA,
	Eigen::VectorXd& ATb
)
{
	if (m_wDeform.size() != m_srcModel->vertices.cols())
		m_wDeform = Eigen::VectorXd::Ones(m_srcModel->vertices.cols());

	std::vector<Eigen::Triplet<double>> triplets;
	Eigen::VectorXd b = Eigen::VectorXd::Zero(m_srcModel->vertices.cols());
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> globalAffineNormalized = m_globalAffine;
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		globalAffineNormalized.block<3, 1>(0, jointId * 4 + 3) -= (m_globalAffine.block<3, 3>(0, jointId * 4)*m_jointsShaped.col(jointId));
	}
	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++) {
		const int tIdx = m_corr[sIdx];
		const double wd = m_wDeform[sIdx];
		if (tIdx == -1 || wd < DBL_EPSILON)
			continue;

		const auto tn = m_tarModel->normals.col(tIdx);
		const auto iv = m_verticesFinal.col(sIdx);
		const auto tv = m_tarModel->vertices.col(tIdx);
		Eigen::Matrix<double, 4, 4, Eigen::ColMajor> globalAffineAverage;
		Eigen::Map<Eigen::VectorXd>(globalAffineAverage.data(), 16)
			= Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>
			(globalAffineNormalized.data(), 16, m_jointNum) * (m_lbsweights.col(sIdx));

		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++) 
		{
			const int ni = mp_nodeGraph->knn(i, sIdx);
			const double w = mp_nodeGraph->weight(i, sIdx) * wd;
			if (w < DBL_EPSILON || ni == -1)
				continue;
			
			const int col = 6 * ni;
			Eigen::MatrixXd dv0 = Eigen::MatrixXd::Zero(3, 6);
			dv0(0, 1) = iv.z();
			dv0(0, 2) = -iv.y();
			dv0(1, 0) = -iv.z();
			dv0(1, 2) = iv.x();
			dv0(2, 0) = iv.y();
			dv0(2, 1) = -iv.x();
			dv0.middleCols(3, 3) = Eigen::Matrix3d::Identity();
			dv0 = globalAffineAverage.block<3, 3>(0, 0) * dv0;

			triplets.emplace_back(Eigen::Triplet<double>(sIdx, col, w * (tn.x()*dv0(0, 0) + tn.y()*dv0(1, 0) + tn.z()*dv0(2, 0) ))); // alpha_ni
			triplets.emplace_back(Eigen::Triplet<double>(sIdx, col + 1, w * (tn.x()*dv0(0,1) +tn.y()*dv0(1,1) + tn.z()*dv0(2,1) ))); // beta_ni
			triplets.emplace_back(Eigen::Triplet<double>(sIdx, col + 2, w * (tn.x()*dv0(0, 2) + tn.y()*dv0(1, 2) + tn.z()*dv0(2, 2)))); // gamma_ni
			triplets.emplace_back(Eigen::Triplet<double>(sIdx, col + 3, w * (tn.x()*dv0(0, 3) + tn.y()*dv0(1, 3) + tn.z()*dv0(2, 3)) )); // tx_ni
			triplets.emplace_back(Eigen::Triplet<double>(sIdx, col + 4, w * (tn.x()*dv0(0, 4) + tn.y()*dv0(1, 4) + tn.z()*dv0(2, 4)) )); // ty_ni
			triplets.emplace_back(Eigen::Triplet<double>(sIdx, col + 5, w * (tn.x()*dv0(0, 5) + tn.y()*dv0(1, 5) + tn.z()*dv0(2, 5)) )); // tz_ni
		}
		b[sIdx] = wd * (tn.dot(tv - iv));
	}
	Eigen::SparseMatrix<double> A(m_srcModel->vertices.cols(), mp_nodeGraph->nodeIdx.size() * 6);
	A.setFromTriplets(triplets.begin(), triplets.end());
	const auto AT = A.transpose();
	ATA = AT * A;
	ATb = AT * b;
}

void PigSolver::CalcSymTerm(
	Eigen::SparseMatrix<double>& ATA,
	Eigen::VectorXd& ATb
)
{
	std::vector<Eigen::Triplet<double>> triplets;
	Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * m_vertexNum);
	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++) 
	{
		const int tIdx = m_symIdx[sIdx];
		if (tIdx == sIdx)
		{
			// do something 
			continue;
		}
		// else 
		const auto iv = m_verticesDeformed.col(sIdx);
		const auto tv = m_verticesDeformed.col(tIdx);

		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++)
		{
			const int ni = mp_nodeGraph->knn(i, sIdx);
			const double wi = mp_nodeGraph->weight(i, sIdx);
			if (wi < DBL_EPSILON || ni == -1)
				continue;
			
			int coli = 6 * ni;
			int row = 3 * sIdx;
			triplets.emplace_back(Eigen::Triplet<double>(row, coli + 1, iv.z()));
			triplets.emplace_back(Eigen::Triplet<double>(row, coli + 2, -iv.y()));
			triplets.emplace_back(Eigen::Triplet<double>(row, coli + 3, 1));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, coli + 0, -iv.z()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, coli + 2, iv.x()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, coli + 4, 1));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, coli + 0, iv.y()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, coli + 1, -iv.x()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, coli + 5, 1));
		}

		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++)
		{
			const int ni = mp_nodeGraph->knn(i, tIdx);
			const double wi = mp_nodeGraph->weight(i, tIdx);
			if (wi < DBL_EPSILON || ni < 0) continue;
			int coli = 6 * ni;
			int row = 3 * sIdx;
			triplets.emplace_back(Eigen::Triplet<double>(row, coli + 1, -tv.z()));
			triplets.emplace_back(Eigen::Triplet<double>(row, coli + 2, tv.y()));
			triplets.emplace_back(Eigen::Triplet<double>(row, coli + 3, -1));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, coli + 0, -tv.z()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, coli + 2, tv.x()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, coli + 4, 1));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, coli + 0, -tv.y()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, coli + 1, tv.x()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, coli + 5, -1));
		}
		
		Eigen::Vector3d r;
		r(0) = iv(0) - tv(0);
		r(1) = iv(1) + tv(1);
		r(2) = iv(2) - tv(2);
		b.segment<3>(3 * sIdx) = -r; 
	}
	Eigen::SparseMatrix<double> A(m_vertexNum * 3, mp_nodeGraph->nodeIdx.size() * 6);
	A.setFromTriplets(triplets.begin(), triplets.end());
	const auto AT = A.transpose();
	ATA = AT * A;
	ATb = AT * b;
}

void PigSolver::CalcSmthTerm(
	Eigen::SparseMatrix<double>& ATA,
	Eigen::VectorXd& ATb
)
{
	std::vector<Eigen::Triplet<double>> triplets;
	Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * mp_nodeGraph->nodeNet.size());
	for (int ni = 0; ni < mp_nodeGraph->nodeIdx.size(); ni++) {
		const auto Ti = m_warpField.block<3, 4>(0, 4 * ni);
		for (int j = 0; j < mp_nodeGraph->nodeNet.rows(); j++) {
			const int nj = mp_nodeGraph->nodeNet(j, ni);
			if (nj == -1)
				continue;
			// sv: node position 
			const auto sv = m_verticesShaped.col(mp_nodeGraph->nodeIdx[nj]);
			const auto Tj = m_warpField.block<3, 4>(0, 4 * nj);
			const Eigen::Vector3d r = Ti * sv.homogeneous();
			const Eigen::Vector3d s = Tj * sv.homogeneous();
			const int row = 3 * (ni * int(mp_nodeGraph->nodeNet.rows()) + j);
			const int coli = 6 * ni;
			const int colj = 6 * nj;

			// 1st row
			triplets.emplace_back(Eigen::Triplet<double>(row, coli + 1, r.z()));
			triplets.emplace_back(Eigen::Triplet<double>(row, coli + 2, -r.y()));
			triplets.emplace_back(Eigen::Triplet<double>(row, coli + 3, 1));
			triplets.emplace_back(Eigen::Triplet<double>(row, colj + 1, -s.z()));
			triplets.emplace_back(Eigen::Triplet<double>(row, colj + 2, s.y()));
			triplets.emplace_back(Eigen::Triplet<double>(row, colj + 3, -1));

			// 2nd row
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, coli + 0, -r.z()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, coli + 2, r.x()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, coli + 4, 1));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, colj + 0, s.z()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, colj + 2, -s.x()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 1, colj + 4, -1));

			// 3rd row
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, coli + 0, r.y()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, coli + 1, -r.x()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, coli + 5, 1));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, colj + 0, -s.y()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, colj + 1, s.x()));
			triplets.emplace_back(Eigen::Triplet<double>(row + 2, colj + 5, -1));

			// bs
			b.segment<3>(row) = s - r;
		}
	}

	Eigen::SparseMatrix<double> A(3 * mp_nodeGraph->nodeNet.size(), 6 * mp_nodeGraph->nodeIdx.size());
	A.setFromTriplets(triplets.begin(), triplets.end());
	const auto AT = A.transpose();
	ATA = m_wSmth * AT * A;
	ATb = m_wSmth * AT * b;
}


void PigSolver::setTargetModel(std::shared_ptr<Model> _targetModel)
{
	m_tarModel = _targetModel;
	m_tarTree = std::make_shared<KDTree<double>>(m_tarModel->vertices);
	m_corr = Eigen::VectorXi::Constant(m_vertexNum, -1); 
	m_wDeform = Eigen::VectorXd::Ones(m_vertexNum);
	m_deltaTwist.resize(6, mp_nodeGraph->nodeIdx.size()); 
}

void PigSolver::setSourceModel()
{
	m_srcModel = std::make_shared<Model>();
	m_srcModel->faces = m_facesVert;
	m_srcModel->vertices = m_verticesFinal;
	m_srcModel->CalcNormal();
	m_iterModel = *m_srcModel;
}

void PigSolver::findCorr()
{
	const double cosine = std::cosf(m_maxAngle);
	m_corr.setConstant(-1);
#pragma omp parallel for
	for (int sIdx = 0; sIdx < m_iterModel.vertices.cols(); sIdx++) {
		if (m_wDeform[sIdx] < FLT_EPSILON)
			continue;
		const Eigen::Vector3d v = m_iterModel.vertices.col(sIdx);
		const std::vector<std::pair<double, size_t>> nbors = m_tarTree->KNNSearch(v, 4);
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

void PigSolver::updateWarpField()
{
#pragma omp parallel for
	for (int ni = 0; ni < m_deltaTwist.cols(); ni++)
		m_warpField.middleCols(4 * ni, 4) = MathUtil::Twist(m_deltaTwist.col(ni))* m_warpField.middleCols(4 * ni, 4);
}

void PigSolver::solveNonrigidDeform(int maxIterTime, double updateThresh)
{
	Eigen::SparseMatrix<double> ATA, ATAs, ATAsym;
	Eigen::VectorXd ATb, ATbs, ATbsym;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

	for (int iterTime = 0; iterTime < maxIterTime; iterTime++) 
	{
		UpdateVertices(); 
		m_iterModel.vertices = m_verticesFinal;
		m_iterModel.CalcNormal();
		findCorr();
		CalcDeformTerm(ATA, ATb);
		CalcSmthTerm(ATAs, ATbs);

		std::cout << "deform term b: " << ATb.norm() << std::endl;
		std::cout << "smooth term b: " << ATbs.norm() << std::endl;

		Eigen::SparseMatrix<double> ATAr(6 * mp_nodeGraph->nodeIdx.size(), 6 * mp_nodeGraph->nodeIdx.size());
		ATAr.setIdentity();
		ATAr *= m_wRegular;
		ATA += ATAr;
		ATA += ATAs * m_wSmth;
		ATb += ATbs * m_wSmth;
		if (m_symIdx.size() > 0)
		{
			CalcSymTerm(ATAsym, ATbsym);
			ATA += ATAsym * m_wSym;
			ATb += ATbsym * m_wSym;
			std::cout << "symm   term b: " << ATbsym.norm() << std::endl;
		}

		Eigen::Map<Eigen::VectorXd>(m_deltaTwist.data(), m_deltaTwist.size()) = solver.compute(ATA).solve(ATb);
		assert(solver.info() == 0);

		if (m_deltaTwist.norm() < updateThresh)
			break;

		// debug 
		std::cout << "delta twist: " << m_deltaTwist.norm() << std::endl;
		updateWarpField();
		updateIterModel();

		//if (debugPath != "") {
		//	static int cnt = 0;
		//	m_iterModel.Save(debugPath + "/" + std::to_string(cnt++) + ".obj");
		//}
	}
}

void PigSolver::updateIterModel()
{
	for (int sIdx = 0; sIdx < m_srcModel->vertices.cols(); sIdx++) {
		Eigen::Matrix4d T = Eigen::Matrix4d::Zero();

		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++) {
			const int ni = mp_nodeGraph->knn(i, sIdx);
			if (ni != -1)
				T += mp_nodeGraph->weight(i, sIdx) * m_warpField.middleCols(4 * ni, 4);
		}

		m_iterModel.vertices.col(sIdx) = T.topLeftCorner(3, 4) * m_srcModel->vertices.col(sIdx).homogeneous();
		m_iterModel.normals.col(sIdx) = T.topLeftCorner(3, 3) * m_srcModel->normals.col(sIdx);
	}
}

void PigSolver::totalSolveProcedure()
{
	for (int i = 0; i < 3; i++)
	{
		std::cout << "Frist phase: " << i << std::endl;
		solvePoseAndShape(1);
		// step1: 
		m_wSmth = 1.0;
		m_wRegular = 0.1;
		 
		solveNonrigidDeform(1, 1e-5);
	}

	for (int i = 0; i < 3; i++)
	{
		std::cout << "Second phase: " << i << std::endl; 
		solvePoseAndShape(1);
		// step1: 
		m_wSmth = 0.1;
		m_wRegular = 0.01;
		m_maxDist = 0.05;
		solveNonrigidDeform(1, 1e-5);
	}
}


/// volumetric method
void PigSolver::computeVolume()
{
	CalcZ();
	m_V.center = Z.col(20).cast<float>();
	m_V.computeVolumeFromRoi(m_rois);
	m_V.getSurface();
	std::stringstream ss;
	ss << "E:/debug_pig2/visualhull/tmp.xyz";
	m_V.saveXYZFileWithNormal(ss.str());
	std::stringstream cmd;
	cmd << "D:/Projects/animal_calib/PoissonRecon.x64.exe --in " << ss.str() << " --out " << ss.str() << ".ply";
	const std::string cmd_str = cmd.str();
	const char* cmd_cstr = cmd_str.c_str();
	system(cmd_cstr);
}

void PigSolver::FitPoseToVerticesSameTopo(const int maxIterTime, const double terminal)
{
	Eigen::VectorXd V_target = Eigen::Map<Eigen::VectorXd>(m_targetVSameTopo.data(), 3 * m_vertexNum);

	int M = m_poseToOptimize.size();
	int N = m_topo.joint_num;

	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();
		std::stringstream ss; 
		ss << "E:/debug_pig2/shape/pose_" << iterTime << ".obj";
		SaveObj(ss.str());

		Eigen::VectorXd r = Eigen::Map<Eigen::VectorXd>(m_verticesFinal.data(), 3 * m_vertexNum) - V_target;

		Eigen::VectorXd theta(3 + 3 * M);
		theta.segment<3>(0) = m_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
		}
		Eigen::VectorXd theta0 = theta;
		// solve
		Eigen::MatrixXd H_view;
		Eigen::VectorXd b_view;
		Eigen::MatrixXd J; 
		Eigen::MatrixXd J_joint; 
		CalcPoseJacobiPartTheta(J_joint, J);
		Eigen::MatrixXd H1 = J.transpose() * J;
		Eigen::MatrixXd b1 = -J.transpose() * r; 
			
		double lambda = 0.001;
		double w1 = 1;
		double w_reg = 0.01;
		Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXd H_reg = DTD;  // reg term 
		Eigen::VectorXd b_reg = -theta; // reg term 

		Eigen::MatrixXd H = H1 * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXd b = b1 * w1 + b_reg * w_reg;

		Eigen::VectorXd delta = H.ldlt().solve(b);

		// update 
		m_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
		}
		// if(iterTime == 1) break; 
		if (delta.norm() < terminal) break;
	}
}

// solve pose and shape 
// to fit visualhull model
void PigSolver::solvePoseAndShape(int maxIterTime)
{
	double terminal = 0.01;

	int M = m_poseToOptimize.size();
	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();
		m_iterModel.vertices = m_verticesFinal;
		m_iterModel.CalcNormal();
		findCorr();
		Eigen::MatrixXd ATA;
		Eigen::VectorXd ATb;
		CalcPoseTerm(ATA, ATb);

		Eigen::VectorXd theta(3 + 3 * M);
		theta.segment<3>(0) = m_translation;
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			theta.segment<3>(3 + 3 * i) = m_poseParam.segment<3>(3 * jIdx);
		}
		// solve
		double lambda = 0.0005;
		double w1 = 1;
		double w_reg = 0.001;
		Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXd H_reg = DTD;  // reg term 
		Eigen::VectorXd b_reg = -theta; // reg term 

		Eigen::MatrixXd H = ATA * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXd b = ATb * w1 + b_reg * w_reg;

		std::cout << "pose data b: " << ATb.norm() << std::endl; 
		std::cout << "pose reg  b: " << b_reg.norm() << std::endl; 

		Eigen::VectorXd delta = H.ldlt().solve(b);

		// update 
		m_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
		}
		// if(iterTime == 1) break; 
		if (delta.norm() < terminal) break;

		std::stringstream ss; 
		ss << "E:/debug_pig2/shape/pose_" << iterTime << ".obj";
		SaveObj(ss.str());
	}

	//
	//for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	//{
	//	UpdateVertices();
	//	UpdateVertices();
	//	m_iterModel.vertices = m_verticesFinal;
	//	m_iterModel.CalcNormal();
	//	findCorr();
	//	Eigen::MatrixXd ATA;
	//	Eigen::VectorXd ATb;
	//	CalcShapeTerm(ATA, ATb);

	//	Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(m_shapeNum, m_shapeNum);  // Leveberg Marquart
	//	Eigen::MatrixXd H_reg = DTD;
	//	Eigen::VectorXd b_reg = -m_shapeParam;
	//	double lambda = 0.001;
	//	double w1 = 1;
	//	double w_reg = 0.01;
	//	Eigen::MatrixXd H = ATA * w1 + H_reg * w_reg + DTD * lambda;
	//	Eigen::VectorXd b = ATb * w1 + b_reg * w_reg;

	//	Eigen::VectorXd delta = H.ldlt().solve(b);
	//	m_shapeParam = m_shapeParam + delta; 

	//	if (delta.norm() < terminal) break;
	//}
}

