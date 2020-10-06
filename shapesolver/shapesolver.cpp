#include "shapesolver.h"
#include "../utils/image_utils.h"
#include "../utils/timer_util.h"

ShapeSolver::ShapeSolver(const std::string& _configFile) : PigSolver(_configFile)
{

}

ShapeSolver::~ShapeSolver()
{

}

//void ShapeSolver::CalcPoseTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
//{
//	Eigen::MatrixXf J;
//	Eigen::MatrixXf _Jjoint;
//	CalcPoseJacobiPartTheta(_Jjoint, J, true);
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
//	Eigen::MatrixXd AT = A.transpose();
//	ATA = AT * A;
//	ATb = AT * b;
//
//	std::cout << "pose r: " << b.norm() << std::endl;
//}

//void ShapeSolver::CalcShapeTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
//{
//	Eigen::MatrixXf jointJ, J;
//	CalcShapeJacobi(jointJ, J);
//	int M = J.cols();
//	//Eigen::VectorXd b = Eigen::VectorXd::Zero(m_vertexNum);
//	Eigen::VectorXd b = Eigen::VectorXd::Zero(m_vertexNum * 3);
//	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m_vertexNum * 3, M);
//	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++)
//	{
//		const int tIdx = m_corr[sIdx];
//		const double wd = m_wDeform[sIdx];
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
//	Eigen::MatrixXd AT = A.transpose();
//	ATA = AT * A;
//	ATb = AT * b;
//}

//void ShapeSolver::CalcDeformTerm(
//	Eigen::SparseMatrix<float>& ATA,
//	Eigen::VectorXf& ATb
//)
//{
//	if (m_wDeform.size() != m_vertexNum)
//		m_wDeform = Eigen::VectorXf::Ones(m_vertexNum);
//
//	std::vector<Eigen::Triplet<float>> triplets;
//	Eigen::VectorXf b = Eigen::VectorXf::Zero(m_vertexNum);
//	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> globalAffineNormalized = m_globalAffine;
//	for (int jointId = 0; jointId < m_jointNum; jointId++)
//	{
//		globalAffineNormalized.block<3, 1>(0, jointId * 4 + 3) -= (m_globalAffine.block<3, 3>(0, jointId * 4)*m_jointsShaped.col(jointId));
//	}
//	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++) {
//		const int tIdx = m_corr[sIdx];
//		const float wd = m_wDeform[sIdx];
//		if (tIdx == -1 || wd < FLT_EPSILON)
//			continue;
//
//		const auto tn = m_tarModel->normals.col(tIdx);
//		const auto iv = m_verticesFinal.col(sIdx);
//		const auto tv = m_tarModel->vertices.col(tIdx);
//		Eigen::Matrix<float, 4, 4, Eigen::ColMajor> globalAffineAverage;
//		Eigen::Map<Eigen::VectorXf>(globalAffineAverage.data(), 16)
//			= Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::ColMajor>>
//			(globalAffineNormalized.data(), 16, m_jointNum) * (m_lbsweights.col(sIdx));
//
//		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++)
//		{
//			const int ni = mp_nodeGraph->knn(i, sIdx);
//			const float w = mp_nodeGraph->weight(i, sIdx) * wd;
//			if (w < FLT_EPSILON || ni == -1)
//				continue;
//
//			const int col = 6 * ni;
//			Eigen::MatrixXf dv0 = Eigen::MatrixXf::Zero(3, 6);
//			dv0(0, 1) = iv.z();
//			dv0(0, 2) = -iv.y();
//			dv0(1, 0) = -iv.z();
//			dv0(1, 2) = iv.x();
//			dv0(2, 0) = iv.y();
//			dv0(2, 1) = -iv.x();
//			dv0.middleCols(3, 3) = Eigen::Matrix3f::Identity();
//			dv0 = globalAffineAverage.block<3, 3>(0, 0) * dv0;
//
//			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col, w * (tn.x()*dv0(0, 0) + tn.y()*dv0(1, 0) + tn.z()*dv0(2, 0)))); // alpha_ni
//			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 1, w * (tn.x()*dv0(0, 1) + tn.y()*dv0(1, 1) + tn.z()*dv0(2, 1)))); // beta_ni
//			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 2, w * (tn.x()*dv0(0, 2) + tn.y()*dv0(1, 2) + tn.z()*dv0(2, 2)))); // gamma_ni
//			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 3, w * (tn.x()*dv0(0, 3) + tn.y()*dv0(1, 3) + tn.z()*dv0(2, 3)))); // tx_ni
//			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 4, w * (tn.x()*dv0(0, 4) + tn.y()*dv0(1, 4) + tn.z()*dv0(2, 4)))); // ty_ni
//			triplets.emplace_back(Eigen::Triplet<float>(sIdx, col + 5, w * (tn.x()*dv0(0, 5) + tn.y()*dv0(1, 5) + tn.z()*dv0(2, 5)))); // tz_ni
//		}
//		b[sIdx] = wd * (tn.dot(tv - iv));
//	}
//	Eigen::SparseMatrix<float> A(m_vertexNum, mp_nodeGraph->nodeIdx.size() * 6);
//	A.setFromTriplets(triplets.begin(), triplets.end());
//	const auto AT = A.transpose();
//	ATA = AT * A;
//	ATb = AT * b;
//	std::cout << "deform r: " << b.norm() << std::endl;
//}
//
//void ShapeSolver::CalcSymTerm(
//	Eigen::SparseMatrix<float>& ATA,
//	Eigen::VectorXf& ATb
//)
//{
//	int symNum = m_symNum;
//	/*symNum = 1; */
//	std::vector< std::vector<Eigen::Triplet<float>> > triplets;
//	triplets.resize(symNum);
//	std::vector<Eigen::VectorXf> bs;
//	bs.resize(symNum, Eigen::VectorXf::Zero(3 * m_vertexNum));
//
//	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++)
//	{
//		for (int k = 0; k < symNum; k++)
//		{
//			const int tIdx = m_symIdx[sIdx][k];
//			float tW = m_symweights[sIdx][k];
//
//			if (tIdx < 0 || tW < 0) continue;
//			if (tIdx == sIdx)
//			{
//				// do something 
//				continue;
//			}
//			// else 
//			const auto iv = m_verticesDeformed.col(sIdx);
//			const auto tv = m_verticesDeformed.col(tIdx);
//
//			for (int i = 0; i < mp_nodeGraph->knn.rows(); i++)
//			{
//				const int ni = mp_nodeGraph->knn(i, sIdx);
//				const float wi = mp_nodeGraph->weight(i, sIdx);
//				if (wi < FLT_EPSILON || ni == -1)
//					continue;
//
//				int coli = 6 * ni;
//				int row = 3 * sIdx;
//				triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 1, iv.z() * tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 2, -iv.y()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 3, 1 * tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 0, -iv.z()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 2, iv.x()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 4, 1 * tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 0, iv.y()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 1, -iv.x()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 5, 1 * tW));
//			}
//
//			for (int i = 0; i < mp_nodeGraph->knn.rows(); i++)
//			{
//				const int ni = mp_nodeGraph->knn(i, tIdx);
//				const float wi = mp_nodeGraph->weight(i, tIdx);
//				if (wi < FLT_EPSILON || ni < 0) continue;
//				int coli = 6 * ni;
//				int row = 3 * sIdx;
//				triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 1, -tv.z()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 2, tv.y()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row, coli + 3, -1 * tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 0, -tv.z()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 2, tv.x()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 1, coli + 4, 1 * tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 0, -tv.y()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 1, tv.x()* tW));
//				triplets[k].emplace_back(Eigen::Triplet<float>(row + 2, coli + 5, -1 * tW));
//			}
//
//			// note that, smal model is symmetric to y axis
//			// but my own model is symmetric to x axis
//			Eigen::Vector3f r = Eigen::Vector3f::Zero();
//			r(0) = iv(0) + tv(0);
//			//r(1) = iv(1) - tv(1);
//			//r(2) = iv(2) - tv(2);
//			bs[k].segment<3>(3 * sIdx) = -r * tW;
//		}
//	}
//	ATA.resize(mp_nodeGraph->nodeIdx.size() * 6, mp_nodeGraph->nodeIdx.size() * 6);
//	ATA.setZero();
//	ATb.resize(mp_nodeGraph->nodeIdx.size() * 6);
//	ATb.setZero();
//	for (int k = 0; k < symNum; k++)
//	{
//		Eigen::SparseMatrix<float> A(m_vertexNum * 3, mp_nodeGraph->nodeIdx.size() * 6);
//		A.setFromTriplets(triplets[k].begin(), triplets[k].end());
//		const auto AT = A.transpose();
//		ATA += AT * A;
//		ATb += AT * bs[k];
//	}
//
//	float r = 0;
//	for (int k = 0; k < symNum; k++)
//	{
//		r += bs[k].norm();
//	}
//	std::cout << "total sym r: " << r << std::endl;
//}
//
//void ShapeSolver::CalcSmthTerm(
//	Eigen::SparseMatrix<float>& ATA,
//	Eigen::VectorXf& ATb
//)
//{
//	std::vector<Eigen::Triplet<float>> triplets;
//	Eigen::VectorXf b = Eigen::VectorXf::Zero(3 * mp_nodeGraph->nodeNet.size());
//	for (int ni = 0; ni < mp_nodeGraph->nodeIdx.size(); ni++) {
//		const auto Ti = m_warpField.block<3, 4>(0, 4 * ni);
//		for (int j = 0; j < mp_nodeGraph->nodeNet.rows(); j++) {
//			const int nj = mp_nodeGraph->nodeNet(j, ni);
//			if (nj == -1)
//				continue;
//			// sv: node position 
//			const auto sv = m_verticesShaped.col(mp_nodeGraph->nodeIdx[nj]);
//			const auto Tj = m_warpField.block<3, 4>(0, 4 * nj);
//			const Eigen::Vector3f r = Ti * sv.homogeneous();
//			const Eigen::Vector3f s = Tj * sv.homogeneous();
//			const int row = 3 * (ni * int(mp_nodeGraph->nodeNet.rows()) + j);
//			const int coli = 6 * ni;
//			const int colj = 6 * nj;
//
//			// 1st row
//			triplets.emplace_back(Eigen::Triplet<float>(row, coli + 1, r.z()));
//			triplets.emplace_back(Eigen::Triplet<float>(row, coli + 2, -r.y()));
//			triplets.emplace_back(Eigen::Triplet<float>(row, coli + 3, 1));
//			triplets.emplace_back(Eigen::Triplet<float>(row, colj + 1, -s.z()));
//			triplets.emplace_back(Eigen::Triplet<float>(row, colj + 2, s.y()));
//			triplets.emplace_back(Eigen::Triplet<float>(row, colj + 3, -1));
//
//			// 2nd row
//			triplets.emplace_back(Eigen::Triplet<float>(row + 1, coli + 0, -r.z()));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 1, coli + 2, r.x()));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 1, coli + 4, 1));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 1, colj + 0, s.z()));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 1, colj + 2, -s.x()));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 1, colj + 4, -1));
//
//			// 3rd row
//			triplets.emplace_back(Eigen::Triplet<float>(row + 2, coli + 0, r.y()));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 2, coli + 1, -r.x()));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 2, coli + 5, 1));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 2, colj + 0, -s.y()));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 2, colj + 1, s.x()));
//			triplets.emplace_back(Eigen::Triplet<float>(row + 2, colj + 5, -1));
//
//			// bs
//			b.segment<3>(row) = s - r;
//		}
//	}
//
//	Eigen::SparseMatrix<float> A(3 * mp_nodeGraph->nodeNet.size(), 6 * mp_nodeGraph->nodeIdx.size());
//	A.setFromTriplets(triplets.begin(), triplets.end());
//	const auto AT = A.transpose();
//	ATA = m_wSmth * AT * A;
//	ATb = m_wSmth * AT * b;
//}


//void ShapeSolver::setTargetModel(std::shared_ptr<MeshEigen> _targetModel)
//{
//	m_tarModel = _targetModel;
//	m_tarTree = std::make_shared<KDTree<float>>(m_tarModel->vertices);
//	m_corr = Eigen::VectorXi::Constant(m_vertexNum, -1);
//	m_wDeform = Eigen::VectorXf::Ones(m_vertexNum);
//	m_deltaTwist.resize(6, mp_nodeGraph->nodeIdx.size());
//
//	m_srcModel = std::make_shared<MeshEigen>();
//	m_srcModel->vertices = m_verticesFinal;
//	m_srcModel->faces = m_facesVert;
//	m_srcModel->CalcNormal();
//}

//void ShapeSolver::findCorr()
//{
//	m_iterModel.faces = m_facesVert;
//	m_iterModel.vertices = m_verticesFinal;
//	m_iterModel.CalcNormal();
//	const float cosine = std::cosf(m_maxAngle);
//	m_corr.setConstant(-1);
//#pragma omp parallel for
//	for (int sIdx = 0; sIdx < m_iterModel.vertices.cols(); sIdx++) {
//		if (m_wDeform[sIdx] < FLT_EPSILON)
//			continue;
//		const Eigen::Vector3d v = m_iterModel.vertices.col(sIdx);
//		const std::vector<std::pair<float, size_t>> nbors = m_tarTree->KNNSearch(v, 256);
//		for (const auto& nbor : nbors) {
//			if (nbor.first < m_maxDist) {
//				if (m_iterModel.normals.col(sIdx).dot(m_tarModel->normals.col(nbor.second)) > cosine) {
//					m_corr[sIdx] = int(nbor.second);
//					break;
//				}
//			}
//		}
//	}
//
//	std::cout << "corr:" << (m_corr.array() >= 0).count() << std::endl;
//}

//void ShapeSolver::updateWarpField()
//{
//#pragma omp parallel for
//	for (int ni = 0; ni < m_deltaTwist.cols(); ni++)
//		m_warpField.middleCols(4 * ni, 4) = Twist(m_deltaTwist.col(ni))* m_warpField.middleCols(4 * ni, 4);
//}

//void ShapeSolver::solveNonrigidDeform(int maxIterTime, float updateThresh)
//{
//	Eigen::SparseMatrix<float> ATA, ATAs, ATAsym;
//	Eigen::VectorXf ATb, ATbs, ATbsym;
//	Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
//
//	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
//	{
//		UpdateVertices();
//		m_iterModel.vertices = m_verticesFinal;
//		m_iterModel.CalcNormal();
//		findCorr();
//		CalcDeformTerm(ATA, ATb);
//		CalcSmthTerm(ATAs, ATbs);
//
//		Eigen::SparseMatrix<float> ATAr(6 * mp_nodeGraph->nodeIdx.size(), 6 * mp_nodeGraph->nodeIdx.size());
//		ATAr.setIdentity();
//		ATAr *= m_wRegular;
//		ATA += ATAr;
//		ATA += ATAs * m_wSmth;
//		ATb += ATbs * m_wSmth;
//		if (m_symIdx.size() > 0)
//		{
//			CalcSymTerm(ATAsym, ATbsym);
//			ATA += ATAsym * m_wSym;
//			ATb += ATbsym * m_wSym;
//		}
//
//		Eigen::Map<Eigen::VectorXf>(m_deltaTwist.data(), m_deltaTwist.size()) = solver.compute(ATA).solve(ATb);
//		assert(solver.info() == 0);
//
//		if (m_deltaTwist.norm() < updateThresh)
//			break;
//
//		// debug 
//		std::cout << "delta twist: " << m_deltaTwist.norm() << std::endl;
//		updateWarpField();
//		updateIterModel();
//	}
//}

//void ShapeSolver::updateIterModel()
//{
//	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++) {
//		Eigen::Matrix4f T = Eigen::Matrix4f::Zero();
//
//		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++) {
//			const int ni = mp_nodeGraph->knn(i, sIdx);
//			if (ni != -1)
//				T += mp_nodeGraph->weight(i, sIdx) * m_warpField.middleCols(4 * ni, 4);
//		}
//
//		m_iterModel.vertices.col(sIdx) = T.topLeftCorner(3, 4) * m_srcModel->vertices.col(sIdx).homogeneous();
//		m_iterModel.normals.col(sIdx) = T.topLeftCorner(3, 3) * m_srcModel->normals.col(sIdx);
//	}
//}

//void ShapeSolver::totalSolveProcedure()
//{
//	for (int i = 0; i < 5; i++)
//	{
//		// step1: 
//		m_wSmth = 1.0;
//		m_wRegular = 0.1;
//		m_maxDist = 0.35;
//		m_wSym = 0.01;
//		solveNonrigidDeform(1, 1e-5);
//
//		std::stringstream ss;
//		ss << "E:/debug_pig3/shapeiter/shape_" << i << ".obj";
//		SaveObj(ss.str());
//	}
//
//	for (int i = 0; i < 10; i++)
//	{
//		std::cout << "Second phase: " << i << std::endl;
//		// step1: 
//		m_wSmth = 1.0;
//		m_wRegular = 0.05;
//		m_maxDist = 0.35;
//		m_wSym = 0.7;
//		solveNonrigidDeform(1, 1e-5);
//		std::stringstream ss;
//		ss << "E:/debug_pig3/shapeiter/shape_2_" << i << ".obj";
//		SaveObj(ss.str());
//
//		solvePoseAndShape(1);
//		std::stringstream ss1;
//		ss1 << "E:/debug_pig3/shapeiter/pose_" << i << ".obj";
//		SaveObj(ss1.str());
//	}
//}


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
