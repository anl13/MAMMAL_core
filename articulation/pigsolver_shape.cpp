#include "pigsolver.h"
#include "../utils/image_utils.h"
#include "../utils/timer_util.h"

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

	std::cout << "pose r: " << b.norm() << std::endl;
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
	if (m_wDeform.size() != m_vertexNum)
		m_wDeform = Eigen::VectorXd::Ones(m_vertexNum);

	std::vector<Eigen::Triplet<double>> triplets;
	Eigen::VectorXd b = Eigen::VectorXd::Zero(m_vertexNum);
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
	Eigen::SparseMatrix<double> A(m_vertexNum, mp_nodeGraph->nodeIdx.size() * 6);
	A.setFromTriplets(triplets.begin(), triplets.end());
	const auto AT = A.transpose();
	ATA = AT * A;
	ATb = AT * b;
	std::cout << "deform r: " << b.norm() << std::endl;
}

void PigSolver::CalcSymTerm(
	Eigen::SparseMatrix<double>& ATA,
	Eigen::VectorXd& ATb
)
{
	int symNum = m_symNum;
	/*symNum = 1; */
	std::vector< std::vector<Eigen::Triplet<double>> > triplets;
	triplets.resize(symNum); 
	std::vector<Eigen::VectorXd> bs;
	bs.resize(symNum, Eigen::VectorXd::Zero(3 * m_vertexNum));

	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++) 
	{
		for (int k = 0; k < symNum; k++)
		{
			const int tIdx = m_symIdx[sIdx][k];
			double tW = m_symweights[sIdx][k];

			if (tIdx < 0 || tW < 0) continue;
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
				triplets[k].emplace_back(Eigen::Triplet<double>(row, coli + 1, iv.z() * tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row, coli + 2, -iv.y()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row, coli + 3, 1 * tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 1, coli + 0, -iv.z()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 1, coli + 2, iv.x()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 1, coli + 4, 1 * tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 2, coli + 0, iv.y()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 2, coli + 1, -iv.x()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 2, coli + 5, 1 * tW));
			}

			for (int i = 0; i < mp_nodeGraph->knn.rows(); i++)
			{
				const int ni = mp_nodeGraph->knn(i, tIdx);
				const double wi = mp_nodeGraph->weight(i, tIdx);
				if (wi < DBL_EPSILON || ni < 0) continue;
				int coli = 6 * ni;
				int row = 3 * sIdx;
				triplets[k].emplace_back(Eigen::Triplet<double>(row, coli + 1, -tv.z()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row, coli + 2, tv.y()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row, coli + 3, -1 * tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 1, coli + 0, -tv.z()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 1, coli + 2, tv.x()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 1, coli + 4, 1 * tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 2, coli + 0, -tv.y()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 2, coli + 1, tv.x()* tW));
				triplets[k].emplace_back(Eigen::Triplet<double>(row + 2, coli + 5, -1 * tW));
			}

			// note that, smal model is symmetric to y axis
			// but my own model is symmetric to x axis
			Eigen::Vector3d r = Eigen::Vector3d::Zero();
			r(0) = iv(0) + tv(0);
			//r(1) = iv(1) - tv(1);
			//r(2) = iv(2) - tv(2);
			bs[k].segment<3>(3 * sIdx) = -r * tW;
		}
	}
	ATA.resize(mp_nodeGraph->nodeIdx.size() * 6, mp_nodeGraph->nodeIdx.size() * 6);
	ATA.setZero();
	ATb.resize(mp_nodeGraph->nodeIdx.size() * 6);
	ATb.setZero(); 
	for (int k = 0; k < symNum; k++)
	{
		Eigen::SparseMatrix<double> A(m_vertexNum * 3, mp_nodeGraph->nodeIdx.size() * 6);
		A.setFromTriplets(triplets[k].begin(), triplets[k].end());
		const auto AT = A.transpose();
		ATA += AT * A;
		ATb += AT * bs[k];
	}

	double r = 0; 
	for (int k = 0; k < symNum; k++)
	{
		r += bs[k].norm();
	}
	std::cout << "total sym r: " << r << std::endl;
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

	m_srcModel = std::make_shared<Model>();
	m_srcModel->vertices = m_verticesFinal;
	m_srcModel->faces = m_facesVert;
	m_srcModel->CalcNormal();
}

void PigSolver::findCorr()
{
	m_iterModel.faces = m_facesVert;
	m_iterModel.vertices = m_verticesFinal;
	m_iterModel.CalcNormal();
	const double cosine = std::cosf(m_maxAngle);
	m_corr.setConstant(-1);
#pragma omp parallel for
	for (int sIdx = 0; sIdx < m_iterModel.vertices.cols(); sIdx++) {
		if (m_wDeform[sIdx] < FLT_EPSILON)
			continue;
		const Eigen::Vector3d v = m_iterModel.vertices.col(sIdx);
		const std::vector<std::pair<double, size_t>> nbors = m_tarTree->KNNSearch(v, 256);
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
		}

		Eigen::Map<Eigen::VectorXd>(m_deltaTwist.data(), m_deltaTwist.size()) = solver.compute(ATA).solve(ATb);
		assert(solver.info() == 0);

		if (m_deltaTwist.norm() < updateThresh)
			break;

		// debug 
		std::cout << "delta twist: " << m_deltaTwist.norm() << std::endl;
		updateWarpField();
		updateIterModel();
	}
}

void PigSolver::updateIterModel()
{
	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++) {
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
	for (int i = 0; i < 5; i++)
	{
		// step1: 
		m_wSmth = 1.0;
		m_wRegular = 0.1;
		m_maxDist = 0.35;
		m_wSym = 0.01;
		solveNonrigidDeform(1, 1e-5);

		std::stringstream ss; 
		ss << "E:/debug_pig3/shapeiter/shape_" << i << ".obj";
		SaveObj(ss.str());
	}

	for (int i = 0; i < 10; i++)
	{
		std::cout << "Second phase: " << i << std::endl; 
		// step1: 
		m_wSmth = 1.0;
		m_wRegular = 0.05;
		m_maxDist = 0.35;
		m_wSym = 0.7;
		solveNonrigidDeform(1, 1e-5);
		std::stringstream ss;
		ss << "E:/debug_pig3/shapeiter/shape_2_" << i << ".obj";
		SaveObj(ss.str());

		solvePoseAndShape(1);
		std::stringstream ss1;
		ss1 << "E:/debug_pig3/shapeiter/pose_" << i << ".obj";
		SaveObj(ss1.str());
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

		Eigen::MatrixXd poseJ3d;
		CalcSkelJacobiPartThetaByPairs(poseJ3d);
		Eigen::MatrixXd skel = getRegressedSkelbyPairs();
		Eigen::MatrixXd Hjoint = Eigen::MatrixXd::Zero(3 + 3 * M, 3 + 3 * M); // data term 
		Eigen::VectorXd bjoint = Eigen::VectorXd::Zero(3 + 3 * M);
		for (int k = 0; k < m_source.view_ids.size(); k++)
		{
			Eigen::MatrixXd H_view;
			Eigen::VectorXd b_view;
			CalcPose2DTermByPairs(k, skel, poseJ3d, H_view, b_view);
		    Hjoint += H_view;
			bjoint += b_view;
		}

		// solve
		double lambda = 0.0005;
		double w1 = 1;
		double w_reg = 0.001;
		Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(3 + 3 * M, 3 + 3 * M);
		Eigen::MatrixXd H_reg = DTD;  // reg term 
		Eigen::VectorXd b_reg = -theta; // reg term 

		Eigen::MatrixXd H = 
			Hjoint + ATA * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXd b = 
			bjoint + ATb * w1 + b_reg * w_reg;

		Eigen::VectorXd delta = H.ldlt().solve(b);

		// update 
		m_translation += delta.segment<3>(0);
		for (int i = 0; i < M; i++)
		{
			int jIdx = m_poseToOptimize[i];
			m_poseParam.segment<3>(3 * jIdx) += delta.segment<3>(3 + 3 * i);
		}
		if (delta.norm() < terminal) break;
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
