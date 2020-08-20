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

double PigSolver::FitPoseToVerticesSameTopo(const int maxIterTime, const double terminal)
{
	Eigen::VectorXd V_target = Eigen::Map<Eigen::VectorXd>(m_targetVSameTopo.data(), 3 * m_vertexNum);

	int M = m_poseToOptimize.size();
	int N = m_topo.joint_num;

	double loss = 0; 
	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		UpdateVertices();
		//std::stringstream ss; 
		//ss << "E:/debug_pig3/shapeiter/pose_" << iterTime << ".obj";
		//SaveObj(ss.str());

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
		Eigen::MatrixXd J; // J_vert
		Eigen::MatrixXd J_joint; 
		CalcPoseJacobiPartTheta(J_joint, J);
		Eigen::MatrixXd H1 = J.transpose() * J;
		Eigen::MatrixXd b1 = -J.transpose() * r; 
			
		double lambda = 0.0001;
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
		loss = delta.norm(); 
		if (loss < terminal) break;
	}
	return loss; 
}

double PigSolver::FitPoseToJointsSameTopo(Eigen::MatrixXd target)
{
	int maxIterTime = 100; 
	double terminal = 0.000000001; 
	TimerUtil::Timer<std::chrono::milliseconds> TT;

	int M = m_poseToOptimize.size();
	double loss = 0;
	int iterTime = 0; 
	Eigen::VectorXd target_vec = Eigen::Map<Eigen::VectorXd>(target.data(), m_jointNum * 3);
	
	TT.Start(); 
	for (; iterTime < maxIterTime; iterTime++)
	{
		UpdateJoints(); 
		Eigen::VectorXd r = Eigen::Map<Eigen::VectorXd>(m_jointsFinal.data(), 3 * m_jointNum) - target_vec;

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
		Eigen::MatrixXd J_vert; // J_vert
		Eigen::MatrixXd J_joint;
		CalcPoseJacobiPartTheta(J_joint, J_vert, false);
		Eigen::MatrixXd H1 = J_joint.transpose() * J_joint;
		Eigen::MatrixXd b1 = -J_joint.transpose() * r;

		double lambda = 0.0001;
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
		loss = delta.norm();
		if (loss < terminal) break;
	}

	double time = TT.Elapsed(); 
	
	std::cout << "iter : " << iterTime << "  loss: " << loss << "  tpi: " << time/ iterTime << std::endl; 
	return loss;
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
		CalcSilhouettePoseTerm(renders, ATA, ATb,iter);
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
	int M = 3+3*m_poseToOptimize.size(); 
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
			A.row(i) = w*(block2d.row(0) * (ddx) + block2d.row(1) * (ddy));
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


void PigSolver::FitPoseToVerticesSameTopoLatent()
{
	int maxIterTime = 100; 
	double terminal = 0.0001; 
	Eigen::VectorXd V_target = Eigen::Map<Eigen::VectorXd>(m_targetVSameTopo.data(), 3 * m_vertexNum);

	for (int iterTime = 0; iterTime < maxIterTime; iterTime++)
	{
		std::cout << "iter: " << iterTime << std::endl; 
		UpdateVertices();
		std::stringstream ss;
		ss << "G:/debug_pig4/poseiter/pose_" << iterTime << ".obj";
		SaveObj(ss.str());

		Eigen::VectorXd r = Eigen::Map<Eigen::VectorXd>(m_verticesFinal.data(), 3 * m_vertexNum) - V_target;

		Eigen::VectorXd theta(3 + 3 + 32);
		theta.segment<3>(0) = m_translation;
		theta.segment<3>(3) = m_poseParam.segment<3>(0); 
		theta.segment<32>(6) = m_latentCode;

		Eigen::VectorXd theta0 = theta;

		Eigen::MatrixXd J_joint, J_vert; 
		CalcPoseJacobiLatent(J_joint, J_vert);
		Eigen::MatrixXd H1 = J_vert.transpose() * J_vert;
		Eigen::MatrixXd b1 = -J_vert.transpose() * r;

		double lambda = 0.001;
		double w1 = 1;
		double w_reg = 0.01;
		Eigen::MatrixXd DTD = Eigen::MatrixXd::Identity(38, 38);
		Eigen::MatrixXd H_reg = DTD;  // reg term 
		Eigen::VectorXd b_reg = -theta; // reg term 

		Eigen::MatrixXd H = H1 * w1 + H_reg * w_reg + DTD * lambda;
		Eigen::VectorXd b = b1 * w1 + b_reg * w_reg;

		Eigen::VectorXd delta = H.ldlt().solve(b);

		// update 
		m_translation += delta.segment<3>(0);
		m_poseParam.segment<3>(0) += delta.segment<3>(3);
		m_latentCode += delta.segment<32>(6); 

		// if(iterTime == 1) break; 
		if (delta.norm() < terminal) break;
	}
}