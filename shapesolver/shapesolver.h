#pragma once
#include "../articulation/pigmodel.h"
#include "../articulation/pigsolver.h" 

#include "../utils/math_utils.h" 
#include "../utils/image_utils.h" 
#include "../utils/mesh.h"
#include "../utils/volume.h"
#include "../utils/node_graph.h"

class ShapeSolver : public PigSolver
{
public: 
	ShapeSolver() = delete;
	ShapeSolver(const std::string &_configfile);
	~ShapeSolver();
	ShapeSolver(const ShapeSolver&) = delete; 
	ShapeSolver& operator=(const ShapeSolver&) = delete; 

	// compute volume 
	Volume m_V;
	MeshEigen m_V_mesh;
	void computeVolume();

	// nodegraph deformation to point cloud
	std::shared_ptr<MeshEigen> m_srcModel, m_tarModel;
	std::shared_ptr<const KDTree<float>> m_tarTree;
	Eigen::VectorXi m_corr;
	Eigen::MatrixXf m_deltaTwist;
	Eigen::VectorXf m_wDeform;
	MeshEigen m_iterModel;
	float m_wSmth = 0.1;
	float m_wRegular = 1e-3;
	float m_maxDist = 0.35;
	float m_wSym = 0.01;
	float m_maxAngle = float(EIGEN_PI) / 6;
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> m_vertJacobiNode;

	//void CalcSmthTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXf& ATb);
	//void CalcDeformTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXf& ATb);
	//void setTargetModel(std::shared_ptr<MeshEigen> m_tarModel);
	//void updateWarpField();
	//void updateIterModel();
	//void solveNonrigidDeform(int maxIterTime, float updateThresh);
	//void totalSolveProcedure();
	//void solvePoseAndShape(int maxIterTime);
	//void findCorr();
	//void CalcPoseTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	//void CalcShapeTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	//void CalcSymTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXf& ATb);

	// other function 


};