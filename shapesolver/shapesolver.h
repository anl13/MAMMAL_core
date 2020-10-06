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

	Eigen::MatrixXf Delta0; // init laplacian coordinates  [3, m_vertexNum]
	Eigen::VectorXf D; // [11239] degree
	Eigen::MatrixXf A; // [11239 * 11239] adjecency matrix
	Eigen::MatrixXf L; // L = I - D^-1 A, laplacian matrix 
	void initLaplacian(); 

	// nodegraph deformation to point cloud
	std::shared_ptr<MeshEigen> m_srcModel, m_tarModel;
	std::shared_ptr<const KDTree<float>> m_tarTree;
	Eigen::VectorXi m_corr;
	Eigen::MatrixXf m_deltaTwist;
	Eigen::VectorXf m_wDeform;
	MeshEigen m_iterModel;
	float m_wSmth = 1;
	float m_wRegular = 0.1;
	float m_maxDist = 0.35;
	float w_point = 1; 
	float m_wSym = 1;
	float m_maxAngle = float(EIGEN_PI) / 6;
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> m_vertJacobiNode;

	void CalcPointTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb); 
	void CalcLaplacianTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb); 

	void CalcSmthTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	void CalcDeformTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	void setTargetModel(std::shared_ptr<MeshEigen> m_tarModel);
	void updateWarpField();
	void updateIterModel();
	void solveNonrigidDeform(int maxIterTime, float updateThresh);
	void totalSolveProcedure();
	//void solvePoseAndShape(int maxIterTime);
	void findCorr();
	void CalcPoseTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	//void CalcShapeTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	void CalcSymTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);

	// other function 
	Eigen::MatrixXf dv_dse3; // [3*VN, 6*M]
	Eigen::MatrixXf dvs_dse3; 
	void CalcDvDSe3(); 
};