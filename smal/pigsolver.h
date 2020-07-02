#pragma once

#include <vector>
#include <string>
#include <iostream> 
#include <iomanip> 

#include <Eigen/Eigen>

#include "pigmodel.h"
#include "../associate/skel.h"
#include "../utils/camera.h"
#include "../utils/colorterminal.h"
#include "../utils/image_utils.h"
#include "../utils/math_utils.h"
#include "../utils/model.h"
#include "../utils/node_graph.h"
#include "../utils/kdtree.h"
#include "../utils/volume.h"
#include "../nanorender/NanoRenderer.h"
#include "../nanorender/RenderObject.h"
#include "../utils/model.h" 
#include "../utils/dataconverter.h"
#include "../utils/objloader.h"

//#define DEBUG_SIL

/*
Some Functions for nonrigid deformation borrows from 
https://github.com/zhangyux15/cpp_nonrigid_icp
*/

struct CorrPair
{
	CorrPair() {
		target = -1; 
		type = 0; 
		index = 0;
		weight = 0; 
	}
	int target;
	int type; 
	int index;
	double weight; 
};
class PigSolver : public PigModel
{
public:
	PigSolver(const std::string& _configfile);
	~PigSolver() {}
	PigSolver() = delete; 
	PigSolver(const PigSolver& _) = delete;
	PigSolver& operator=(const PigSolver& _) = delete; 

	void setSource(const MatchedInstance& _source);
	void setMapper(const std::vector<std::pair<int, int> > _mapper) { m_mapper = _mapper; }
	void setCameras(const vector<Camera>& _cameras);
	void setBodySize(const double _alpha) { m_scale = _alpha; }
	void setFrameId(const double _frameid) { m_frameid = _frameid; }
	void setId(const int _id) { m_id = _id;  }
	void setTargetVSameTopo(const Eigen::MatrixXd& _targetV) { m_targetVSameTopo = _targetV; }

	Eigen::MatrixXd              getZ() { return Z; }
	double                       getBodySize() { return m_scale; }
	std::vector<Eigen::Vector3d> getPivot() { return m_pivot;  }
	BodyState&                   getBodyState() { return m_bodystate; }

	void readBodyState(std::string filename); 
	void normalizeCamera();
	void normalizeSource();

	// Fit functions
	Eigen::MatrixXd getRegressedSkel(); 
	Eigen::MatrixXd getRegressedSkelTPose(); 
	Eigen::MatrixXd getRegressedSkelbyPairs(); 

	void globalAlign(); 
	void globalAlignToVerticesSameTopo(); 
	void optimizePose(const int maxIterTime = 100, const double terminal = 0.001);
	void optimizeShapeToBoneLength(int maxIter, double terminal); 
	Eigen::VectorXd getRegressedSkelProj(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& T);
	void computePivot(); 
	void FitShapeToVerticesSameTopo(const int maxIterTime, const double terminal);
	void FitPoseToVerticesSameTopo(const int maxIterTime, const double terminal);

	void CalcZ();

	// targets to fit 
	MatchedInstance m_source;
	vector<Camera> m_cameras;
	std::vector<double> m_weights;
	Eigen::VectorXd m_weightsEigen;
	Eigen::MatrixXd m_targetVSameTopo;

	// fit pose to chamfer map, 20200430
	vector<ROIdescripter> m_rois;
	vector<BodyState>     m_bodies;
	vector<cv::Mat>       m_renders; 
	void optimizePoseSilhouette(int maxIter);
	void CalcSilhouettePoseTerm(const std::vector<cv::Mat>& renders, Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb, int iter);
	nanogui::ref<OffscreenRenderObject> animal_offscreen; 

	std::vector<int> m_poseToOptimize;

	// nodegraph deformation to point cloud
	std::shared_ptr<Model> m_srcModel, m_tarModel;
	std::shared_ptr<const KDTree<double>> m_tarTree;
	Eigen::VectorXi m_corr;
	Eigen::MatrixXd m_deltaTwist;
	Eigen::VectorXd m_wDeform;
	Model m_iterModel;
	double m_wSmth = 0.1;
	double m_wRegular = 1e-3;
	double m_maxDist = 0.35;
	double m_wSym = 0.01;
	double m_maxAngle = double(EIGEN_PI) / 6;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiNode;
	void CalcSmthTerm(Eigen::SparseMatrix<double>& ATA, Eigen::VectorXd& ATb);
	void CalcDeformTerm(Eigen::SparseMatrix<double>& ATA, Eigen::VectorXd& ATb);
	void setTargetModel(std::shared_ptr<Model> m_tarModel);
	void updateWarpField();
	void updateIterModel();
	void solveNonrigidDeform(int maxIterTime, double updateThresh);
	void totalSolveProcedure(); 
	void solvePoseAndShape(int maxIterTime);
	void findCorr();
	void CalcPoseTerm(Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb);
	void CalcShapeTerm(Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb);
	void CalcSymTerm(Eigen::SparseMatrix<double>& ATA, Eigen::VectorXd& ATb);
	// compute volume 
	Volume m_V;
	Model m_V_mesh; 
	void computeVolume();
	std::vector<cv::Mat> m_rawImgs; 

	Eigen::VectorXd theta_last; 
private: 
	// control info 
	int m_id; 
	double m_frameid; 
	std::vector<std::pair<int, int> > m_mapper;
	std::vector<CorrPair> m_optimPairs; 
	SkelTopology m_topo;
	bool tmp_init;

	int m_symNum; 
	std::vector<std::vector<int> > m_symIdx;
	std::vector<std::vector<double> > m_symweights; 

	// inferred data
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> Z; // inferred 3d joints; [3, joint num]
	BodyState m_bodystate; 
	std::vector<Eigen::Vector3d> m_pivot; // head, center, tail.

	
	// Calculate Terms to solve theta 
	void CalcPose2DTermByMapper(const int view, const Eigen::MatrixXd& skel2d,
		const Eigen::MatrixXd& Jacobi3d, 
		Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb);
	void CalcPose2DTermByPairs(const int view, const Eigen::MatrixXd& skel2d,
		const Eigen::MatrixXd& Jacobi3d,
		Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb);

	// Calculate Jacobi Matrix
	void CalcShapeJacobi(Eigen::MatrixXd& jointJacobiShape, Eigen::MatrixXd& vertJacobiShape);

	void CalcShapeJacobiToSkel(Eigen::MatrixXd& J);
	void CalcPoseJacobiFullTheta(Eigen::MatrixXd& J_joint, Eigen::MatrixXd& J_vert);
	void CalcPoseJacobiPartTheta(Eigen::MatrixXd& J_joint, Eigen::MatrixXd& J_vert);
	void CalcSkelJacobiPartThetaByMapper(Eigen::MatrixXd& J);
	void CalcSkelJacobiPartThetaByPairs(Eigen::MatrixXd& J);

	// numeric, only for test
	void CalcPoseJacobiNumeric();
	void CalcShapeJacobiNumeric();
	void Calc2DJacobiNumeric(const int k, const Eigen::MatrixXd& skel,
		Eigen::MatrixXd& H, Eigen::VectorXd& b);
};