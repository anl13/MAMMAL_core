#pragma once

#include <vector>
#include <string>
#include <iostream> 
#include <iomanip> 

#include <Eigen/Eigen>

#include "pigmodel.h"
#include "../utils/skel.h"
#include "../utils/camera.h"
#include "../utils/colorterminal.h"
#include "../utils/image_utils.h"
#include "../utils/math_utils.h"
#include "../utils/node_graph.h"
#include "../utils/kdtree.h"
#include "../utils/volume.h"
#include "../nanorender/NanoRenderer.h"
#include "../nanorender/RenderObject.h"

//#define DEBUG_SIL

/*
Some Functions for nonrigid deformation borrows from 
https://github.com/zhangyux15/cpp_nonrigid_icp
*/


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
	void setBodySize(const float _alpha) { m_scale = _alpha; }
	void setFrameId(const float _frameid) { m_frameid = _frameid; }
	void setId(const int _id) { m_id = _id;  }
	void setTargetVSameTopo(const Eigen::MatrixXf& _targetV) { m_targetVSameTopo = _targetV; }

	Eigen::MatrixXf              getZ() { return Z; }
	float                       getBodySize() { return m_scale; }
	//std::vector<Eigen::Vector3f> getPivot() { return m_pivot;  }
	//BodyState&                   getBodyState() { return m_bodystate; }

	//void readBodyState(std::string filename); 
	void normalizeCamera();
	void normalizeSource();

	// Fit functions
	Eigen::MatrixXf getRegressedSkel(); 
	Eigen::MatrixXf getRegressedSkelTPose(); 
	Eigen::MatrixXf getRegressedSkelbyPairs(); 

	void globalAlign(); 
	
	void optimizePose(const int maxIterTime = 100, const float terminal = 0.001);
	//void optimizeShapeToBoneLength(int maxIter, float terminal); 
	Eigen::VectorXf getRegressedSkelProj(const Eigen::Matrix3f& K, const Eigen::Matrix3f& R, const Eigen::Vector3f& T);
	//void computePivot(); 

	void CalcZ();

	std::vector<Eigen::Vector4f> projectBoxes();

	// targets to fit 
	MatchedInstance m_source;
	vector<Camera> m_cameras;
	std::vector<float> m_weights;
	Eigen::VectorXf m_weightsEigen;


	// fit pose to chamfer map, 20200430
	vector<ROIdescripter> m_rois;
	//vector<BodyState>     m_bodies;
	vector<cv::Mat>       m_renders; 
	void optimizePoseSilhouette(int maxIter);
	void CalcSilhouettePoseTerm(const std::vector<cv::Mat>& renders, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, int iter);
	nanogui::ref<OffscreenRenderObject> animal_offscreen; 

	std::vector<int> m_poseToOptimize;

	//// nodegraph deformation to point cloud
	//std::shared_ptr<Model> m_srcModel, m_tarModel;
	//std::shared_ptr<const KDTree<float>> m_tarTree;
	//Eigen::VectorXi m_corr;
	//Eigen::MatrixXd m_deltaTwist;
	//Eigen::VectorXd m_wDeform;
	//Model m_iterModel;
	//float m_wSmth = 0.1;
	//float m_wRegular = 1e-3;
	//float m_maxDist = 0.35;
	//float m_wSym = 0.01;
	//float m_maxAngle = float(EIGEN_PI) / 6;
	//Eigen::Matrix<float, -1, -1, Eigen::ColMajor> m_vertJacobiNode;
	//void CalcSmthTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXd& ATb);
	//void CalcDeformTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXd& ATb);
	//void setTargetModel(std::shared_ptr<Model> m_tarModel);
	//void updateWarpField();
	//void updateIterModel();
	//void solveNonrigidDeform(int maxIterTime, float updateThresh);
	//void totalSolveProcedure(); 
	//void solvePoseAndShape(int maxIterTime);
	//void findCorr();
	//void CalcPoseTerm(Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb);
	//void CalcShapeTerm(Eigen::MatrixXd& ATA, Eigen::VectorXd& ATb);
	//void CalcSymTerm(Eigen::SparseMatrix<float>& ATA, Eigen::VectorXd& ATb);
	//// compute volume 
	//Volume m_V;
	//Model m_V_mesh; 
	//void computeVolume();
	//std::vector<cv::Mat> m_rawImgs; 

	Eigen::VectorXf theta_last; 

	void debug_numericJacobiLatent();
	void debug_numericJacobiAA(); 

	// debug: 20200801 
	Eigen::MatrixXf m_targetVSameTopo;
	void globalAlignToVerticesSameTopo();
	void FitShapeToVerticesSameTopo(const int maxIterTime, const float terminal);
	float FitPoseToVerticesSameTopo(const int maxIterTime, const float terminal);
	void FitPoseToVerticesSameTopoLatent(); 
	float FitPoseToJointsSameTopo(Eigen::MatrixXf target); 

//private: 
	// control info 
	int m_id; 
	float m_frameid; 
	std::vector<std::pair<int, int> > m_mapper;
	std::vector<CorrPair> m_optimPairs; 
	SkelTopology m_topo;
	bool tmp_init;

	int m_symNum; 
	std::vector<std::vector<int> > m_symIdx;
	std::vector<std::vector<float> > m_symweights; 

	// inferred data
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> Z; // inferred 3d joints; [3, joint num]
	//BodyState m_bodystate; 
	//std::vector<Eigen::Vector3f> m_pivot; // head, center, tail.

	
	// Calculate Terms to solve theta 
	void CalcPose2DTermByMapper(const int view, const Eigen::MatrixXf& skel2d,
		const Eigen::MatrixXf& Jacobi3d, 
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	void CalcPose2DTermByPairs(const int view, const Eigen::MatrixXf& skel2d,
		const Eigen::MatrixXf& Jacobi3d,
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);

	// Calculate Jacobi Matrix
	void CalcShapeJacobi(Eigen::MatrixXf& jointJacobiShape, Eigen::MatrixXf& vertJacobiShape);

	void CalcShapeJacobiToSkel(Eigen::MatrixXf& J);
	void CalcPoseJacobiFullTheta(Eigen::MatrixXf& J_joint, Eigen::MatrixXf& J_vert, bool with_vert=true);
	void CalcPoseJacobiPartTheta(Eigen::MatrixXf& J_joint, Eigen::MatrixXf& J_vert, bool with_vert=true);
	void CalcSkelJacobiPartThetaByMapper(Eigen::MatrixXf& J);
	void CalcSkelJacobiPartThetaByPairs(Eigen::MatrixXf& J);

	// calc jacobi for latent code 
	void CalcPoseJacobiLatent(Eigen::MatrixXf& J_joint, Eigen::MatrixXf& J_vert, bool is_joint_only=false); 
	void CalcSkelJacobiByPairsLatent(Eigen::MatrixXf& J); 
	

	// numeric, only for test
	void CalcPoseJacobiNumeric();
	void CalcShapeJacobiNumeric();
	void Calc2DJacobiNumeric(const int k, const Eigen::MatrixXf& skel,
		Eigen::MatrixXf& H, Eigen::VectorXf& b);
};