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
	void globalAlign(); 
	void globalAlignToVerticesSameTopo(); 
	void optimizePose(const int maxIterTime = 100, const double terminal = 0.001);
	Eigen::VectorXd getRegressedSkelProj(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& T);
	void computePivot(); 
	void FitShapeToVerticesSameTopo(const int maxIterTime, const double terminal);

	// node graph deformation
	NodeGraph m_ng;
	void UpdateWarpField();
	void UpdateModel();

	// 2020-03-11 shape deformation solver
	vector<ROIdescripter> m_rois;
	vector<BodyState>     m_bodies;
	vector<cv::Mat>       m_renders; 
	void feedData(const ROIdescripter& _roi, 
		const BodyState& _body);
	void feedRender(const cv::Mat& _render);
	void iterateStep(int iter); 
	void clearData(); 

	void CalcZ();

	// 2020-01-17 Tried to solve Regressor matrix, but failed due to memory error.
	void solveR(int samplenum=10); 
	void generateSamples(int samplenum); 
	void solve(); 
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> R; // regressor to solve

	void debug_jacobi();

private: 
	// control info 
	int m_id; 
	double m_frameid; 
	std::vector<std::pair<int, int> > m_mapper;
	SkelTopology m_topo;
	std::vector<int> m_poseToOptimize;
	bool tmp_init;

	// targets to fit 
	MatchedInstance m_source;
	vector<Camera> m_cameras;
	std::vector<double> m_weights;
	Eigen::VectorXd m_weightsEigen;
	Eigen::MatrixXd m_targetVSameTopo;

	// inferred data
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> Z; // inferred 3d joints; [3, joint num]
	BodyState m_bodystate; 
	std::vector<Eigen::Vector3d> m_pivot; // head, center, tail.
	Eigen::Matrix4Xd m_warpField;
	Eigen::VectorXi m_corr;
	Eigen::MatrixXd m_deltaTwist;
	Eigen::VectorXd m_wDeform;
	double m_wSmth = .1f;
	double m_wRegular = 1e-3f;
	double m_maxDist = 0.2f;
	double m_maxAngle = double(EIGEN_PI) / 4;
	
	void Calc2DJacobi(const int k, const Eigen::MatrixXd& skel,
		Eigen::MatrixXd& H, Eigen::VectorXd& b);
	void Calc2DJacobiNumeric(const int k, const Eigen::MatrixXd& skel,
		Eigen::MatrixXd& H, Eigen::VectorXd& b);
	void CalcPoseJacobi();
	void CalcShapeJacobi();
	void CalcPoseJacobiNumeric();
	void CalcShapeJacobiNumeric();
	void CalcVertJacobiNode();

	// optimization parameters
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_JacobiPose;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_2DJacobi;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jointJacobiPose;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiPose;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jointJacobiShape;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiShape;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jointJacobiPoseNumeric;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiPoseNumeric;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jointJacobiShapeNumeric;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiShapeNumeric;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiNode;
	void CalcSmthTerm(Eigen::SparseMatrix<double>& ATA, Eigen::VectorXd& ATb);
};