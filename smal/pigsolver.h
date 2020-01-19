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

class PigSolver : public PigModel
{
public:
	PigSolver(std::string folder) :PigModel(folder) {
		m_topo = getSkelTopoByType("UNIV"); 
		m_poseToOptimize =
		{
			0, 4, 5, 6, 7, 8, 20, 21, 22, 23, 26, 27, 28, 38, 39 ,40, 11, 14
		};
	} 
	~PigSolver() {}
	PigSolver() = delete; 
	PigSolver(const PigSolver& _) = delete;
	PigSolver& operator=(const PigSolver& _) = delete; 

	void setSource(const MatchedInstance& _source);
	void setMapper(const std::vector<std::pair<int, int> > _mapper) { m_mapper = _mapper; }
	void setCameras(const vector<Camera>& _cameras);
	void normalizeCamera();
	void normalizeSource();
	Eigen::MatrixXd getZ() { return Z; }

	Eigen::MatrixXd getRegressedSkel(); 
	void globalAlign(); 
	void optimizePose(const int maxIterTime = 100, const double terminal = 0.001);
	Eigen::VectorXd getRegressedSkelProj(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& T);

	// 2020-01-17 Tried to solve Regressor matrix, but failed due to memory error.
	void solveR(int samplenum=10); 
	void generateSamples(int samplenum); 
	void solve(); 
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> R; // regressor to solve
private: 
	
	MatchedInstance m_source;
	vector<Camera> m_cameras;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> Z; // inferred 3d joints; [3, joint num]

	std::vector<std::pair<int, int> > m_mapper; 
	void CalcPoseJacobi(); 
	void Calc2DJacobi(const int k, const Eigen::MatrixXd& skel,
		Eigen::MatrixXd& H, Eigen::VectorXd& b);
	void CalcZ(); 

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_JacobiPose;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_2DJacobi;

	std::vector<double> m_weights;
	Eigen::VectorXd m_weightsEigen;
	SkelTopology m_topo;
	std::vector<int> m_poseToOptimize;
};