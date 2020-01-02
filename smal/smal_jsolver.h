#pragma once 

#include <vector>
#include <Eigen/Eigen> 
#include <string> 
#include <iostream> 
#include "smal.h" 
#include "../associate/skel.h" 

class SMAL_JSOLVER: public SMAL 
{
public: 
    SMAL_JSOLVER(std::string folder); 
	~SMAL_JSOLVER(){};
	SMAL_JSOLVER() = delete;
	SMAL_JSOLVER(const SMAL_JSOLVER& _) = delete;
	SMAL_JSOLVER& operator=(const SMAL_JSOLVER& _) = delete;
    void setY(const Eigen::MatrixXd _Y){Y=_Y;}
    void setMapper(const std::vector<std::pair<int,int> > _mapper){m_mapper = _mapper;}
    void globalAlign(); 
    void optimizePose(const int maxIterTime = 100, const double terminal = 0.001);
	// void OptimizeShape(const int maxIterTime = 100, const double updateTolerance = 0.001);
    Eigen::MatrixXd getRegressedSkel(); 
private: 
    Eigen::Matrix<double,-1,-1,Eigen::ColMajor> Y; // [3, target_joint_num]
    // map smal joint/surface to Y. 
    // (0,x): smal joint with id x
    // (1,k): smal surface point with id k
    // Mapper.size() == Y.size() 
    std::vector<std::pair<int,int> > m_mapper; 
    void CalcPoseJacobi(); 
    // void CalcShapeJacobi(); 

    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_JacobiPose;
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_JacobiShape; 
    std::vector<double> m_weights; 
    Eigen::VectorXd m_weightsEigen; 
    SkelTopology m_topo; 
    std::vector<int> m_poseToOptimize; 
}; 