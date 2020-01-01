#pragma once 

#include <vector>
#include <Eigen/Eigen> 
#include <string> 
#include <iostream> 
#include "smal.h" 

class SMAL_SOLVER: public SMAL 
{
public: 
    SMAL_SOLVER(std::string folder); 
	~SMAL_SOLVER(){};
	SMAL_SOLVER() = delete;
	SMAL_SOLVER(const SMAL_SOLVER& _) = delete;
	SMAL_SOLVER& operator=(const SMAL_SOLVER& _) = delete;

    void debug(); 

    void setY(const Eigen::MatrixXd _Y){Y=_Y;}
    void globalAlign(); 
    void globalAlignByVertices(); 
    void OptimizePose(const int maxIterTime = 100, const double terminal = 0.001, bool is_numeric=false);
	void OptimizeShape(const int maxIterTime = 100, const double updateTolerance = 0.001);

private: 
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> Y; // optimization target, surface points, [3 * vertexNum]
    void CalcPoseJacobi(); 
    void CalcShapeJacobi(); 
    void CalcPoseJacobiNumeric(); 
    void CalcShapeJacobiNumeric(); 
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jointJacobiPose;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiPose; 
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jointJacobiShape; 
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiShape; 

    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jointJacobiPoseNumeric;
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiPoseNumeric;
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jointJacobiShapeNumeric;
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_vertJacobiShapeNumeric; 
}; 