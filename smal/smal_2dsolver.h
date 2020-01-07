#pragma once 

#include <vector>
#include <Eigen/Eigen> 
#include <string> 
#include <iostream> 
#include "smal.h" 
#include "../associate/skel.h" 
#include "../utils/camera.h"

class SMAL_2DSOLVER: public SMAL 
{
public: 
    SMAL_2DSOLVER(std::string folder); 
	~SMAL_2DSOLVER(){};
	SMAL_2DSOLVER() = delete;
	SMAL_2DSOLVER(const SMAL_2DSOLVER& _) = delete;
	SMAL_2DSOLVER& operator=(const SMAL_2DSOLVER& _) = delete;

    void setSource(const MatchedInstance& _source);
    void setMapper(const std::vector<std::pair<int,int> > _mapper){m_mapper = _mapper;}
    void setCameras(const vector<Camera>& _cameras);
    void setY(const Eigen::MatrixXd& _Y){Y=_Y;}
    void normalizeCamera();
    void normalizeSource();  
    Eigen::MatrixXd getZ(){return Z; } 

    void globalAlign(); 
    void optimizePose(const int maxIterTime = 100, const double terminal = 0.001);
    Eigen::MatrixXd getRegressedSkel(); 
    Eigen::VectorXd getRegressedSkelProj(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& T); 
    void optimizeJoints(); 
private: 
    MatchedInstance m_source; 
    vector<Camera> m_cameras; // all 10 cameras here. 
    Eigen::Matrix<double,-1,-1,Eigen::ColMajor> Y; // [3, target_joint_num]
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> Z; // inferred 3d joints. 
    

    // map smal joint/surface to Y. 
    // (0,x): smal joint with id x
    // (1,k): smal surface point with id k
    // Mapper.size() == Y.size() 
    std::vector<std::pair<int,int> > m_mapper; 
    void CalcPoseJacobi(); 
    void Calc2DJacobi(const int k, const Eigen::MatrixXd& skel, 
            Eigen::MatrixXd& H, Eigen::VectorXd& b); 
    void Calc2DJacobiNumeric(const int k, const Eigen::MatrixXd& skel,
            Eigen::MatrixXd& H, Eigen::VectorXd& b); 

    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_JacobiPose;
    Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_2DJacobi; 

    std::vector<double> m_weights; 
    Eigen::VectorXd m_weightsEigen; 
    SkelTopology m_topo; 
    std::vector<int> m_poseToOptimize; 
}; 