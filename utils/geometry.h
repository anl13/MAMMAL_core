#pragma once 

#include <vector>
#include <iostream> 
#include <fstream> 
#include <iomanip> 
#include <sstream> 

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp> 
#include <ceres/ceres.h> 

#include "math_utils.h" 
#include "image_utils.h"
#include "camera.h" 

// N view triangulation with DLT methods (eigen vector)
Eigen::Vector3f NViewDLT(
    const std::vector<Camera>          &cams, 
    const std::vector<Eigen::Vector2f> &xs
); 

class Joint3DSolver 
{
public: 
    Joint3DSolver(){
        verbose = false; 
        status_init_set = false; 
        }
    ~Joint3DSolver(){}
    void SetParam(const std::vector<Camera>& cams, const std::vector<Eigen::Vector3d>& points);
    void SetParam(const std::vector<Camera>& cams, const std::vector<Eigen::Vector2d>& points, const std::vector<double>& confidences);
    void SetInit( Eigen::Vector3d _x_init); 
    void SetVerbose(bool _v){verbose = _v;}
    // solver function 
    void Solve3D(); // regression with autodiff 
    Eigen::Vector3d GetX(){return X;}
private: 
    // params
    std::vector<Camera>          cams; 
    std::vector<Eigen::Vector2d> u; 
    std::vector<double>          conf; 
    // target variable
    Eigen::Vector3d X; 
    // verbose flag
    bool verbose; 
    bool status_init_set; 
}; 

float getEpipolarDistL2L(const Camera& cam1, const Camera& cam2, const Eigen::Vector2f& u1, const Eigen::Vector2f& u2);
float getEpipolarDistL2L(const Camera& cam1, const Camera& cam2, const Eigen::Vector3f& u1, const Eigen::Vector3f& u2);
float getEpipolarDist(const Eigen::Vector3f& p1, const Eigen::Matrix3f& F, const Eigen::Vector3f& p2);
float getEpipolarDist(const Camera& cam1, const Camera& cam2, const Eigen::Vector3f& u1, const Eigen::Vector3f& u2);


void project(const Camera& cam, const vector<Eigen::Vector3f> &points3d, vector<Eigen::Vector3f> &points2d);
void project(const Camera& cam, const Eigen::Vector3f& p3d, Eigen::Vector3f &p2d);
Eigen::Vector3f project(const Camera& cam, const Eigen::Vector3f& p3d);

Eigen::Vector3d triangulate_ceres(const std::vector<Camera> cams, const std::vector<Eigen::Vector3d> joints2d); 

void test_epipolar(
    const std::vector<Camera> &cameras, 
    std::vector<cv::Mat> &imgs, 
    const std::vector< std::vector<Eigen::Vector2d> > &joints2d, 
    int camid0, 
    int camid1,
    int jid); 

void test_epipolar_all(const std::vector<Camera> &cameras, 
    std::vector<cv::Mat> &imgs, 
    const std::vector< std::vector<Eigen::Vector2f> > &joints2d); 

void test_epipole(
    const std::vector<Camera> &cameras, 
    std::vector<cv::Mat> &imgs, 
    int camid0, 
    int camid1); 

void test_epipole_all(
    const std::vector<Camera> &cameras, 
    std::vector<cv::Mat> &imgs); 