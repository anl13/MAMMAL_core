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

using std::vector; 

// N view triangulation with DLT methods (eigen vector)
Eigen::Vector3d NViewDLT(
    const std::vector<Camera>          &cams, 
    const std::vector<Eigen::Vector2d> &xs
); 

double getEpipolarDistL2L(const Camera& cam1, const Camera& cam2, const Vec2& u1, const Vec2& u2);
double getEpipolarDistL2L(const Camera& cam1, const Camera& cam2, const Vec3& u1, const Vec3& u2);
double getEpipolarDist(const Camera& cam1, const Camera& cam2, const Vec3& u1, const Vec3& u2); 
double getEpipolarDist(const Vec3& p1, const Mat3& F, const Vec3& p2); 

void project(const Camera& cam, const vector<Eigen::Vector3d> &points3d, vector<Eigen::Vector3d> &points2d); 
void project(const Camera& cam, Eigen::Vector3d p3d, Eigen::Vector3d &p2d); 
Eigen::Vector3d project(const Camera& cam, Eigen::Vector3d p3d); 

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
    const std::vector< std::vector<Eigen::Vector2d> > &joints2d); 

void test_epipole(
    const std::vector<Camera> &cameras, 
    std::vector<cv::Mat> &imgs, 
    int camid0, 
    int camid1); 

void test_epipole_all(
    const std::vector<Camera> &cameras, 
    std::vector<cv::Mat> &imgs); 