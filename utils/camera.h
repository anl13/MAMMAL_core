#pragma once 

#include <string> 

#include <Eigen/Dense> 
#include <opencv2/opencv.hpp> 

#include "math_utils.h" 

struct Camera 
{
    Camera() {
        W = 1920; 
        H = 1080; 
    }
    Camera(const Eigen::Matrix3f& _K, 
           const Eigen::Matrix3f& _R, 
           const Eigen::Vector3f& _T);

    void SetK(const Eigen::Matrix3f &_K) {K = _K; inv_K = K.inverse(); }
    void SetRT(const Eigen::Matrix3f &_R, const Eigen::Vector3f &_T); 
    void SetDistortion(const Eigen::Vector3f &_k, const Eigen::Vector2f &_p){k = _k; p = _p;}
    void SetRT(const Eigen::Vector3f &Rvec, const Eigen::Vector3f &T); 
    void NormalizeK(); 

    int W; 
    int H; 

    Eigen::Matrix3f K; // camera intrinsic matrix 
    Eigen::Matrix3f inv_K; // inverse of camera intrinsic matrix
    Eigen::Matrix3f R; // 
    Eigen::Vector3f T; // 
    Eigen::Matrix3f inv_R; 

    Eigen::Vector3f k;  // k1, k2, k3
    Eigen::Vector2f p;  // p1, p2

    Eigen::Matrix4f P_g; // projection from global coordinate to image

    // Given cam2, compute E and F relative to this cam. 
    Eigen::Matrix3f GetEnssential(const Eigen::Matrix3f& globalR, 
                                const Eigen::Vector3f& globalT) const; 
    Eigen::Matrix3f GetEnssential(const Camera& cam2) const; 
    Eigen::Matrix3f GetFundamental(const Eigen::Matrix3f& R2, 
                                   const Eigen::Vector3f& T2, 
                                   const Eigen::Matrix3f& inv_K2) const; 
    Eigen::Matrix3f GetFundamental(const Camera& cam2) const; 
    Eigen::Matrix3f GetRelR(const Camera& cam2) const;
    Eigen::Vector3f GetRelT(const Camera& cam2) const; 

	static Camera getDefaultCameraRaw();
	static Camera getDefaultCameraUndist();
};
