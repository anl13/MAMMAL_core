#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <string> 
#include <Eigen/Dense> 
#include "math_utils.h" 
#include <opencv2/opencv.hpp> 


class Camera 
{
public: 
    Camera() {
        W = 1920; 
        H = 1080; 
    }
    Camera(const Mat3& _K, 
           const Mat3& _R, 
           const Vec3& _T);

    void SetK(const Mat3 &_K) {K = _K; inv_K = K.inverse(); }
    void SetRT(const Mat3 &_R, const Vec3 &_T); 
    void SetDistortion(const Vec3 &_k, const Vec2 &_p){k = _k; p = _p;}
    void SetRT(const Vec3 &Rvec, const Vec3 &T); 
    void NormalizeK(); 

    int uniqueID;  
    int W; 
    int H; 

    Mat3 K; // camera intrinsic matrix 
    Mat3 inv_K; // inverse of camera intrinsic matrix
    Mat3 R; // 
    Vec3 T; // 
    Mat3 inv_R; 

    Vec3 k;  // k1, k2, k3
    Vec2 p;  // p1, p2


    Mat4 P_g; // projection from global coordinate to image

    // Given cam2, compute E and F relative to this cam. 
    Mat3 GetEnssential(const Mat3& globalR, 
                                const Vec3& globalT) const; 
    Mat3 GetEnssential(const Camera& cam2) const; 
    Mat3 GetFundamental(const Mat3& R2, 
                                   const Vec3& T2, 
                                   const Mat3& inv_K2) const; 
    Mat3 GetFundamental(const Camera& cam2) const; 
    Mat3 GetRelR(const Camera& cam2) const;
    Vec3 GetRelT(const Camera& cam2) const; 

};

Camera getDefaultCameraRaw(); 
Camera getDefaultCameraUndist(); 

#endif 