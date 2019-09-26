#include "camera.h" 
#include <json/json.h> 
#include <fstream> 


Camera::Camera(const Mat3& _K,
               const Mat3& _R, 
               const Vec3& _T)
{
    K = _K; 
    inv_K = K.inverse(); 

    SetRT(_R, _T); 
}

void Camera::SetRT(const Mat3 &_R, const Vec3 &_T)
{
    R = _R;
    T = _T; 
    inv_R = R.transpose(); 

    // projection directly from global 3-space to local image space
    P_g = Mat4::Zero(); 
    P_g.block<3,3>(0,0) = R; 
    P_g.block<3,1>(0,3) = T;
    P_g(3,3) = 1; 
    P_g.block<3,4>(0,0) = K * P_g.block<3,4>(0,0); 
}

void Camera::SetRT(const Vec3 &Rvec, const Vec3 &_T)
{
    Mat3 Rmat = GetRodrigues(Rvec); 
    SetRT(Rmat, _T); 
}

Mat3 Camera::GetEnssential(
    const Mat3& R2, // R of camera2 
    const Vec3& T2  // T of camera2
) const 
{
    Mat3 R_rel = R * R2.transpose(); 
    Vec3 T_rel = - R * R2.transpose() * T2 + T;
    Mat3 T_skew = GetSkewMatrix(T_rel); 
    return T_skew * R_rel; 
}

Mat3 Camera::GetFundamental(const Mat3& R2, 
                               const Vec3& T2, 
                               const Mat3& inv_K2) const
{
    Mat3 E = GetEnssential(R2, T2); 
    return inv_K.transpose() * E * inv_K2; 
}

Mat3 Camera::GetEnssential(const Camera& cam2)const 
{
    return GetEnssential(cam2.R, cam2.T); 
    
}

Mat3 Camera::GetFundamental(const Camera& cam2)const 
{
    return GetFundamental(cam2.R, cam2.T, cam2.inv_K); 
}

Mat3 Camera::GetRelR(const Camera& cam2) const 
{
    Mat3 R2 = cam2.R; 
    Vec3 T2 = cam2.T; 
    Mat3 R_rel = R * R2.transpose(); 
    Vec3 T_rel = - R * R2.transpose() * T2 + T;
    return R_rel; 
}

Vec3 Camera::GetRelT(const Camera& cam2) const 
{
    Mat3 R2 = cam2.R; 
    Vec3 T2 = cam2.T; 
    Mat3 R_rel = R * R2.transpose(); 
    Vec3 T_rel = - R * R2.transpose() * T2 + T;
    return T_rel; 
}
/// default cameras for pig data
Camera getDefaultCameraRaw()
{
    Mat3 K;
    K << 1623.3,      0, 950.4, 
                   0, 1623.3, 532.7, 
                   0,      0,     1; 
    Vec3 k = {-0.36439, 0.18555, 0};
    Vec2 p = {-0.00121, 0.00036}; 
    Camera camera;
    camera.SetK(K); 
    camera.SetDistortion(k, p); 

    Mat3 R = Mat3::Identity(); 
    Vec3 T = Vec3::Zero(); 
    camera.SetRT(R,T); 

    return camera; 
}

Camera getDefaultCameraUndist()
{
    Mat3 K;
    K << 1362.977,        0, 949.88, 
                     0, 1364.693, 530.18, 
                     0,        0,      1; 
    Vec3 k = {0, 0, 0};
    Vec2 p = {0, 0}; 
    Camera camera;
    camera.SetK(K); 
    camera.SetDistortion(k, p); 

    Mat3 R = Mat3::Identity(); 
    Vec3 T = Vec3::Zero(); 
    camera.SetRT(R,T); 

    return camera; 
}