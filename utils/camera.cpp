#include <fstream> 

#include <json/json.h> 

#include "camera.h" 


Camera::Camera(const Eigen::Matrix3f& _K,
               const Eigen::Matrix3f& _R, 
               const Eigen::Vector3f& _T)
{
    K = _K; 
    inv_K = K.inverse(); 
    SetRT(_R, _T); 
    W = 1920; 
    H = 1080; 
}

void Camera::SetRT(const Eigen::Matrix3f &_R, const Eigen::Vector3f &_T)
{
    R = _R;
    T = _T; 
    inv_R = R.transpose(); 

    // projection directly from global 3-space to local image space
    P_g = Eigen::Matrix4f::Zero(); 
    P_g.block<3,3>(0,0) = R; 
    P_g.block<3,1>(0,3) = T;
    P_g(3,3) = 1; 
    P_g.block<3,4>(0,0) = K * P_g.block<3,4>(0,0); 
}

void Camera::SetRT(const Eigen::Vector3f &Rvec, const Eigen::Vector3f &_T)
{
    Eigen::Matrix3f Rmat = GetRodrigues(Rvec); 
    SetRT(Rmat, _T); 
}

Eigen::Matrix3f Camera::GetEnssential(
    const Eigen::Matrix3f& R2, // R of camera2 
    const Eigen::Vector3f& T2  // T of camera2
) const 
{
    Eigen::Matrix3f R_rel = R * R2.transpose(); 
    Eigen::Vector3f T_rel = - R * R2.transpose() * T2 + T;
    Eigen::Matrix3f T_skew = GetSkewMatrix(T_rel); 
    return T_skew * R_rel; 
}

Eigen::Matrix3f Camera::GetFundamental(const Eigen::Matrix3f& R2, 
                               const Eigen::Vector3f& T2, 
                               const Eigen::Matrix3f& inv_K2) const
{
    Eigen::Matrix3f E = GetEnssential(R2, T2); 
    return inv_K.transpose() * E * inv_K2; 
}

Eigen::Matrix3f Camera::GetEnssential(const Camera& cam2)const 
{
    return GetEnssential(cam2.R, cam2.T); 
    
}

Eigen::Matrix3f Camera::GetFundamental(const Camera& cam2)const 
{
    return GetFundamental(cam2.R, cam2.T, cam2.inv_K); 
}

Eigen::Matrix3f Camera::GetRelR(const Camera& cam2) const 
{
    Eigen::Matrix3f R2 = cam2.R; 
    Eigen::Vector3f T2 = cam2.T; 
    Eigen::Matrix3f R_rel = R * R2.transpose(); 
    Eigen::Vector3f T_rel = - R * R2.transpose() * T2 + T;
    return R_rel; 
}

Eigen::Vector3f Camera::GetRelT(const Camera& cam2) const 
{
    Eigen::Matrix3f R2 = cam2.R; 
    Eigen::Vector3f T2 = cam2.T; 
    Eigen::Matrix3f R_rel = R * R2.transpose(); 
    Eigen::Vector3f T_rel = - R * R2.transpose() * T2 + T;
    return T_rel; 
}

void Camera::NormalizeK()
{
    K.row(0) /= W;
    K.row(1) /= H; 
    inv_K = K.inverse(); 
    P_g.block<3,3>(0,0) = K * R; 
    P_g.block<3,1>(0,3) = K * T; 
}

/// default cameras for pig data
 Camera Camera::getDefaultCameraRaw()
{
    Eigen::Matrix3f K;
    K << 1625.30923f,      0.f, 963.88710f, 
                   0.f, 1625.34802f, 523.45901f, 
                   0.f,      0.f,     1.f; 
    Eigen::Vector3f k = {-0.35582f, 0.14595f, 0.f};
    Eigen::Vector2f p = { -0.00031f, -0.00004f}; 
    Camera camera;
    camera.SetK(K); 
    camera.SetDistortion(k, p); 

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity(); 
    Eigen::Vector3f T = Eigen::Vector3f::Zero(); 
    camera.SetRT(R,T); 
    return camera; 
}

 Camera Camera::getDefaultCameraUndist()
{
    Eigen::Matrix3f K;
    K << 1340.0378f,        0.f, 964.7579f, 
                     0.f, 1342.6888f, 521.4926f, 
                     0.f,        0.f,      1.f; 
    Eigen::Vector3f k = {0.f, 0.f, 0.f};
    Eigen::Vector2f p = {0.f, 0.f}; 
    Camera camera;
    camera.SetK(K); 
    camera.SetDistortion(k, p); 

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity(); 
    Eigen::Vector3f T = Eigen::Vector3f::Zero(); 
    camera.SetRT(R,T); 

    return camera; 
}

