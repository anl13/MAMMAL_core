#include "math_utils.h"
#include <algorithm>

Mat3 GetSkewMatrix(const Vec3& w)
{
    Mat3 expw = Mat3::Zero(); 
    expw(0,1) = -w(2); 
    expw(0,2) = w(1);
    expw(1,0) = w(2); 
    expw(1,2) = -w(0); 
    expw(2,0) = -w(1); 
    expw(2,1) = w(0); 
    return expw; 
}

// w: axis-angle vector 
Mat3 GetRodrigues(const Vec3& w)
{
    cv::Mat cv_w(3,1,CV_64F); 
    cv::eigen2cv(w,cv_w); 
    
    cv::Mat cv_rod(3,3,CV_64F); 
    cv::Rodrigues(cv_w,cv_rod); 

    Mat3 eigen_rod; 
    cv::cv2eigen(cv_rod, eigen_rod); 
    return eigen_rod; 
}

Vec4 ToHomogeneous(const Vec3 &_v)
{
    Vec4 v2;
    v2.x() = _v.x();
    v2.y() = _v.y();
    v2.z() = _v.z();
    v2.w() = 1.0f;

    return v2;
}

Vec3 ToHomogeneous(const Vec2 &_v)
{
    Vec3 v2;
    v2.x() = _v.x();
    v2.y() = _v.y();
    v2.z() = 1.0;
    return v2;
}

Vec3 FromHomogeneous(const Vec4 &_v)
{
    Vec3 v2;
    v2 = _v.block<3,1>(0,0);
    if(_v(3) < 1e-8) return v2; 
    else return v2 / _v(3);
}

Mat4 ToHomogeneous(const Mat3 &_m, 
                                        Vec3 &_v)
{
    Mat4 m2 = Mat4::Identity();
    m2.block<3,3>(0,0) = _m; 
    m2.block<3,1>(0,3) = _v;

    return m2;
}

Vec4 AxisAngleToQuat(const Vec3 &v)
{
    float angle = v.norm(); 
    if(angle == 0) return Vec4::Zero(); 
    Vec3 n = v/angle; 
    Vec4 quat; 
    quat.block<3,1>(0,0) = n * sinf(angle/2); 
    quat[3] = cosf(angle/2); 

    return quat; 
}

Vec3 Quat2AxisAngle(const Vec4 &q)   // TODO: bugs here, NOT reciprocal inverse to AxisAngleToQuat
{
    float angle = acosf(q[3]) * 2; 
    Vec3 v; 
    float sin_angle = sqrt(1-q[3]*q[3]);
    v = q.block<3,1>(0,0) / sin_angle;  // TODO: handle corner case 
    return v * angle; 
}

Vec3 AvgAxisAngles(const std::vector<Vec3> &rots)
{
    std::vector<Vec4> quats; 
    for(int i=0;i<rots.size();i++) quats.push_back(AxisAngleToQuat(rots[i])); 
    Vec4 quat_avg = AvgQuats(quats); 

    return Quat2AxisAngle(quat_avg); 
}

//http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872_2007014421.pdf
// 
Vec4 AvgQuats(const std::vector<Vec4> &quats)
{
    Mat4 M = Mat4::Zero(); 
    for(int i=0;i<quats.size();i++)
    {
        M = M + quats[i] * quats[i].transpose(); 
    }

    Eigen::JacobiSVD<Mat4> svd(M, 
        Eigen::ComputeFullU | Eigen::ComputeFullV | Eigen::ColPivHouseholderQRPreconditioner); 
    
    auto V = svd.matrixV(); 
    auto q_avg = V.col(0); // eigen vector corresponding to the largest eigen value 

    return q_avg; 
}

/*********double functions for ceres***************/ 
// Eigen::Matrix3d CalcRodrigues(const Eigen::Vector3d &vec) 
// {
// 	double theta = sqrt(pow(vec(0), 2) + pow(vec(1), 2) + pow(vec(2), 2));
// 	if (theta < 1e-6)
// 	{
// 		return Eigen::Matrix3d::Identity();
// 	}
// 	else
// 	{
// 		Eigen::Vector3d r = vec / theta;

// 		Eigen::Matrix3d last = Eigen::Matrix3d::Zero();
// 		{
// 			last(0, 1) = -r(2);
// 			last(0, 2) = r(1);
// 			last(1, 0) = r(2);
// 			last(1, 2) = -r(0);
// 			last(2, 0) = -r(1);
// 			last(2, 1) = r(0);
// 		}

// 		Eigen::Matrix3d rodriguesMatrix = cos(theta)*Eigen::Matrix3d::Identity()
// 			+ (1.0 - cos(theta))*r * r.transpose() + sin(theta)*last;
// 		return rodriguesMatrix;
// 	}
// }

Eigen::Matrix3f GetRodriguesF(const Eigen::Vector3f &w)
{
    cv::Mat cv_w(3,1,CV_32F); 
    cv::eigen2cv(w,cv_w); 
    
    cv::Mat cv_rod(3,3,CV_32F); 
    cv::Rodrigues(cv_w,cv_rod); 

    Eigen::Matrix3f eigen_rod; 
    cv::cv2eigen(cv_rod, eigen_rod); 
    return eigen_rod; 
}

double L2LDist(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4)
{
    Vec3 a = x2 - x1; 
    Vec3 b = x4 - x3;
    Vec3 c = x3 - x1;
    double d = c.dot(a.cross(b)) / (a.cross(b)).norm();
    return fabs(d); 
}

double L2LDist(const Vec3& a, const Vec3& b, const Vec3& c)
{
    double d = c.dot(a.cross(b)) / (a.cross(b)).norm();
    return fabs(d); 
}


Eigen::Matrix<float, 3, 9, Eigen::ColMajor> RodriguesJacobiF(const Eigen::Vector3f& vec)
{
    cv::Mat cvVec;
    cv::Mat cvRodriguesMat;
    cv::Mat cvJacobiMat;
    Eigen::Matrix<float, 3, 9, Eigen::RowMajor> eigenJacobiMat;

    cv::eigen2cv(vec, cvVec);
    cv::Rodrigues(cvVec, cvRodriguesMat, cvJacobiMat);
    cv::cv2eigen(cvJacobiMat, eigenJacobiMat);

    Eigen::Matrix<float, 3, 9, Eigen::ColMajor> jacobiMat;
    for (int i = 0; i < 3; i++)
    {
        jacobiMat.block<3, 3>(0, 3 * i) = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(eigenJacobiMat.row(i).data());
    }
    return jacobiMat;
}

Eigen::Matrix<double, 3, 9, Eigen::ColMajor> RodriguesJacobiD(const Eigen::Vector3d& vec)
{
    cv::Mat cvVec;
    cv::Mat cvRodriguesMat;
    cv::Mat cvJacobiMat;
    Eigen::Matrix<double, 3, 9, Eigen::RowMajor> eigenJacobiMat;

    cv::eigen2cv(vec, cvVec);
    cv::Rodrigues(cvVec, cvRodriguesMat, cvJacobiMat);
    cv::cv2eigen(cvJacobiMat, eigenJacobiMat);

    Eigen::Matrix<double, 3, 9, Eigen::ColMajor> jacobiMat;
    for (int i = 0; i < 3; i++)
    {
        jacobiMat.block<3, 3>(0, 3 * i) = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(eigenJacobiMat.row(i).data());
    }
    return jacobiMat;
}

double p2ldist( Vec3 x,  Vec3 line){
    x.normalize(); 
    line.normalize(); 
    return (x.cross(line)).norm(); 
}

double p2ldist( Vec3 x,  Vec3 a,  Vec3 b)
{
    Vec3 line = a-b; 
    Vec3 x_line = x-b;
    return p2ldist(x_line, line); 
}

bool my_equal(std::vector<int> a, std::vector<int> b)
{
    if(a.size() != b.size()) return false;
    // bool indicator = true;  
    // std::sort(a.begin(), a.end()); 
    // std::sort(b.begin(), b.end()); 
    for(int i = 0; i < a.size(); i++)
    {
        if(a[i] != b[i]) {return false;} 
    }
    return true; 
}

bool my_contain(std::vector<int> full, std::vector<int> sub)
{
    if(full.size() < sub.size()) return false; 
    std::sort(full.begin(), full.end()); 
    std::sort(sub.begin(), sub.end()); 
    for(int i = 0; i < sub.size(); i++)
    {
        int s = sub[i]; 
        bool is_found = false;
        for(int j = 0; j < full.size(); j++)
        {
            if(s == full[j]) {is_found=true; break;}
        }
        if(!is_found) return false; 
    }
    return true; 
}


bool in_list(const int& query, const std::vector<int>& list)
{
    for(int i = 0; i < list.size(); i++) 
    {
        if(list[i] == query) return true;
    }
    return false; 
}

// check if a and b has same element. 
// if yes, return false 
// else return true 
bool my_exclude(std::vector<int> a, std::vector<int> b)
{
    bool exclude = true; 
    if(a.size() == 0 || b.size() == 0) return true; 
    for(int i = 0; i < a.size(); i++)
    {
        if(in_list(a[i], b))
        {
            exclude = false; 
            break;
        }
    }
    return exclude; 
}

bool in_box_test(Eigen::Vector2d x, Eigen::Vector4d box)
{
    if(
        x(0) >= box(0) && x(0) <= box(2) && x(1) >= box(1) && x(1) <= box(3)
    )
    return true; 
    else return false; 
}

double welsch(double x, double c)
{
    double y = 1 - exp(-0.5 * x * x / c /c); 
}