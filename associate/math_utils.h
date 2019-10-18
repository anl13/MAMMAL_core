#ifndef __MATH_UTILS_HPP__
#define __MATH_UTILS_HPP__

#include <opencv2/opencv.hpp>
#include <Eigen/Dense> 
#include <math.h> 
#include <opencv2/core/eigen.hpp> 
#include <vector> 

typedef Eigen::Matrix4d Mat4; 
typedef Eigen::Matrix3d Mat3; 
typedef Eigen::Vector3d Vec3; 
typedef Eigen::Vector4d Vec4; 
typedef Eigen::Vector2d Vec2; 


typedef Eigen::aligned_allocator<Mat4>  Alloc_Mat4; 
typedef std::vector<Mat4, Alloc_Mat4>  ProjectionList; 

/*****************homogeneous coordinate************/ 
Vec3 ToHomogeneous(const Vec2& _v); 
Vec4 ToHomogeneous(const Vec3& _v);
Vec3 FromHomogeneous(const Vec4& _v); 
Mat4 ToHomogeneous(const Mat3 &_m, const Vec3 &_v);

/*****************rotation convert********************/ 
Vec4 AxisAngleToQuat(const Vec3 &v); 

Vec3 QuatToAxisAngle(const Vec4 &q);

// Get skew matrix of a norm vec3
Mat3 GetSkewMatrix(const Vec3& w);

// w: axis-angle vector 
// return 3x3 rotation matrix 
Mat3 GetRodrigues(const Vec3& w);

Eigen::Matrix<float, 3, 9, Eigen::ColMajor> RodriguesJacobiF(const Eigen::Vector3f& vec);
Eigen::Matrix<double, 3, 9, Eigen::ColMajor> RodriguesJacobiD(const Eigen::Vector3d& vec); 

/*****************average rotations*******************/
Vec3 AvgAxisAngles(const std::vector<Vec3> &rots); 

Vec4 AvgQuats(const std::vector<Vec4> &quats); 

/*******************double functions for ceres******************************/
// Eigen::Matrix3d CalcRodrigues(const Eigen::Vector3d &v); 

Eigen::Vector3d RodriguesToVec(const Eigen::Matrix3d &R); 
Eigen::Matrix3f GetRodriguesF(const Eigen::Vector3f &w); 

//refering to: http://mathworld.wolfram.com/Line-LineDistance.html
double L2LDist(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4); 
double L2LDist(const Vec3& a, const Vec3& b, const Vec3& c); 

double p2ldist(Vec3 x,  Vec3 line); 
double p2ldist(Vec3 x,  Vec3 a,  Vec3 b); // a,b form a line 

bool my_equal(std::vector<int> a, std::vector<int> b); 
bool my_contain(std::vector<int> full, std::vector<int> sub); 
bool in_list(const int& query, const std::vector<int>& list); 
bool my_exclude(std::vector<int> a, std::vector<int> b); 

#endif 