#pragma once 

#include <opencv2/opencv.hpp>
#include <Eigen/Dense> 
#include <math.h> 
#include <opencv2/core/eigen.hpp> 
#include <vector> 
#include <cmath> 
#include <type_traits>
#include <algorithm> 

#ifndef M_PI
#define M_PI 3.1415926
#endif 

namespace Eigen {
	typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
	typedef Eigen::Matrix<unsigned char, 3, 3> Matrix3b;
	typedef Eigen::Matrix<unsigned char, 3, Eigen::Dynamic> Matrix3Xb;
	typedef Eigen::Matrix<unsigned char, 4, Eigen::Dynamic> Matrix4Xb;
	typedef Eigen::Matrix<unsigned char, 2, 1> Vector2b;
	typedef Eigen::Matrix<unsigned char, 3, 1> Vector3b;
	typedef Eigen::Matrix<unsigned char, 4, 1> Vector4b;
	typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> MatrixXu;
	typedef Eigen::Matrix<unsigned int, 3, 3> Matrix3u;
	typedef Eigen::Matrix<unsigned int, 3, Eigen::Dynamic> Matrix3Xu;
	typedef Eigen::Matrix<unsigned int, 4, Eigen::Dynamic> Matrix4Xu;
	typedef Eigen::Matrix<unsigned int, 2, 1> Vector2u;
	typedef Eigen::Matrix<unsigned int, 3, 1> Vector3u;
	typedef Eigen::Matrix<unsigned int, 4, 1> Vector4u;
	typedef Eigen::Matrix<float, 6, 1> Vector6f;
	typedef Eigen::Matrix<float, 3, 4> Matrix34f;
	typedef Eigen::Matrix<float, 3, 2> Matrix32f;
	typedef Eigen::Matrix<float, 6, 1> Vector6d;
	typedef Eigen::Matrix<float, 3, 4> Matrix34d;
	typedef Eigen::Matrix<float, 3, 2> Matrix32d;
	typedef Eigen::Matrix<float, 2, 3> Matrix23f;
}

/***************** homogeneous coordinate ************/ 
Eigen::Vector3f ToHomogeneous(const Eigen::Vector2f& _v); 
Eigen::Vector4f ToHomogeneous(const Eigen::Vector3f& _v);
Eigen::Vector3f FromHomogeneous(const Eigen::Vector4f& _v); 

/***************** rotation convert ********************/ 
Eigen::Vector4f AxisAngleToQuat(const Eigen::Vector3f &v); 
Eigen::Matrix3f GetSkewMatrix(const Eigen::Vector3f& w);
Eigen::Matrix3f GetRodrigues(const Eigen::Vector3f& w);
Eigen::Matrix<float, 3, 9, Eigen::ColMajor> RodriguesJacobiF(const Eigen::Vector3f& vec);

/*************** distance computation ****************/
//refering to: http://mathworld.wolfram.com/Line-LineDistance.html
float L2LDist(const Eigen::Vector3f& x1, const Eigen::Vector3f& x2, const Eigen::Vector3f& x3, const Eigen::Vector3f& x4); 
float L2LDist(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c); 
float p2ldist(Eigen::Vector3f x,  Eigen::Vector3f line); 
float p2ldist(Eigen::Vector3f x,  Eigen::Vector3f a,  Eigen::Vector3f b); // a,b form a line 

/******************* vector comparison ******************/
bool my_equal(std::vector<int> a, std::vector<int> b); 
bool my_contain(std::vector<int> full, std::vector<int> sub); 
bool in_list(const int& query, const std::vector<int>& list); 
int find_in_list(const int &query, const std::vector<int>& list);
bool my_exclude(std::vector<int> a, std::vector<int> b); 

inline float my_min(float a, float b){ return a>b?b:a; }
inline float my_max(float a, float b){ return a>b?a:b; }

// box: (x1,y1,x2,y2) [x1,y1]: top left point; [x2, y2]: right bottom point
bool in_box_test(const Eigen::Vector2f& x, const Eigen::Vector4f& box); 
bool in_box_test(const Eigen::Vector2i& x, const Eigen::Vector4i& box);
float welsch(float x, float c); 

float IoU_xyxy(Eigen::Vector4f b1, Eigen::Vector4f b2); 
void IoU_xyxy_ratio(Eigen::Vector4f b1, Eigen::Vector4f b2, float& iou, float &iou2b1, float &iou2b2); 

bool in_image(float w, float h, float x, float y); 

// XYZ: roll pitch yaw convention. 
Eigen::Matrix3f EulerToRotRad(float x, float y, float z, std::string type="XYZ");
Eigen::Matrix3f EulerToRotDegree(float x, float y, float z, std::string type="XYZ");
Eigen::Matrix3f EulerToRotRad(Eigen::Vector3f rads, std::string type="XYZ"); 
Eigen::Matrix3f EulerToRotDegree(Eigen::Vector3f rads, std::string type="XYZ");

// check Whether z is on the left of vector xy 
bool to_left_test(const Eigen::Vector3f& x, const Eigen::Vector3f& y, const Eigen::Vector3f& z); 

float vec2angle(const Eigen::Vector2f& vec); 
Eigen::Matrix4f Twist(const Eigen::Vector6f &_twist);

/***************** utils for rendering *****************/
Eigen::Matrix4f LookAt(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _target, const Eigen::Vector3f& _up);
Eigen::Matrix4f Transform(const Eigen::Vector3f& _translation, const Eigen::Vector3f& _rotation, const float _scale);
Eigen::Matrix4f Perspective(const float fovy, const float aspect, const float zNear, const float zFar);
Eigen::Matrix4f calcRenderExt(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _up, const Eigen::Vector3f& _center);
Eigen::Matrix4f calcRenderExt(const Eigen::Matrix3f& R, const Eigen::Vector3f& T);