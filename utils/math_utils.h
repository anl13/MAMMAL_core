#pragma once 

#include <opencv2/opencv.hpp>
#include <Eigen/Dense> 
#include <math.h> 
#include <opencv2/core/eigen.hpp> 
#include <vector> 
#include <cmath> 
#include <type_traits>

#define PI 3.14159265359 


typedef Eigen::Matrix4d Mat4; 
typedef Eigen::Matrix3d Mat3; 
typedef Eigen::Vector3d Vec3; 
typedef Eigen::Vector4d Vec4; 
typedef Eigen::Vector2d Vec2; 
typedef Eigen::Matrix<unsigned int, -1, -1> MatXui;

typedef Eigen::aligned_allocator<Mat4>  Alloc_Mat4; 
typedef std::vector<Mat4, Alloc_Mat4>  ProjectionList; 

/*****************homogeneous coordinate************/ 
Vec3 ToHomogeneous(const Vec2& _v); 
Vec4 ToHomogeneous(const Vec3& _v);
Vec3 FromHomogeneous(const Vec4& _v); 
Mat4 ToHomogeneous(const Mat3 &_m, const Vec3 &_v);

/*****************rotation convert********************/ 
Vec4 AxisAngleToQuat(const Vec3 &v); 

//Vec3 QuatToAxisAngle(const Vec4 &q);

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

Eigen::Matrix3f GetRodriguesF(const Eigen::Vector3f &w); 

//refering to: http://mathworld.wolfram.com/Line-LineDistance.html
double L2LDist(const Vec3& x1, const Vec3& x2, const Vec3& x3, const Vec3& x4); 
double L2LDist(const Vec3& a, const Vec3& b, const Vec3& c); 

double p2ldist(Vec3 x,  Vec3 line); 
double p2ldist(Vec3 x,  Vec3 a,  Vec3 b); // a,b form a line 

bool my_equal(std::vector<int> a, std::vector<int> b); 
bool my_contain(std::vector<int> full, std::vector<int> sub); 
bool in_list(const int& query, const std::vector<int>& list); 
int find_in_list(const int &query, const std::vector<int>& list);
bool my_exclude(std::vector<int> a, std::vector<int> b); 
inline double my_min(double a, double b){ return a>b?b:a; }
inline double my_max(double a, double b){ return a>b?a:b; }

// box: (x1,y1,x2,y2)
bool in_box_test(const Eigen::Vector2d& x, const Eigen::Vector4d& box); 
bool in_box_test(const Eigen::Vector2i& x, const Eigen::Vector4i& box);
double welsch(double x, double c); 

double IoU_xyxy(Eigen::Vector4d b1, Eigen::Vector4d b2); 
void IoU_xyxy_ratio(Eigen::Vector4d b1, Eigen::Vector4d b2, double& iou, double &iou2b1, double &iou2b2); 

bool in_image(float w, float h, float x, float y); 


// XYZ: roll pitch yaw convention. 
Mat3 EulerToRotRadD(double x, double y, double z, std::string type="XYZ");
Mat3 EulerToRotDegreeD(double x, double y, double z, std::string type="XYZ");
Mat3 EulerToRotRadD(Vec3 rads, std::string type="XYZ"); 
Mat3 EulerToRotDegreeD(Vec3 rads, std::string type="XYZ");


bool to_left_test(const Eigen::Vector3d& x, const Eigen::Vector3d& y, const Eigen::Vector3d& z); 

double vec2angle(const Eigen::Vector2d& vec); 

// Yuxiang Zhang's math util, added at 2020/Mar/28

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
	typedef Eigen::Matrix<double, 6, 1> Vector6d;
	typedef Eigen::Matrix<double, 3, 4> Matrix34d;
	typedef Eigen::Matrix<double, 3, 2> Matrix32d;
}

namespace MathUtil {
	// Linear Algebra
	inline Eigen::Matrix3d Skew(const Eigen::Vector3d& vec)
	{
		Eigen::Matrix3d skew;
		skew << 0, -vec.z(), vec.y(),
			vec.z(), 0, -vec.x(),
			-vec.y(), vec.x(), 0;
		return skew;
	}


	inline Eigen::Matrix4d Twist(const Eigen::Vector6d &_twist)
	{
		// calculate exponential mapping from Lie Algebra (se(3)) to Lie Group (SE(3))
		Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
		Eigen::Vector3d axis = _twist.head(3);

		if (axis.cwiseAbs().sum() > 1e-5f) {
			double angle = axis.norm();
			axis.normalize();

			// rotation
			T.topLeftCorner(3, 3) = Eigen::AngleAxisd(angle, axis).matrix();

			// translation
			Eigen::Vector3d rho(_twist.tail(3));
			const double s = std::sin(angle) / angle;
			const double t = (1 - std::cos(angle)) / angle;

			Eigen::Matrix3d skew = Skew(axis);
			Eigen::Matrix3d J = s * Eigen::Matrix3d::Identity() + (1 - s) * (skew * skew + Eigen::Matrix3d::Identity()) + t * skew;
			Eigen::Vector3d trans = J * rho;
			T.topRightCorner(3, 1) = trans;
		}
		return T;
	}
}