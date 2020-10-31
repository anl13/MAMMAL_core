#include "math_utils.h"
#include <algorithm>


//*********************homogeneous coordinate***************
Eigen::Vector4f ToHomogeneous(const Eigen::Vector3f &_v)
{
    Eigen::Vector4f v2;
    v2.x() = _v.x();
    v2.y() = _v.y();
    v2.z() = _v.z();
    v2.w() = 1.0f;

    return v2;
}

Eigen::Vector3f ToHomogeneous(const Eigen::Vector2f &_v)
{
    Eigen::Vector3f v2;
    v2.x() = _v.x();
    v2.y() = _v.y();
    v2.z() = 1.0;
    return v2;
}

Eigen::Vector3f FromHomogeneous(const Eigen::Vector4f &_v)
{
    Eigen::Vector3f v2;
    v2 = _v.block<3,1>(0,0);
    if(_v(3) < 1e-8) return v2; 
    else return v2 / _v(3);
}

/*****************rotation convert********************/

Eigen::Vector4f AxisAngleToQuat(const Eigen::Vector3f &v)
{
    float angle = v.norm(); 
    if(angle == 0) return Eigen::Vector4f::Zero(); 
    Eigen::Vector3f n = v/angle; 
    Eigen::Vector4f quat; 
    quat.block<3,1>(0,0) = n * sinf(angle/2); 
    quat[3] = cosf(angle/2); 

    return quat; 
}

Eigen::Matrix3f GetSkewMatrix(const Eigen::Vector3f& w)
{
	Eigen::Matrix3f expw = Eigen::Matrix3f::Zero();
	expw(0, 1) = -w(2);
	expw(0, 2) = w(1);
	expw(1, 0) = w(2);
	expw(1, 2) = -w(0);
	expw(2, 0) = -w(1);
	expw(2, 1) = w(0);
	return expw;
}

Eigen::Matrix3f GetRodrigues(const Eigen::Vector3f& w)
{
	cv::Mat cv_w(3, 1, CV_32F);
	cv::eigen2cv(w, cv_w);

	cv::Mat cv_rod(3, 3, CV_32F);
	cv::Rodrigues(cv_w, cv_rod);

	Eigen::Matrix3f eigen_rod;
	cv::cv2eigen(cv_rod, eigen_rod);
	return eigen_rod;
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

////http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872_2007014421.pdf
//// 
//Eigen::Vector4f AvgQuats(const std::vector<Eigen::Vector4f> &quats)
//{
//    Eigen::Matrix4f M = Eigen::Matrix4f::Zero(); 
//    for(int i=0;i<quats.size();i++)
//    {
//        M = M + quats[i] * quats[i].transpose(); 
//    }
//
//    Eigen::JacobiSVD<Eigen::Matrix4f> svd(M, 
//        Eigen::ComputeFullU | Eigen::ComputeFullV | Eigen::ColPivHouseholderQRPreconditioner); 
//    
//    auto V = svd.matrixV(); 
//    auto q_avg = V.col(0); // eigen vector corresponding to the largest eigen value 
//
//    return q_avg; 
//}

/*********************distance computation************************/
float L2LDist(const Eigen::Vector3f& x1, const Eigen::Vector3f& x2, const Eigen::Vector3f& x3, const Eigen::Vector3f& x4)
{
    Eigen::Vector3f a = x2 - x1; 
    Eigen::Vector3f b = x4 - x3;
    Eigen::Vector3f c = x3 - x1;
    float d = c.dot(a.cross(b)) / (a.cross(b)).norm();
    return fabs(d); 
}

float L2LDist(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c)
{
    float d = c.dot(a.cross(b)) / (a.cross(b)).norm();
    return fabs(d); 
}

float p2ldist( Eigen::Vector3f x,  Eigen::Vector3f line){
    x.normalize(); 
    line.normalize(); 
    return (x.cross(line)).norm(); 
}

float p2ldist( Eigen::Vector3f x,  Eigen::Vector3f a,  Eigen::Vector3f b)
{
    Eigen::Vector3f line = a-b; 
    Eigen::Vector3f x_line = x-b;
    return p2ldist(x_line, line); 
}

/******************* vector comparison ******************/

bool my_equal(std::vector<int> a, std::vector<int> b)
{
    if(a.size() != b.size()) return false;
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

int find_in_list(const int&query, const std::vector<int>& list)
{
	for (int i = 0; i < list.size(); i++)
	{
		if (list[i] == query) return i;
	}
	return -1; 
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

bool in_box_test(const Eigen::Vector2f& x, const Eigen::Vector4f& box)
{
    if(
        x(0) >= box(0) && x(0) <= box(2) && x(1) >= box(1) && x(1) <= box(3)
    )
    return true; 
    else return false; 
}

bool in_box_test(const Eigen::Vector2i& x, const Eigen::Vector4i& box)
{
	if (
		x(0) >= box(0) && x(0) < box(2) && x(1) >= box(1) && x(1) < box(3)
		)
		return true;
	else return false;
}


float welsch(float x, float c)
{
    float y = 1 - expf(-0.5 * x * x / c /c); 
}

float IoU_xyxy(Eigen::Vector4f b1, Eigen::Vector4f b2)
{
	if (b1.norm() == 0 || b2.norm() == 0) return 0; 
    float xA = my_max(b1(0), b2(0));
    float yA = my_max(b1(1), b2(1)); 
    float xB = my_min(b1(2), b2(2)); 
    float yB = my_min(b1(3), b2(3)); 
    float inter = my_max(0, xB - xA +1) * my_max(0, yB - yA + 1); 
    float areaA = (b1(2) - b1(0) + 1) * (b1(3) - b1(1) + 1); 
    float areaB = (b2(2) - b2(0) + 1) * (b2(3) - b2(1) + 1); 
    float iou = inter / (areaA + areaB - inter);
    return iou; 
}

void IoU_xyxy_ratio(Eigen::Vector4f b1, Eigen::Vector4f b2, float& iou, float &iou2b1, float &iou2b2)
{
    float xA = my_max(b1(0), b2(0));
    float yA = my_max(b1(1), b2(1)); 
    float xB = my_min(b1(2), b2(2)); 
    float yB = my_min(b1(3), b2(3)); 
    float inter = my_max(0, xB - xA +1) * my_max(0, yB - yA + 1); 
    float areaA = (b1(2) - b1(0) + 1) * (b1(3) - b1(1) + 1); 
    float areaB = (b2(2) - b2(0) + 1) * (b2(3) - b2(1) + 1); 
    iou = inter / (areaA + areaB - inter);
    iou2b1 = inter / areaA; 
    iou2b2 = inter / areaB;
}

bool in_image(float w, float h, float x, float y)
{
    return (x>=0 && x<w && y>=0 && y<h); 
}



//Q.x∗R.y+P.x∗Q.y+P.y∗R.x−P.x∗R.y−Q.y∗R.x−P.y∗Q.x
bool to_left_test(const Eigen::Vector3f& p, const Eigen::Vector3f& q, const Eigen::Vector3f& r)
{
	float v = q(0) * r(1) + p(0) * q(1) + p(1) * r(0)
		- p(0) * r(1) - q(1) * r(0) - p(1) * q(0);
	if (v > 0) return true; 
	return false; 
}

float vec2angle(const Eigen::Vector2f& vec)
{
	float angleInRadians = std::atan2(vec(1), vec(0));
	float angleInDegrees = (angleInRadians / M_PI) * 180.0;
	return angleInDegrees;
	// -180 ~ 180
}

Eigen::Matrix4f Twist(const Eigen::Vector6f &_twist)
{
	// calculate exponential mapping from Lie Algebra (se(3)) to Lie Group (SE(3))
	Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
	Eigen::Vector3f axis = _twist.head(3);

	if (axis.cwiseAbs().sum() > 1e-5f) {
		float angle = axis.norm();
		axis.normalize();

		// rotation
		T.topLeftCorner(3, 3) = Eigen::AngleAxisf(angle, axis).matrix();

		// translation
		Eigen::Vector3f rho(_twist.tail(3));
		const float s = std::sin(angle) / angle;
		const float t = (1 - std::cos(angle)) / angle;

		Eigen::Matrix3f skew = GetSkewMatrix(axis);
		Eigen::Matrix3f J = s * Eigen::Matrix3f::Identity() + (1 - s) * (skew * skew + Eigen::Matrix3f::Identity()) + t * skew;
		Eigen::Vector3f trans = J * rho;
		T.topRightCorner(3, 1) = trans;
	}
	return T;
}

Eigen::Matrix4f LookAt(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _target, const Eigen::Vector3f& _up)
{
	const Eigen::Vector3f direct = (_pos - _target).normalized();
	const Eigen::Vector3f right = (_up.cross(direct)).normalized();
	const Eigen::Vector3f up = (direct.cross(right)).normalized();

	Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
	Eigen::Matrix3f R = mat.block<3, 3>(0, 0);
	R.row(0) = right.transpose();
	R.row(1) = up.transpose();
	R.row(2) = direct.transpose();
	mat.block<3, 3>(0, 0) = R;
	mat.block<3, 1>(0, 3) = R * (-_pos);

	return mat;
}

Eigen::Matrix4f Transform(const Eigen::Vector3f& _translation, const Eigen::Vector3f& _rotation, const float _scale)
{
	Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
	mat.block<3, 3>(0, 0) = _scale * GetRodrigues(_rotation);
	mat.block<3, 1>(0, 3) = _translation;
	return mat;
}

Eigen::Matrix4f Perspective(const float fovy, const float aspect, const float zNear, const float zFar)
{
	Eigen::Matrix4f mat = Eigen::Matrix4f::Zero();
	float tangent = tanf(0.5f * fovy);
	mat(0, 0) = 1.0f / (tangent*aspect);
	mat(1, 1) = 1.0f / tangent;
	mat(2, 2) = (zNear + zFar) / (zNear - zFar);
	mat(3, 2) = -1.0f;
	mat(2, 3) = 2.0f*zFar*zNear / (zNear - zFar);
	return mat;
}

Eigen::Matrix4f calcRenderExt(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _up, const Eigen::Vector3f& _center)
{
	Eigen::Vector3f pos = _pos;
	Eigen::Vector3f up = _up;
	Eigen::Vector3f center = _center;

	Eigen::Matrix4f viewMat = LookAt(pos, center, up);
	return viewMat; 
}

Eigen::Matrix4f calcRenderExt(const Eigen::Matrix3f& R, const Eigen::Vector3f& T)
{
	//Eigen::Vector3f front = -R.row(2).transpose();
	//Eigen::Vector3f up = -R.row(1).transpose();
	//Eigen::Vector3f pos = -R.transpose() * T;
	//Eigen::Vector3f center = pos - 1.0f*front;

	////std::cout << "R: " << R << std::endl << "T: " << T.transpose() << std::endl; 
	////std::cout << calcRenderExt(pos, up, center) << std::endl; 
	//return calcRenderExt(pos, up, center);
	Eigen::Matrix4f Transform = Eigen::Matrix4f::Identity(); 
	Transform.block<3, 3>(0, 0) = R; 
	Transform.block<3, 1>(0, 3) = T; 
	Transform.row(1) *= -1; 
	Transform.row(2) *= -1;
	return Transform;
}

Eigen::Vector3f Mat2Rotvec(Eigen::Matrix3f mat)
{
	cv::Mat cvVec;
	cv::Mat cvRodriguesMat;
	cv::eigen2cv(mat, cvRodriguesMat);

	cv::Rodrigues(cvRodriguesMat, cvVec);
	Eigen::Vector3f rotvec; 
	cv::cv2eigen(cvVec, rotvec); 
	return rotvec; 
}


Eigen::Matrix3f EulerToRotRad(float z, float y, float x, std::string type)
{
	if (type == "ZYX") // matlab default type ZYX
	{
		float cx = cos(x);
		float sx = sin(x);
		float cy = cos(y);
		float sy = sin(y);
		float cz = cos(z);
		float sz = sin(z);
		Eigen::Matrix3f Rx = Eigen::Matrix3f::Identity();
		Rx(1, 1) = Rx(2, 2) = cx;
		Rx(1, 2) = -sx; Rx(2, 1) = sx;
		Eigen::Matrix3f Ry = Eigen::Matrix3f::Identity();
		Ry(0, 0) = Ry(2, 2) = cy;
		Ry(0, 2) = sy; Ry(2, 0) = -sy;
		Eigen::Matrix3f Rz = Eigen::Matrix3f::Identity();
		Rz(0, 0) = Rz(1, 1) = cz;
		Rz(0, 1) = -sz; Rz(1, 0) = sz;

		return Rz * Ry * Rx; 
	}
	else {
		std::cout << "euler type " << type << " not implemented yet." << std::endl;
		return Eigen::Matrix3f::Identity();
	}
}

Eigen::Matrix3f EulerToRotDegree(float x, float y, float z, std::string type)
{
	float xrad = x * 3.14159265359 / 180;
	float yrad = y * 3.14159265359 / 180;
	float zrad = z * 3.14159265359 / 180;
	return EulerToRotRad(xrad, yrad, zrad, type);
}

Eigen::Matrix3f EulerToRotRad(Eigen::Vector3f rads, std::string type)
{
	return EulerToRotRad(rads(0), rads(1), rads(2), type);
}

Eigen::Matrix3f EulerToRotDegree(Eigen::Vector3f rads, std::string type)
{
	return EulerToRotDegree(rads(0), rads(1), rads(2), type);
}

Eigen::Vector3f Mat2Euler(Eigen::Matrix3f R)
{
	Eigen::Vector3f euler; 
	if (R(0, 0) == 0 && R(1, 0) == 0)
	{
		euler(0) = 0; 
		euler(1) = M_PI / 2; 
		euler(2) = atan(R(0, 1) / R(1, 1));
	}
	else
	{
		euler(1) = atan(-R(2, 0) / sqrtf(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0)));
		euler(0) = atan(R(1, 0) / R(0, 0)); 
		euler(2) = atan(R(2, 1) / R(2, 2)); 
	}
	return euler; 
}

Eigen::Matrix<float, 3, 9, Eigen::ColMajor> EulerJacobiF(const Eigen::Vector3f& euler)
{
	float x = euler(2); float y = euler(1); float z = euler(0); 
	float cx = cos(x);
	float sx = sin(x);
	float cy = cos(y);
	float sy = sin(y);
	float cz = cos(z);
	float sz = sin(z);
	Eigen::Matrix3f Rx = Eigen::Matrix3f::Identity();
	Rx(1, 1) = Rx(2, 2) = cx;
	Rx(1, 2) = -sx; Rx(2, 1) = sx;
	Eigen::Matrix3f Ry = Eigen::Matrix3f::Identity();
	Ry(0, 0) = Ry(2, 2) = cy;
	Ry(0, 2) = sy; Ry(2, 0) = -sy;
	Eigen::Matrix3f Rz = Eigen::Matrix3f::Identity();
	Rz(0, 0) = Rz(1, 1) = cz;
	Rz(0, 1) = -sz; Rz(1, 0) = sz;

	Eigen::Matrix3f dRx = Eigen::Matrix3f::Zero(); 
	Eigen::Matrix3f dRy = Eigen::Matrix3f::Zero();
	Eigen::Matrix3f dRz = Eigen::Matrix3f::Zero(); 

	dRx(1, 1) = dRx(2, 2) = -sx;
	dRx(1, 2) = -cx;
	dRx(2, 1) = cx;

	dRy(0, 0) = dRy(2, 2) = -sy;
	dRy(0, 2) = cy;
	dRy(2, 0) = -cy;

	dRz(0, 0) = dRz(1, 1) = -sz;
	dRz(0, 1) = -cz;
	dRz(1, 0) = cz;

	Eigen::Matrix<float, 3, 9, Eigen::ColMajor> dR; 
	dR.middleCols(0, 3) = dRz * Ry * Rx;
	dR.middleCols(3, 3) = Rz * dRy * Rx; 
	dR.middleCols(6, 3) = Rz * Ry * dRx;
	return dR; 
}

Eigen::Matrix<float, 3, 9, Eigen::ColMajor> EulerJacobiFNumeric(const Eigen::Vector3f& euler)
{
	Eigen::Matrix<float, 3, 9, Eigen::ColMajor> dR; 
	float alpha = 0.001; 
	float invalpha = 1000;
	Eigen::Matrix3f R0 = EulerToRotRad(euler);
	
	for (int i = 0; i < 3; i++)
	{
		Eigen::Vector3f delta = euler; 
		delta(i) += alpha; 
		Eigen::Matrix3f Delta = EulerToRotRad(delta); 
		Eigen::Vector3f delta1 = euler; 
		delta1(i) -= alpha; 
		Eigen::Matrix3f Delta1 = EulerToRotRad(delta1); 
		Eigen::Matrix3f D = (Delta - Delta1) / 2 * invalpha;
		dR.middleCols(3 * i, 3) = D; 
	}
	return dR; 
}

std::vector<Eigen::Vector3f> doubleToFloat(const std::vector<Eigen::Vector3d>& list)
{
	std::vector<Eigen::Vector3f> listf;
	listf.resize(list.size()); 
	for (int i = 0; i < list.size(); i++)listf[i] = list[i].cast<float>(); 
	return listf;
}