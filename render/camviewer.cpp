#include <iostream>
#include <string>
#include <Eigen/Eigen>

#include "camviewer.h"
#include "eigen_util.h"

CamViewer::CamViewer()
{
	SetIntrinsic(0.5f*EIGEN_PI, 1.0f, RENDER_NEAR_PLANE, RENDER_FAR_PLANE);
	SetExtrinsic(Eigen::Vector3f::Ones(), Eigen::Vector3f(0.0f, 1.0f, 0.0f), Eigen::Vector3f::Zero());
}


CamViewer::~CamViewer() {}


void CamViewer::SetExtrinsic(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _up, const Eigen::Vector3f& _center)
{
	pos = _pos;
	up = _up;
	center = _center;

	front = (pos - center).normalized();
	right = (front.cross(up)).normalized();
	up = (right.cross(front)).normalized();

	viewMat = EigenUtil::LookAt(pos, center, up);
	// std::cout << "viewMat: " << std::endl << viewMat << std::endl; 
	// std::cout << "right:   " << right.transpose() << std::endl; 
	// std::cout << "front:   " << front.transpose() << std::endl; 
	// std::cout << "up   :   " << up.transpose() << std::endl; 

}


void CamViewer::SetIntrinsic(const float fovy, const float aspect, const float zNear, const float zFar)
{
	projectionMat = EigenUtil::Perspective(fovy, aspect, zNear, zFar);
}


void CamViewer::SetIntrinsic(const Eigen::Matrix3f& K, const int width, const int height)
{
	const float fx = K(0, 0);
	const float fy = K(1, 1);
	const float cx = K(0, 2);
	const float cy = K(1, 2);

	projectionMat <<
		2.0f * fx / float(width), 0.0f, 2.0f*cx / float(width) - 1.0f, 0.0f,
		0.0f, 2.0f * fy / float(height), 2.0f*cy / float(height) - 1.0f, 0.0f,
		0.0f, 0.0f, (RENDER_NEAR_PLANE + RENDER_FAR_PLANE) / (RENDER_NEAR_PLANE - RENDER_FAR_PLANE), 2.0f*RENDER_FAR_PLANE*RENDER_NEAR_PLANE / (RENDER_NEAR_PLANE - RENDER_FAR_PLANE),
		0.0f, 0.0f, -1.0f, 0.0f;
}

// ATTENTION: transform coordinate system to opencv image system
void CamViewer::SetExtrinsic(const Eigen::Matrix3f& _R, const Eigen::Vector3f& _T)
{
	R = _R;
	T = _T;
	front = -R.row(2).transpose();
	up = -R.row(1).transpose();
	pos = -R.transpose() * T;

	center = pos - 1.0f*front;
	// center = Eigen::Vector3f::Zero(); 

	SetExtrinsic(pos, up, center);
}


void CamViewer::ConfigShader(Shader& shader) const
{
	shader.SetMat4("view", viewMat);
	shader.SetMat4("projection", projectionMat);
	shader.SetVec3("view_pos", pos);
}

void CamViewer::GetRT(Eigen::Matrix3f& _R, Eigen::Vector3f& _T)
{
	_R = R;
	_T = T; 
}