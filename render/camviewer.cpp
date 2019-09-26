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

	projMat = EigenUtil::LookAt(pos, center, up);
}


void CamViewer::SetIntrinsic(const float fovy, const float aspect, const float zNear, const float zFar)
{
	perspectiveMat = EigenUtil::Perspective(fovy, aspect, zNear, zFar);
}


void CamViewer::SetIntrinsic(const Eigen::Matrix3f& K, const int photoSize)
{
	const float fx = K(0, 0);
	const float fy = K(1, 1);
	const float cx = K(0, 2);
	const float cy = K(1, 2);

	perspectiveMat <<
		2.0f * fx / float(photoSize), 0.0f, 2.0f*cx / float(photoSize) - 1.0f, 0.0f,
		0.0f, 2.0f * fy / float(photoSize), 2.0f*cy / float(photoSize) - 1.0f, 0.0f,
		0.0f, 0.0f, (RENDER_NEAR_PLANE + RENDER_FAR_PLANE) / (RENDER_NEAR_PLANE - RENDER_FAR_PLANE), 2.0f*RENDER_FAR_PLANE*RENDER_NEAR_PLANE / (RENDER_NEAR_PLANE - RENDER_FAR_PLANE),
		0.0f, 0.0f, -1.0f, 0.0f;
}


void CamViewer::SetExtrinsic(const Eigen::Matrix3f& R, const Eigen::Vector3f& T)
{
	front = -R.row(2).transpose();
	up = -R.row(1).transpose();

	pos = -R.transpose() * T;
	center = pos - 1.0f*front;

	SetExtrinsic(pos, up, center);
}


void CamViewer::ConfigShader(Shader& shader) const
{
	shader.SetMat4("proj", projMat);
	shader.SetMat4("perspective", perspectiveMat);
	shader.SetVec3("view_pos", pos);
}

