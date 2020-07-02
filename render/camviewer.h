#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <Eigen/Eigen>

#include "shader.h"

#define RENDER_NEAR_PLANE 0.01f
#define RENDER_FAR_PLANE 50.0f

class CamViewer
{
public:
	CamViewer();
	~CamViewer();

	const Eigen::Vector3f& GetPos() const { return pos; }
	const Eigen::Vector3f& GetFront() const { return front; }
	const Eigen::Vector3f& GetUp() const { return up; }
	const Eigen::Vector3f& GetRight() const { return right; }
	const Eigen::Vector3f& GetCenter() const { return center; }

	void SetExtrinsic(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _up, const Eigen::Vector3f& _center);
	void SetExtrinsic(const Eigen::Matrix3f& R, const Eigen::Vector3f& T);

	void SetIntrinsic(const float fovy, const float aspect, const float zNear, const float zFar);
	void SetIntrinsic(const Eigen::Matrix3f& K, const int width, const int height);

	void ConfigShader(SimpleShader& shader) const;

	void GetRT(Eigen::Matrix3f& R, Eigen::Vector3f& T);

private:
	Eigen::Vector3f pos;
	Eigen::Vector3f up;
	Eigen::Vector3f front;
	Eigen::Vector3f right;
	Eigen::Vector3f center;
	Eigen::Matrix3f R; 
	Eigen::Matrix3f K;
	Eigen::Vector3f T;

	Eigen::Matrix<float, 4, 4, Eigen::ColMajor> viewMat;
	Eigen::Matrix<float, 4, 4, Eigen::ColMajor> projectionMat;
};
