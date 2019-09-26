#pragma once

#include <Eigen/Eigen>
#include <cfloat> 
#include <opencv2/calib3d.hpp> 
#include <opencv2/core/eigen.hpp> 

namespace EigenUtil
{
	inline Eigen::Matrix3f Rodrigues(const Eigen::Vector3f& vec)
	{
		const float angle = vec.norm();
		const Eigen::Vector3f axis = angle < FLT_EPSILON ? Eigen::Vector3f(0.0f, 0.0f, 1.0f) : vec.normalized();
		return Eigen::AngleAxisf(angle, axis).matrix();
	}

	inline Eigen::Matrix<float, 3, 9, Eigen::ColMajor> RodriguesJacobi(const Eigen::Vector3f& vec)
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

	inline Eigen::Matrix4f LookAt(const Eigen::Vector3f& _pos, const Eigen::Vector3f& _center, const Eigen::Vector3f& _up)
	{
		const Eigen::Vector3f front = (_pos - _center).normalized();
		const Eigen::Vector3f right = (front.cross(_up)).normalized();
		const Eigen::Vector3f up = (right.cross(front)).normalized();

		Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
		Eigen::Matrix3f R = mat.block<3, 3>(0, 0);
		R.row(0) = -right.transpose();
		R.row(1) = up.transpose();
		R.row(2) = front.transpose();
		mat.block<3, 1>(0, 3) = R*(-_pos);

		return mat;
	}


	inline Eigen::Matrix4f Transform(const Eigen::Vector3f& _translation, const Eigen::Vector3f& _rotation, const float _scale)
	{
		Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
		mat.block<3, 3>(0, 0) = _scale * Rodrigues(_rotation);
		mat.block<3, 1>(0, 3) = _translation;
		return mat;
	}

	inline Eigen::Matrix4f Perspective(const float fovy, const float aspect, const float zNear, const float zFar)
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
};

