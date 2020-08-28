#pragma once

#include <Eigen/Eigen>
#include <string>
class LinearLayer
{
public:
	LinearLayer(int dimin, int dimout) {
		m_dimin = dimin; 
		m_dimout = dimout; 
	}

	void load_params(std::string filew, std::string fileb); 

	Eigen::MatrixXf W;
	Eigen::VectorXf b;
	Eigen::MatrixXf J;

	Eigen::VectorXf input;
	Eigen::VectorXf output;

	void forward();
	void backward();

	int m_dimin;
	int m_dimout;
};

class LeakyReLU
{
public:
	LeakyReLU(float r)
	{
		m_r = r; 
	}

	Eigen::VectorXf input;
	Eigen::VectorXf output;
	Eigen::MatrixXf J;

	void forward();
	void backward();

	float m_r;
};


class Tanh
{
public:
	Tanh() {};

	Eigen::VectorXf input;
	Eigen::VectorXf output;
	Eigen::MatrixXf J;

	void forward();
	void backward();
};


class ContinousRotation
{
public: 
	ContinousRotation() {
		jointnum = 62;
	}

	int jointnum; 

	Eigen::VectorXf input;  // [62 * 6]
	Eigen::VectorXf output; // [62 * 9]
	Eigen::MatrixXf J; // [(62*9) * (62*6)] 

	void forward(); 
	void  backward(); 

	Eigen::VectorXf computey(Eigen::VectorXf x); 
	Eigen::MatrixXf computeJBlock(Eigen::VectorXf x); 

private:
	// y = ||x||_2, return dy/dx^T
	Eigen::Matrix3f computeJVecNorm(Eigen::Vector3f x);

	//c = a - (a.dot(b))b (assume b is normalized)
	// return dc/da^T, dc/db^T
	void computeJr2(Eigen::Vector3f a, Eigen::Vector3f b, Eigen::Matrix3f& Ja, Eigen::Matrix3f& Jb);

	// c = a.cross(b)
	// return dc/da^T, dc/db^T
	void computeJCross(Eigen::Vector3f a, Eigen::Vector3f b, Eigen::Matrix3f d_b_d_a, Eigen::Matrix3f& Ja, Eigen::Matrix3f& Jb);

};