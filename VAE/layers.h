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

	Eigen::MatrixXd W;
	Eigen::VectorXd b;
	Eigen::MatrixXd J;

	Eigen::VectorXd input;
	Eigen::VectorXd output;

	void forward();
	void backward();

	int m_dimin;
	int m_dimout;
};

class LeakyReLU
{
public:
	LeakyReLU(double r)
	{
		m_r = r; 
	}

	Eigen::VectorXd input;
	Eigen::VectorXd output;
	Eigen::MatrixXd J;

	void forward();
	void backward();

	double m_r;
};


class Tanh
{
public:
	Tanh() {};

	Eigen::VectorXd input;
	Eigen::VectorXd output;
	Eigen::MatrixXd J;

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

	Eigen::VectorXd input;  // [62 * 6]
	Eigen::VectorXd output; // [62 * 9]
	Eigen::MatrixXd J; // [(62*9) * (62*6)] 

	void forward(); 
	void  backward(); 

	Eigen::VectorXd computey(Eigen::VectorXd x); 
	Eigen::MatrixXd computeJBlock(Eigen::VectorXd x); 

private:
	// y = ||x||_2, return dy/dx^T
	Eigen::Matrix3d computeJVecNorm(Eigen::Vector3d x);

	//c = a - (a.dot(b))b (assume b is normalized)
	// return dc/da^T, dc/db^T
	void computeJr2(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Matrix3d& Ja, Eigen::Matrix3d& Jb);

	// c = a.cross(b)
	// return dc/da^T, dc/db^T
	void computeJCross(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Matrix3d d_b_d_a, Eigen::Matrix3d& Ja, Eigen::Matrix3d& Jb);

};