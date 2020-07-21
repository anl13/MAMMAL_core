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
	Eigen::VectorXd grad;

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

	Eigen::MatrixXd input;
	Eigen::MatrixXd output;
	Eigen::MatrixXd grad;

	void forward();
	void backward();

	double m_r;
};


class Tanh
{
public:
	Tanh() {};

	Eigen::MatrixXd input;
	Eigen::MatrixXd output;
	Eigen::MatrixXd grad;

	void forward();
	void backward();
};