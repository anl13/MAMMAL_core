#pragma once

#include <vector>
#include <Eigen/Eigen>

class GMM
{
public: 
	GMM() {}
	~GMM() {}

	void Load(); 
	void CalcGMMTerm(const Eigen::VectorXf& pose, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb); 
	void CalcAnchorTerm(const Eigen::VectorXf& pose, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, int type);


	int M; // modal number 
	int dim;
	std::vector<Eigen::VectorXf> mu; // mean of each modal 
	std::vector<Eigen::MatrixXf> sigma;  // covariance of each modal 
};
