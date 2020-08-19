#pragma once

#include <vector>
#include <Eigen/Eigen>

class GMM
{
public: 
	GMM() {}
	~GMM() {}

	int M; // modal number 
	std::vector<Eigen::VectorXd> mu; // mean of each modal 
	std::vector<Eigen::MatrixXd> sigma;  // covariance of each modal 
};
