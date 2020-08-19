#pragma once
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen>

class GmmSolver
{
public:
	void Set(const int& _gmmNum, const int& _dimNum, const std::vector<Eigen::VectorXd>& _data);
	void Solve(const int iterTime);

private:
	int gmmNum;
	int dimNum;
	std::vector<Eigen::VectorXd> data;

	Eigen::MatrixXd gammas;

	Eigen::VectorXd weights;
	Eigen::VectorXd dets;
	std::vector<Eigen::VectorXd> means;
	std::vector<Eigen::MatrixXd> covs;
	std::vector<Eigen::MatrixXd> covInvs;

	void StepE();
	void StepM();
};