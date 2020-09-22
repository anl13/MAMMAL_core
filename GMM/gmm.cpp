#include "gmm.h"
#include <string> 
#include <vector> 
#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <assert.h>
#include <Eigen/Eigen>

void GMM::Load()
{
	M = 9; 
	dim = 63; 
	mu.resize(M);
	sigma.resize(M); 

	std::string folder = "D:/Projects/animal_calib/data/artist_model/gmm/";
	for (int i = 0; i < M; i++)
	{
		std::stringstream ss;
		ss << folder << "prior_mean" << i << ".txt"; 
		std::ifstream file(ss.str());
		mu[i].resize(dim); 
		for (int k = 0; k < dim; k++) file >> mu[i](k);
		file.close(); 

		std::stringstream ss_cov; 
		ss_cov << folder << "prior_cov_inv" << i << ".txt"; 
		std::ifstream file_cov(ss_cov.str()); 
		sigma[i].resize(dim, dim);
		for (int k = 0; k < dim; k++)
		{
			for (int j = 0; j < dim; j++)
			{
				file_cov >> sigma[i](k, j);
			}
		}
		file_cov.close(); 

	}
}

void GMM::CalcGMMTerm(const Eigen::VectorXf& theta, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb)
{
	int paramNum = 69; 
	assert(theta.rows() == paramNum); 
	// find max
	float max_loss = 10000; 
	float max_id = -1; 
	for (int i = 0; i < M; i++)
	{
		float dist = (theta.tail(dim) - mu[i]).norm();
		std::cout << "dist: " << dist << std::endl;
		if (dist < max_loss)
		{
			max_loss = dist; 
			max_id = i; 
		}
	}
	assert(max_id >= 0);
	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum); 
	ATb = Eigen::VectorXf::Zero(paramNum); 
	//ATA.block<6, 6>(0, 0) = Eigen::MatrixXf::Identity(6, 6);
	//ATA.bottomRightCorner<63, 63>() = sigma[max_id];
	//ATb.segment<63>(6) = -M * (theta.tail(dim) - mu[max_id]);
	ATA = Eigen::MatrixXf::Identity(paramNum, paramNum); 
	ATb.segment<63>(6) = mu[max_id] - theta.segment<63>(6);
	std::cout << "gmm mu, theta" << std::endl;
	for (int i = 0; i < 63; i++)
	{
		std::cout << mu[max_id](6 + i) << ", " << theta(6 + i) << std::endl;
	}
}

void GMM::CalcAnchorTerm(const Eigen::VectorXf& theta, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, int type)
{
	int paramNum = 69;
	assert(theta.rows() == paramNum);

	ATA = Eigen::MatrixXf::Zero(paramNum, paramNum);
	ATb = Eigen::VectorXf::Zero(paramNum);
	//ATA.block<6, 6>(0, 0) = Eigen::MatrixXf::Identity(6, 6);
	//ATA.bottomRightCorner<63, 63>() = sigma[max_id];
	//ATb.segment<63>(6) = -M * (theta.tail(dim) - mu[max_id]);
	ATA = Eigen::MatrixXf::Identity(paramNum, paramNum);
	ATb.segment<63>(6) = mu[type] - theta.segment<63>(6);
}