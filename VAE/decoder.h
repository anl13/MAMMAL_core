#pragma once
#include <Eigen/Eigen>
#include <memory>
#include "layers.h"

class Decoder
{
public: 
	Decoder();
	~Decoder(); 

	std::shared_ptr<LinearLayer> p_dec_fc1; 
	std::shared_ptr<LinearLayer> p_dec_fc2;
	std::shared_ptr<LinearLayer> p_dec_out; 
	std::shared_ptr<Tanh> p_dec_tanh; 
	std::shared_ptr<LeakyReLU> p_dec_lrelu1;
	std::shared_ptr<LeakyReLU> p_dec_lrelu2; 

	Eigen::VectorXd latent; // [32] latent code 
	Eigen::VectorXd output; // [62*9] output rotation matrix in vector 
	Eigen::VectorXd gt;     // [62*9] ground truth 

	Eigen::VectorXd lrelu1_out; 
	Eigen::VectorXd lrelu2_out; 

	void forward(); 
	void backward();

	Eigen::MatrixXd grad;
	Eigen::MatrixXd J; 
	Eigen::MatrixXd outrotmats; // [3, 3*62]

};