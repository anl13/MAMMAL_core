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

	void forward(); 
	void computeJacobi();

	Eigen::MatrixXd J; 

	Eigen::VectorXd end_grad; // gradient on output, to backpropagate
	Eigen::MatrixXd grad;     // grad = J.transpose() * end_grad 
	//Eigen::MatrixXd outrotmats; // [3, 3*62]

};