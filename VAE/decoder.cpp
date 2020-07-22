#include "decoder.h"
#include <iostream> 

Decoder::Decoder()
{
	p_dec_fc1 = std::make_shared<LinearLayer>(32, 512); 
	p_dec_fc2 = std::make_shared<LinearLayer>(512, 512); 
	p_dec_out = std::make_shared<LinearLayer>(512, 62 * 9);
	p_dec_tanh = std::make_shared<Tanh>(); 
	p_dec_lrelu1 = std::make_shared<LeakyReLU>(0.2); 
	p_dec_lrelu2 = std::make_shared<LeakyReLU>(0.2); 

	std::string folder = "F:/projects/model_preprocess/designed_pig/pig_prior/params_txt2/";
	p_dec_fc1->load_params(folder + "bodyprior_dec_fc1.weight.txt",
		folder+"bodyprior_dec_fc1.bias.txt");
	p_dec_fc2->load_params(folder + "bodyprior_dec_fc2.weight.txt",
		folder + "bodyprior_dec_fc2.bias.txt");
	p_dec_out->load_params(folder + "bodyprior_dec_out.weight.txt",
		folder + "bodyprior_dec_out.bias.txt");

}

Decoder::~Decoder()
{

}

void Decoder::forward()
{
	p_dec_fc1->input = latent; 
	std::cout << "p_dec_fc1.forward" << std::endl; 
	p_dec_fc1->forward(); 
	p_dec_lrelu1->input = p_dec_fc1->output;
	std::cout << "p_dec_lrelu1.forward" << std::endl;

	p_dec_lrelu1->forward(); 
	p_dec_fc2->input = p_dec_lrelu1->output;
	lrelu1_out = p_dec_lrelu1->output; 
	std::cout << "p_dec_fc2.forward" << std::endl;

	p_dec_fc2->forward(); 
	p_dec_lrelu2->input = p_dec_fc2->output;
	std::cout << "p_dec_lrelu2.forward" << std::endl;

	p_dec_lrelu2->forward();
	lrelu2_out = p_dec_lrelu2->output; 
	p_dec_out->input = p_dec_lrelu2->output;
	std::cout << "p_dec_out.forward" << std::endl;

	p_dec_out->forward();
	p_dec_tanh->input = p_dec_out->output; 
	std::cout << "p_dec_tanh.forward" << std::endl;

	p_dec_tanh->forward(); 
	output = p_dec_tanh->output; 
}

void Decoder::backward()
{
	Eigen::VectorXd diff = gt - output; 
}