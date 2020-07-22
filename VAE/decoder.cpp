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
	p_dec_fc1->forward(); 
	p_dec_lrelu1->input = p_dec_fc1->output;

	p_dec_lrelu1->forward(); 
	p_dec_fc2->input = p_dec_lrelu1->output;

	p_dec_fc2->forward(); 
	p_dec_lrelu2->input = p_dec_fc2->output;

	p_dec_lrelu2->forward();
	p_dec_out->input = p_dec_lrelu2->output;

	p_dec_out->forward();
	p_dec_tanh->input = p_dec_out->output; 

	p_dec_tanh->forward(); 
	output = p_dec_tanh->output; 
}

void Decoder::computeJacobi()
{
	p_dec_tanh->backward();
	p_dec_out->backward();
	p_dec_lrelu2->backward(); 
	p_dec_fc2->backward();
	p_dec_lrelu1->backward();
	p_dec_fc1->backward();

	//std::cout << "tanh (" << p_dec_tanh->J.rows() << "," << p_dec_tanh->J.cols() << ")" << std::endl;
	//std::cout << "out  (" << p_dec_out->J.rows() << "," << p_dec_out->J.cols() << ")" << std::endl;
	//std::cout << "relu (" << p_dec_lrelu2->J.rows() << "," << p_dec_lrelu2->J.cols() << ")" << std::endl;
	//std::cout << "fc2  (" << p_dec_fc2->J.rows() << "," << p_dec_fc2->J.cols() << ")" << std::endl;
	//std::cout << "relu (" << p_dec_lrelu1->J.rows() << "," << p_dec_lrelu1->J.cols() << ")" << std::endl;
	//std::cout << "fc1 (" << p_dec_fc1->J.rows() << "," << p_dec_fc1->J.cols() << ")" << std::endl;

	J = p_dec_tanh->J * 
		p_dec_out->J * 
		p_dec_lrelu2->J * 
		p_dec_fc2->J * 
		p_dec_lrelu1->J * 
		p_dec_fc1->J;

	//grad = J.transpose() * end_grad; 
}