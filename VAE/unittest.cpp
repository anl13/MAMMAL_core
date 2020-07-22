#include "decoder.h"
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen>  
#include <string> 
#include <sstream> 



Eigen::VectorXd readinput(int i)
{
	std::string folder = "F:/projects/model_preprocess/designed_pig/pig_prior/unittest/";
	std::stringstream ss;
	ss << folder << "input" << i << ".txt";
	std::ifstream inputstream(ss.str());
	if (!inputstream.is_open())
	{
		std::cout << "unit test error: cannot open " << ss.str() << std::endl;
		exit(-1);
	}
	Eigen::VectorXd latent;
	latent.resize(32);
	for (int k = 0; k < 32; k++)
	{
		inputstream >> latent(k);
	}
	inputstream.close();
	return latent;
}

Eigen::MatrixXd readoutput(int sample_id)
{
	std::string folder = "F:/projects/model_preprocess/designed_pig/pig_prior/unittest/";
	std::stringstream ss;
	ss << folder << "output" << sample_id << ".txt";
	std::ifstream inputstream(ss.str());
	if (!inputstream.is_open())
	{
		std::cout << "unit test error: cannot open " << ss.str() << std::endl;
		exit(-1);
	}
	Eigen::MatrixXd output;
	output.resize(62, 9);
	for (int i = 0; i < 62; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			inputstream >> output(i, j);
		}
	}
	inputstream.close();
	return output;
}

void readrelu(Eigen::VectorXd& r1, Eigen::VectorXd& r2)
{
	std::string folder = "F:/projects/model_preprocess/designed_pig/pig_prior/unittest/";
	std::string relu1file = folder + "lrelu1_out.txt";
	std::string relu2file = folder + "lrelu2_out.txt";
	r1.resize(512);
	r2.resize(512);
	std::ifstream stream1(relu1file);
	for (int i = 0; i < 512; i++)stream1 >> r1(i);
	stream1.close();
	std::ifstream stream2(relu2file);
	for (int i = 0; i < 512; i++)stream2 >> r2(i);
	stream2.close();
}

bool compare_output(Eigen::MatrixXd pred, Eigen::MatrixXd gt)
{
	//std::cout << "gt   cols: " << gt.cols() << " rows: " << gt.rows() << std::endl;
	//std::cout << "pred cols: " << pred.cols() << " rows: " << pred.rows() << std::endl;

	double max_err = 0;
	for (int i = 0; i < 62; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			double gt_value = gt(i, j);
			double pred_value = pred(i * 9 + j);
			double diff = fabs(gt_value - pred_value);
			if (diff > max_err) max_err = diff;
		}
	}

	if (max_err < 1e-6) return true; 
	return false;
}

bool compare_reluout(Eigen::VectorXd pred, Eigen::VectorXd gt)
{
	Eigen::VectorXd diff = pred - gt;
	double err = diff.norm();
	std::cout << "l2 error: " << err << std::endl;
	return false;

}


int unittest_forward()
{
	Decoder dec;

	int sample_id = 0;
	for (sample_id = 0; sample_id < 30; sample_id++)
	{
		Eigen::VectorXd input = readinput(sample_id);
		Eigen::MatrixXd output_gt = readoutput(sample_id);
		Eigen::VectorXd relu1_gt;
		Eigen::VectorXd relu2_gt;
		readrelu(relu1_gt, relu2_gt);

		dec.latent = input;
		dec.forward();
		Eigen::MatrixXd output = dec.output;

		bool state = compare_output(output, output_gt);
		if (state) std::cout << "PASS. case " << sample_id << std::endl; 
		else
		{
			std::cout << "FAIL. case " << sample_id << std::endl; 
		}
	}

	return 0;
}

Eigen::VectorXd readgrad(int sample_id)
{
	std::string folder = "F:/projects/model_preprocess/designed_pig/pig_prior/unittest/";
	std::stringstream ss;
	ss << folder << "grad" << sample_id << ".txt";
	std::ifstream inputstream(ss.str());
	if (!inputstream.is_open())
	{
		std::cout << "unit test error: cannot open " << ss.str() << std::endl;
		exit(-1);
	}
	Eigen::VectorXd latent;
	latent.resize(32);
	for (int k = 0; k < 32; k++)
	{
		inputstream >> latent(k);
	}
	inputstream.close();
	return latent;
}

Eigen::VectorXd compute_endgrad(Eigen::VectorXd output)
{
	Eigen::VectorXd grad = output;
	for (int i = 0; i < grad.rows(); i++)
	{
		grad(i) = 2 * (output(i) - 1);
	}
	return grad; 
}


int unittest_backward()
{
	Decoder dec; 
	int sample_id = 0; 

	for (sample_id = 0; sample_id < 20; sample_id++)
	{
		Eigen::VectorXd input = readinput(sample_id);
		Eigen::VectorXd grad = readgrad(sample_id);
		
		dec.latent = input;
		dec.forward();
		Eigen::MatrixXd output = dec.output;

		dec.end_grad = compute_endgrad(output);
		dec.backward(); 

		//std::cout << dec.grad.transpose() << std::endl; 

		Eigen::VectorXd diff = dec.grad - grad; 
		std::cout << "err: " << diff.norm() << std::endl;

	}

	return 0; 
}