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
	std::cout << "gt   cols: " << gt.cols() << " rows: " << gt.rows() << std::endl;
	std::cout << "pred cols: " << pred.cols() << " rows: " << pred.rows() << std::endl;
	
	double max_err = 0; 
	for (int i = 0; i < 62; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			double gt_value = gt(i,j);
			double pred_value = pred(i*9+j);
			double diff = fabs(gt_value - pred_value);
			if (diff > max_err) max_err = diff; 
		}
	}

	std::cout << "max err:" << max_err << std::endl; 

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
	Eigen::VectorXd input = readinput(sample_id); 
	Eigen::MatrixXd output_gt = readoutput(sample_id);
	Eigen::VectorXd relu1_gt;
	Eigen::VectorXd relu2_gt; 
	readrelu(relu1_gt, relu2_gt); 

	dec.latent = input;
	dec.forward();
	Eigen::MatrixXd output = dec.output; 

	compare_reluout(dec.lrelu1_out, relu1_gt);
	compare_reluout(dec.lrelu2_out, relu2_gt);

	compare_output(output, output_gt); 

	return 0; 
}


void main()
{
	unittest_forward();

	system("pause");
	return; 
}