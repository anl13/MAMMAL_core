#include "decoder.h"
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen>  
#include <string> 
#include <sstream> 



Eigen::VectorXf readinput(int i)
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
	Eigen::VectorXf latent;
	latent.resize(32);
	for (int k = 0; k < 32; k++)
	{
		inputstream >> latent(k);
	}
	inputstream.close();
	return latent;
}

Eigen::MatrixXf readoutput(int sample_id)
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
	Eigen::MatrixXf output;
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

void readrelu(Eigen::VectorXf& r1, Eigen::VectorXf& r2)
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

void readmiddleoutput(Eigen::VectorXf& out)
{
	out.resize(372);
	std::string folder = "F:/projects/model_preprocess/designed_pig/pig_prior/unittest/";
	std::string relu1file = folder + "middle_out.txt";
	std::ifstream stream(relu1file);
	for (int i = 0; i < 372; i++)stream >> out(i);
	stream.close(); 
}

bool compare_output(Eigen::MatrixXf pred, Eigen::MatrixXf gt)
{
	//std::cout << "gt   cols: " << gt.cols() << " rows: " << gt.rows() << std::endl;
	//std::cout << "pred cols: " << pred.cols() << " rows: " << pred.rows() << std::endl;

	float max_err = 0;
	for (int i = 0; i < 62; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			float gt_value = gt(i, j);
			float pred_value = pred(i * 9 + j);
			float diff = fabs(gt_value - pred_value);
			if (diff > max_err) max_err = diff;
		}
	}

	if (max_err < 1e-5) return true; 
	else {
		std::cout << "max error: " << max_err << std::endl; 
	}
	return false;
}

bool compare_reluout(Eigen::VectorXf pred, Eigen::VectorXf gt)
{
	Eigen::VectorXf diff = pred - gt;
	float err = diff.norm();
	std::cout << "l2 error: " << err << std::endl;
	return false;

}


int unittest_forward()
{
	Decoder dec;

	int sample_id = 0;
	for (sample_id = 0; sample_id < 30; sample_id++)
	{
		Eigen::VectorXf input = readinput(sample_id);
		Eigen::MatrixXf output_gt = readoutput(sample_id);

		dec.latent = input;
		dec.forward();
		Eigen::VectorXf output = dec.output;
	
		//Eigen::Matrix<double, 9, 62, Eigen::ColMajor> output_mat =
		//	Eigen::Map < Eigen::Matrix<double, 9, 62, Eigen::ColMajor> >(output.data()); 
		////std::cout << "prediction: " << std::endl; 
		//std::cout << output_mat.transpose() << std::endl; 
		//std::cout << "gt: " << std::endl;
		//std::cout << output_gt << std::endl; 

		bool state = compare_output(output, output_gt);
		if (state) std::cout << "PASS. case " << sample_id << std::endl; 
		else
		{
			std::cout << "FAIL. case " << sample_id << std::endl; 
		}

		//Eigen::VectorXd middleout;
		//readmiddleoutput(middleout);
		//Eigen::VectorXd diff = middleout - dec.p_dec_out->output; 
		//std::cout << "middle out error: " << diff.norm() << std::endl; 
	}

	return 0;
}

Eigen::VectorXf readgrad(int sample_id)
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
	Eigen::VectorXf latent;
	latent.resize(32);
	for (int k = 0; k < 32; k++)
	{
		inputstream >> latent(k);
	}
	inputstream.close();
	return latent;
}

Eigen::VectorXf compute_endgrad(Eigen::VectorXf output)
{
	Eigen::VectorXf grad = output;
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
		Eigen::VectorXf input = readinput(sample_id);
		Eigen::VectorXf grad = readgrad(sample_id);
		
		dec.latent = input;
		dec.forward();
		Eigen::MatrixXf output = dec.output;

		dec.end_grad = compute_endgrad(output);
		dec.computeJacobi(); 
		Eigen::VectorXf grad_est = dec.J.transpose() * dec.end_grad; 
		Eigen::VectorXf diff = grad_est - grad; 
		std::cout << "err: " << diff.norm() << std::endl;
	}

	return 0; 
}


int unittest_cr()
{
	Eigen::VectorXf input = Eigen::VectorXf::Zero(6); 
	Eigen::VectorXf output_gt = Eigen::VectorXf::Zero(9); 
	Eigen::VectorXf grad_gt = Eigen::VectorXf::Zero(6); 
	std::string folder = "F:/projects/model_preprocess/designed_pig/pig_prior/unittest/";
	std::ifstream inputstream(folder + "/rotinput.txt");
	for (int i = 0; i < 6; i++)inputstream >> input(i); 
	inputstream.close();
	std::ifstream outputstream(folder + "/rotoutput.txt");
	for (int i = 0; i < 9; i++)outputstream >> output_gt(i); 
	outputstream.close(); 
	std::ifstream gradstream(folder + "/rotgrad.txt"); 
	for (int i = 0; i < 6; i++)gradstream >> grad_gt(i); 
	gradstream.close(); 


	ContinousRotation CR; 
	CR.jointnum = 1; 
	CR.input = input.transpose(); 

	CR.forward(); 
	
	std::cout << "est: " << std::endl; 
	std::cout << CR.output.transpose() << std::endl; 
	std::cout << "gt: " << std::endl; 
	std::cout << output_gt.transpose() << std::endl; 

	CR.backward(); 
	Eigen::VectorXf diff = compute_endgrad(CR.output); 
	Eigen::VectorXf grad = CR.J.transpose() * diff; 

	std::cout << std::endl; 
	std::cout << "grad est: " << std::endl
		<< grad.transpose() << std::endl; 
	std::cout << "grad gt: " << std::endl
		<< grad_gt.transpose() << std::endl; 

	return 0; 
}

int unittest_numeric()
{
	Decoder dec;
	int sample_id = 0;

	Eigen::VectorXf input = readinput(sample_id);
	Eigen::VectorXf grad = readgrad(sample_id);

	dec.latent = input;
	dec.forward();
	Eigen::MatrixXf output = dec.output;

	dec.end_grad = compute_endgrad(output);
	dec.computeJacobi();
	Eigen::VectorXf grad_est = dec.J.transpose() * dec.end_grad;
	Eigen::VectorXf diff = grad_est - grad;
	std::cout << "err: " << diff.norm() << std::endl;

	dec.latent = input; 
	dec.forward(); 
	Eigen::VectorXf output0 = dec.output;
	dec.computeJacobi(); 
	Eigen::MatrixXf J = dec.J; 

	Eigen::MatrixXf J_numeric = Eigen::MatrixXf::Zero(9 * 62, 32);
	float alpha = 0.00001; 
	float inv_alpha = 1 / alpha; 
	for (int i = 0; i < 32; i++)
	{
		input(i) += alpha; 
		dec.latent = input; 
		dec.forward();
		J_numeric.col(i) = (dec.output - output0) * inv_alpha; 
		input(i) -= alpha; 
	}
	
	Eigen::MatrixXf D = J - J_numeric; 
	std::cout << "diff.norm(): " << D.norm() << std::endl; 
	std::cout << "J block: " << std::endl  << J.block<10, 10>(0, 0) << std::endl; 
	std::cout << "J_numeric block: "  << std::endl << J_numeric.block<10, 10>(0, 0) << std::endl; 

	return 0; 
}