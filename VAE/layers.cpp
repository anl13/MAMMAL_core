#include <math.h>
#include <fstream> 
#include <iostream> 
#include "layers.h"


void LinearLayer::load_params(std::string filew, std::string fileb)
{
	W.resize(m_dimout, m_dimin);
	b.resize(m_dimout);
	std::ifstream wstream(filew); 
	if (!wstream.is_open())
	{
		std::cout << filew << " not open" << std::endl; 
		exit(-1); 
	}
	for (int i = 0; i < m_dimout; i++)
	{
		for (int j = 0; j < m_dimin; j++)
		{
			wstream >> W(i, j);
		}
	}
	wstream.close(); 
	std::ifstream bstream(fileb); 
	if (!bstream.is_open())
	{
		std::cout << fileb << " not open " << std::endl;
		exit(-1);
	}
	for (int i = 0; i < m_dimout; i++)
	{
		bstream >> b(i); 
	}
	bstream.close(); 
}

void LinearLayer::forward()
{
	output = W * input + b; 
}

void LinearLayer::backward()
{
	grad = W; 
}

void LeakyReLU::forward()
{
	output = input; 
	for (int i = 0; i < output.rows(); i++)
	{
		for (int j = 0; j < output.cols(); j++)
		{
			if (output(j, i) < 0)
				output(j, i) = output(j, i) * m_r; 
		}
	}
}

void LeakyReLU::backward()
{
	grad = input; 
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			if (grad(j, i) > 0) grad(j, i) = 1; 
			else grad(j, i) = m_r; 
		}
	}
}

void Tanh::forward()
{
	output = input; 
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			output(j, i) = tanh(input(j, i)); 
		}
	}
}

void Tanh::backward()
{
	int col = output.cols();
	int row = output.rows(); 
	Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(row, col);
	grad = ones - (output.array() * output.array()).matrix();
}