#include <math.h>
#include <fstream> 
#include <iostream> 
#include "layers.h"

#define INFO 

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
#ifdef INFO
	std::cout << "W: " << W.rows() << ", " << W.cols() << std::endl; 
	std::cout << "b: " << b.rows() << std::endl; 
#endif 

	output = W * input + b; 
}

void LinearLayer::backward()
{
	J = W; 
}

void LeakyReLU::forward()
{
#ifdef INFO
	std::cout << "intput.rows: " << input.rows() << ", cols:" << input.cols() << std::endl; 
#endif 
	output = input; 
	for (int i = 0; i < output.rows(); i++)
	{
		for (int j = 0; j < output.cols(); j++)
		{
			if (output(i,j) < 0)
				output(i,j) = output(i,j) * m_r; 
		}
	}
}

void LeakyReLU::backward()
{
	J = input; 
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			if (J(j, i) > 0) J(j, i) = 1; 
			else J(j, i) = m_r; 
		}
	}
}

void Tanh::forward()
{
#ifdef INFO 
	std::cout << "input.rows: " << input.rows() << ", cols: " << input.cols() << std::endl; 
#endif 
	output = input; 
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			output(i,j) = tanh(input(i,j)); 
		}
	}
}

void Tanh::backward()
{
	int col = output.cols();
	int row = output.rows(); 
	Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(row, col);
	J = ones - (output.array() * output.array()).matrix();
}