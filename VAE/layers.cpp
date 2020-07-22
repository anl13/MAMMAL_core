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
	output = W * input + b; 
}

void LinearLayer::backward()
{
	J = W; 
}

void LeakyReLU::forward()
{
	output = input; 
	for (int i = 0; i < output.rows(); i++)
	{
		if (output(i) < 0)
			output(i) = output(i) * m_r; 
	}
}

void LeakyReLU::backward()
{
	int rows = input.rows(); 
	J = Eigen::MatrixXd::Zero(rows, rows);
	for (int i = 0; i < input.rows(); i++)
	{
		if (input(i) > 0) J(i,i) = 1; 
		else J(i,i) = m_r; 
	}
}

void Tanh::forward()
{
	output = input; 
	for (int i = 0; i < input.rows(); i++)
	{
		output(i) = tanh(input(i)); 
	}
}

void Tanh::backward()
{
	J = Eigen::MatrixXd::Identity(558, 558);
	for (int i = 0; i < 558; i++)
	{
		J(i, i) = 1 - output(i)*output(i); 
	}
}