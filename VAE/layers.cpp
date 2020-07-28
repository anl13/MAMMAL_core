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

void ContinousRotation::forward()
{
	
	output.resize(jointnum * 9);
	output.setZero(); 
	for (int jid = 0; jid < jointnum; jid++)
	{
		Eigen::VectorXd vec = input.segment<6>(jid * 6); 

		output.segment<9>(jid * 9) = computey(vec); 
	}
}

Eigen::VectorXd ContinousRotation::computey(Eigen::VectorXd x)
{
	Eigen::Matrix<double, 2, 3, Eigen::ColMajor> vec_mat = Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::ColMajor>>(x.data());
	Eigen::Vector3d r1 = vec_mat.row(0);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(9);
	if (r1.norm() < 1e-6) return y;
	Eigen::Vector3d r2 = vec_mat.row(1);
	Eigen::Vector3d r1_norm = r1.normalized();
	double dot = r2.dot(r1_norm);
	Eigen::Vector3d r2_norm = r2 - dot * r1_norm;
	r2_norm.normalize();
	Eigen::Vector3d r3_norm = r1_norm.cross(r2_norm);
	Eigen::Matrix3d R;
	R.col(0) = r1_norm;
	R.col(1) = r2_norm;
	R.col(2) = r3_norm;

	y.segment<3>(0) = R.row(0); 
	y.segment<3>(3) = R.row(1); 
	y.segment<3>(6) = R.row(2); 
	return y; 
}


Eigen::Matrix3d ContinousRotation::computeJVecNorm(Eigen::Vector3d x)
{
	double n = x.norm();
	Eigen::Matrix3d J; 
	double n3 = n * n*n;
	J(0, 0) = 1 / n - x[0] * x[0] / n3;
	J(0, 1) = -x[0] * x[1] / n3;
	J(0, 2) = -x[0] * x[2] / n3; 
	J(1, 0) = -x[1] * x[0] / n3;
	J(1, 1) = 1 / n - x[1] * x[1] / n3;
	J(1, 2) = -x[1] * x[2] / n3;
	J(2, 0) = -x[2] * x[0] / n3; 
	J(2, 1) = -x[2] * x[1] / n3; 
	J(2, 2) = 1 / n - x[2] * x[2] / n3;
	return J; 
}

// c = a - (a.dot(b))b (assume b is normalized)
// return dc/da^T, dc/db^T
void ContinousRotation::computeJr2(Eigen::Vector3d a,
	Eigen::Vector3d b, Eigen::Matrix3d& Ja, Eigen::Matrix3d& Jb)
{
	Ja(0, 0) = 1 - b[0] * b[0];
	Ja(0, 1) = -b[1] * b[0];
	Ja(0, 2) = -b[2] * b[0];
	Ja(1, 0) = -b[0] * b[1];
	Ja(1, 1) = 1 - b[1] * b[1];
	Ja(1, 2) = -b[2] * b[1];
	Ja(2, 0) = -b[0] * b[2];
	Ja(2, 1) = -b[1] * b[2];
	Ja(2, 2) = 1 - b[2] * b[2];

	double d = a.dot(b); 
	Jb(0, 0) = -d -a[0] * b[0];
	Jb(0, 1) = -a[1] * b[0];
	Jb(0, 2) = -a[2] * b[0];
	Jb(1, 0) = -a[0] * b[1];
	Jb(1, 1) = -d -a[1] * b[1];
	Jb(1, 2) = -a[2] * b[1];
	Jb(2, 0) = -a[0] * b[2]; 
	Jb(2, 1) = -a[1] * b[2];
	Jb(2, 2) = -d -a[2] * b[2];
}

void ContinousRotation::computeJCross(Eigen::Vector3d a,
	Eigen::Vector3d b,Eigen::Matrix3d d_b_d_a, Eigen::Matrix3d& Ja, Eigen::Matrix3d& Jb)
{
	Ja = Eigen::Matrix3d::Zero(); 
	Jb = Eigen::Matrix3d::Zero(); 
	Ja(0, 1) = b[2];
	Ja(0, 2) = -b[1];
	Ja(1, 0) = -b[2];
	Ja(1, 2) = b[0];
	Ja(2, 0) = b[1];
	Ja(2, 1) = -b[0];

	Jb(0, 1) = -a[2];
	Jb(0, 2) = a[1];
	Jb(1, 0) = a[2];
	Jb(1, 2) = -a[0];
	Jb(2, 0) = -a[1];
	Jb(2, 1) = a[0];

	Ja = Ja + Jb * d_b_d_a;
}

Eigen::MatrixXd ContinousRotation::computeJBlock(Eigen::VectorXd x)
{
	// x: 6
	// y: 9
	// y: r1n(0), r2n(0), r3n(0), r1n(1) ...
	// x: r1(0), r2(0), r1(1) ..
	Eigen::Matrix<double, 9, 6> J; 
	J.setZero();
	if (x.norm() < 1e-6) return J; 
	
	Eigen::Matrix<double, 2, 3, Eigen::ColMajor> vec_mat = Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::ColMajor>>(x.data());
	Eigen::Vector3d r1 = vec_mat.row(0);
	Eigen::VectorXd y = Eigen::VectorXd::Zero(9);
	
	Eigen::Vector3d r2 = vec_mat.row(1);
	Eigen::Vector3d r1_norm = r1.normalized();
	double dot = r2.dot(r1_norm);
	Eigen::Vector3d r2_nonorm = r2 - dot * r1_norm;
	Eigen::Vector3d r2_norm = r2_nonorm.normalized();
	Eigen::Vector3d r3_norm = r1_norm.cross(r2_norm);
	
	Eigen::Matrix3d d_r2n_d_r2nn = computeJVecNorm(r2_nonorm);
	Eigen::Matrix3d d_r2nn_d_r2, d_r2nn_d_r1n;
	computeJr2(r2, r1_norm, d_r2nn_d_r2, d_r2nn_d_r1n);
	Eigen::Matrix3d d_r1n_d_r1 = computeJVecNorm(r1);
	Eigen::Matrix3d d_r2n_d_r1n = d_r2n_d_r2nn * d_r2nn_d_r1n;

	Eigen::Matrix3d d_r3n_d_r1n, d_r3n_d_r2n;
	computeJCross(r1_norm, r2_norm, d_r2n_d_r1n, d_r3n_d_r1n, d_r3n_d_r2n);
	
	Eigen::Matrix3d d_r2n_d_r1 = d_r2n_d_r2nn * d_r2nn_d_r1n * d_r1n_d_r1;
	Eigen::Matrix3d d_r2n_d_r2 = d_r2n_d_r2nn * d_r2nn_d_r2;
	Eigen::Matrix3d d_r3n_d_r1 = d_r3n_d_r1n * d_r1n_d_r1 /*+ d_r3n_d_r2n * d_r2n_d_r1*/;
	Eigen::Matrix3d d_r3n_d_r2 = d_r3n_d_r2n * d_r2n_d_r2;

	// reoder: 
	// y: r1n(0), r1n(1), r1n(2), r2n(0), ...
	// x: r1(0), r1(1), r1(2) ...
	Eigen::Matrix<double, 9, 6> J_reorder; 
	J_reorder.setZero(); 
	J_reorder.block<3, 3>(0, 0) = d_r1n_d_r1;
	J_reorder.block<3, 3>(3, 0) = d_r2n_d_r1;
	J_reorder.block<3, 3>(3, 3) = d_r2n_d_r2;
	J_reorder.block<3, 3>(6, 0) = d_r3n_d_r1;
	J_reorder.block<3, 3>(6, 3) = d_r3n_d_r2; 

	//J_reorder.block<3, 3>(0, 0) = d_r3n_d_r1;
	//J_reorder.block<3, 3>(0, 3) = d_r3n_d_r2;
	//J_reorder.block<3, 3>(3, 0) = d_r3n_d_r1;
	//J_reorder.block<3, 3>(3, 3) = d_r3n_d_r2;
	//J_reorder.block<3, 3>(6, 0) = d_r3n_d_r1;
	//J_reorder.block<3, 3>(6, 3) = d_r3n_d_r2; 



	Eigen::Matrix<double, 9, 6> J_roworder; 
	std::vector<int> row_order = { 0, 3,6,1,4,7,2,5,8 };
	for (int i = 0; i < 9; i++)J_roworder.row(i) = J_reorder.row(row_order[i]);
	std::vector<int> col_order = { 0,3,1,4,2,5 };
	for (int i = 0; i < 6; i++)J.col(i) = J_roworder.col(col_order[i]);

	return J; 
}

// compute dy/dx^T
// output: [9,6] matrix 
void ContinousRotation::backward()
{
	int rows = jointnum * 9;
	int cols = jointnum * 6;
	J.resize(rows, cols); 
	J.setZero();
	for (int i = 0; i < jointnum; i++)
	{
		J.block<9, 6>(9 * i, 6 * i) = computeJBlock(input.segment<6>(6 * i));
	}
}