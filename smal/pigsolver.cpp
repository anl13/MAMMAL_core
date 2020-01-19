#include "pigsolver.h"
#include <ceres/ceres.h>
#include <ceres/numeric_diff_cost_function.h>
#include <iostream>
#include <fstream>
#include <stdio.h> 
#include <cstring>

using ceres::NumericDiffCostFunction;
using ceres::Problem;
using ceres::Solver; 
using ceres::Solve;

#define JN 43
#define VN 2176 

struct E_R
{
	E_R(double* _V, double* _J)
	{
		V = _V; 
		J = _J; 
	}

	template<typename T> 
	bool operator()(const T* const X, T* residuals)const
	{
		for (int i = 0; i < JN; i++)
		{
			T AJ[3];
			AJ[0] = T(0.0); 
			AJ[1] = T(0.0); 
			AJ[2] = T(0.0); 
			for (int j = 0; j < VN; j++)
			{
				AJ[0] = 
					T(V[3 * j + 0]) *
					X[i * VN + j] + AJ[0];
				AJ[1] = T(V[3 * j + 1]) * X[i * VN + j] + AJ[1];
				AJ[2] = T(V[3 * j + 2]) * X[i * VN + j] + AJ[2];
			}
			residuals[3 * i + 0] = AJ[0] - J[3 * i + 0];
			residuals[3 * i + 1] = AJ[1] - J[3 * i + 1];
			residuals[3 * i + 2] = AJ[2] - J[3 * i + 2]; 
		}
		return true; 
	}

	double* V; 
	double* J;
};


void PigSolver::solveR(int samplenum)
{
	R.resize(VN, JN); 
	R.setZero();
	double *Rvec = new double[VN*JN]; 
	for (int i = 0; i < VN*JN; i++)
		Rvec[i] = 0; 

	Problem problem; 
	std::vector<double*> Vs;
	std::vector<double*> Js; 
	for (int i = 0; i < 800; i++)
	{
		std::cout << "add term " << i << std::endl; 
		m_poseParam = Eigen::VectorXd::Random(3 * m_jointNum); 
		UpdateVertices(); 
		double *V = new double[VN * 3];
		double *J = new double[JN * 3];
		Vs.push_back(V);
		Js.push_back(J); 
		memcpy(V, m_verticesFinal.data(), VN * 3 * sizeof(double)); 
		memcpy(J, m_jointsFinal.data(), JN * 3 * sizeof(double)); 
		E_R * term = new E_R(
			V, J
		); 
		ceres::CostFunction *costfunc =
			//new NumericDiffCostFunction<E_R, ceres::CENTRAL, VN * JN, 3 * JN>(term);
			new ceres::AutoDiffCostFunction<E_R, VN * JN, 3 * JN>(term); 
		problem.AddResidualBlock(costfunc, NULL, Rvec); 
	}
	Solver::Options options; 
	options.minimizer_progress_to_stdout = true; 
	options.minimizer_type = ceres::TRUST_REGION;
	options.max_num_iterations = 200; 
	Solver::Summary summary; 
	Solve(options, &problem, &summary); 

	//memcpy(R.data(), Rvec, JN*VN * sizeof(double)); 
	//std::ofstream outfile(m_folder + "regressor.txt");
	//outfile << R;
	//outfile.close(); 
}

void PigSolver::generateSamples(int samplenum)
{
	for (int i = 0; i < samplenum; i++)
	{
		std::stringstream ss; 
		ss << m_folder << "samples/" << std::setw(6) << std::setfill('0') << i << ".txt"; 
		std::ofstream is(ss.str());
		m_poseParam = Eigen::VectorXd::Random(3 * m_jointNum);
		UpdateVertices();
		is << m_verticesFinal.transpose() << std::endl; 
		is << m_jointsFinal.transpose(); 
		is.close(); 
	}
}

void PigSolver::solve()
{
	
}