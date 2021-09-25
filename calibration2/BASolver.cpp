#include "BASolver.h"
//#include <ceres/rotation.h>
#include <iostream> 
#include <ceres/rotation.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
//using ceres::LossFunction;
//using ceres::LossFunctionWrapper;

const float c_dxi = 0; 
const float c_dyi = 0; 
struct ReprojectionError {
	ReprojectionError(double observed_x, double observed_y)
		: observed_x(observed_x), observed_y(observed_y) {}

	template <typename T>
	bool operator()(const T* const rvec,
		const T* const tvec, 
		const T* const point,
		T* residuals) const {
		// camera[0,1,2] are the angle-axis rotation.
		T p_global[3];
		p_global[0] = point[0]; 
		p_global[1] = point[1];
		p_global[2] = point[2];
		T p[3];
		ceres::AngleAxisRotatePoint(rvec, p_global, p);
		// camera[3,4,5] are the translation.
		p[0] += tvec[0]; p[1] += tvec[1]; p[2] += tvec[2];

		T x = p[0] / p[2];
		T y = p[1] / p[2]; 

		// The error is the difference between the predicted and observed position.
		residuals[0] = x - T(observed_x);
		residuals[1] = y - T(observed_y);
		return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x,
		const double observed_y) {
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 3>(
			new ReprojectionError(observed_x, observed_y)));
	}

	double observed_x;
	double observed_y;
};

struct HeightTerm {
	HeightTerm()
	{
	
	}

	template <typename T>
	bool operator()(const T* const point, 
		T* residuals) const {

		// The error is the difference between the predicted and observed position.
		residuals[0] = point[2];
		return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create() {
		return (new ceres::AutoDiffCostFunction<HeightTerm, 1, 3>(
			new HeightTerm()));
	}
};


void BASolver::readInit(std::string _folder)
{
	m_folder = _folder;
	vector<int> camids = m_camids; 
	std::string file_folder = m_folder;
	for(int i = 0; i < m_camNum; i++)
	{
		std::stringstream ss; 
		ss << file_folder << std::setw(2) << std::setfill('0') << camids[i] << ".txt";
		std::ifstream is;
		is.open(ss.str());  
		if (!is.is_open())
		{
			std::cout << "error reading file " << ss.str() << std::endl; 
			exit(-1); 
		}
		for(int j = 0; j < 6; j++)
		{
			double a; 
			is >> a; 
			if(j < 3)
			{
				m_rvecs[i][j] = a; 
			}
			else 
			{
				m_tvecs[i][j-3] = a; 
			}
		}
		is.close(); 
	}

	std::cout << "...ba.readInit()..." << std::endl; 
	for (int k = 0; k < m_camNum; k++)
	{

	}
}

void BASolver::solve_again()
{
	Problem problem;

	// add grid markers 
	for (int camid = 0; camid < m_camNum; camid++)
	{
		std::cout << "m_obs[camid].size(): " << m_obs[camid].size() << std::endl; 
		for (int pid = 0; pid < m_pointNum; pid++)
		{
			double obs_x = m_obs[camid][pid][0];
			double obs_y = m_obs[camid][pid][1]; 
			if (obs_x < 0 || obs_y < 0) continue; // rule out invalid points 
			if (m_points[pid].norm() == 0) continue; // rule out points where no init triangulation is accessible.
			CostFunction *cost = ReprojectionError::Create(obs_x, obs_y); 
			problem.AddResidualBlock(cost, NULL,
				m_rvecs[camid].data(), 
				m_tvecs[camid].data(), 
				m_points[pid].data()
			); 
		}
	}
	//for (int pid = 0; pid < 42; pid++)
	//{
	//	CostFunction *cost = HeightTerm::Create();
	//	problem.AddResidualBlock(cost, NULL, m_points[pid].data());
	//}

	Solver::Options options; 
	options.minimizer_progress_to_stdout = true; 
	options.minimizer_type = ceres::TRUST_REGION; 
	options.max_num_iterations = 200; 
	options.eta = 0.000001; 
	Solver::Summary summary; 
	Solve(options, &problem, &summary); 
}


std::vector<Eigen::Vector3f> BASolver::getPointsF()
{
	return doubleToFloat(m_points); 
}

std::vector<Eigen::Vector3f> BASolver::getRvecsF()
{
	return doubleToFloat(m_rvecs); 
}

std::vector<Eigen::Vector3f> BASolver::getTvecsF()
{
	return doubleToFloat(m_tvecs); 
}

void BASolver::setInit3DPoints(std::vector<Eigen::Vector3f> points)
{
	m_pointNum = points.size(); 
	m_points.resize(m_pointNum); 
	for (int i = 0; i < m_pointNum; i++)
	{
		m_points[i] = points[i].cast<double>();
	}
}