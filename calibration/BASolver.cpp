#include "BASolver.h"
//#include <ceres/rotation.h>
#include <iostream> 

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
		//ceres::AngleAxisRotatePoint(rvec, p_global, p);
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

struct ReprojectionErrorRatio {
	ReprojectionErrorRatio(double observed_x, double observed_y, int _index)
		: observed_x(observed_x), observed_y(observed_y), index(_index) 
	{
		int c = index / 6; 
		int r = index % 6; 
		x = (c - c_dxi) * 1.0; 
		y = (r - c_dyi) * 1.0; 
		z = 0; 
	}

	template <typename T>
	bool operator()(const T* const rvec,
		const T* const tvec,
		const T* const ratio,
		T* residuals) const {
		// camera[0,1,2] are the angle-axis rotation.
		T p[3];
		T point[3]; 
		point[0] = T(x);
		point[1] = T(y) * (*ratio); 
		point[2] = T(z); 
		//ceres::AngleAxisRotatePoint(rvec, point, p);
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
		const double observed_y, const int _index) {
		return (new ceres::AutoDiffCostFunction<ReprojectionErrorRatio, 2, 3, 3, 1>(
			new ReprojectionErrorRatio(observed_x, observed_y, _index)));
	}

	double observed_x;
	double observed_y;
	int index; 
	double x;
	double y; 
	double z; 
};

void BASolver::readInit(std::string _folder)
{
	m_folder = _folder;
	vector<int> camids = m_camids; 
	std::string file_folder = m_folder + "/data/calibdata/init/";
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

}

void BASolver::initMarkers(vector<int> camids, int pointNum)
{
	m_camids = camids; 
	int camNum = m_camids.size(); 
	m_ratio = 1.08575; 
	m_points.resize(pointNum); 
	m_rvecs.resize(camNum); 
	m_tvecs.resize(camNum); 
	m_camNum = camNum; 
	m_pointNum = pointNum; 
	// set init points 3d 
	for(int index = 0; index < m_points.size(); index++)
	{
		int c = index / 6;
		int r = index % 6; 
		m_points[index][0] = (c - c_dxi) * 0.0848; 
		m_points[index][1] = (r - c_dyi) * 0.0848 * m_ratio; 
		m_points[index][2] = 0; 
	}
}

void BASolver::solve_init_calib(bool optim_points)
{
	Problem problem;  
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int pid = 0; pid < m_obs[camid].size(); pid++)
		{
			double obs_x = m_obs[camid][pid][0];
			double obs_y = m_obs[camid][pid][1]; 
			if(optim_points)
			{
				CostFunction *cost = ReprojectionError::Create(obs_x, obs_y); 
				problem.AddResidualBlock(cost, NULL,
					m_rvecs[camid].data(), 
					m_tvecs[camid].data(), 
					m_points[pid].data()
				); 
			}
			else 
			{
				CostFunction *cost = ReprojectionErrorRatio::Create(obs_x, obs_y, pid); 
				problem.AddResidualBlock(cost, NULL,
					m_rvecs[camid].data(), 
					m_tvecs[camid].data(), 
					&m_ratio
				); 
			}
		}
	}

	Solver::Options options; 
	options.minimizer_progress_to_stdout = true; 
	//options.minimizer_type = ceres::TRUST_REGION; 
	options.minimizer_type = ceres::LINE_SEARCH;
	options.max_num_iterations = 200; 
	options.eta = 0.00001; 
	Solver::Summary summary; 
	Solve(options, &problem, &summary); 

	if(!optim_points)
	{
		for(int index = 0; index < m_points.size(); index++)
		{
			int c = index / 6;
			int r = index % 6; 
			c = c - 4; 
			r = r - 3; 
			m_points[index][0] = (c - c_dxi) * 1.0; 
			m_points[index][1] = (r - c_dyi) * 1.0 * m_ratio; 
			m_points[index][2] = 0; 
		}
	}
}

void BASolver::addMarker(const vector<Vec3>& marks, const Vec3& mark3d)
{
	m_added_markers.push_back(marks); 
	m_added_points.push_back(mark3d); 
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
			// if(pid < 42)
			// {
			// 	CostFunction *cost = ReprojectionErrorRatio::Create(obs_x, obs_y, pid); 
			// 	problem.AddResidualBlock(cost, NULL,
			// 		m_rvecs[camid].data(), 
			// 		m_tvecs[camid].data(), 
			// 		&m_ratio
			// 	); 
			// }
			// else 
			// {
				CostFunction *cost = ReprojectionError::Create(obs_x, obs_y); 
				problem.AddResidualBlock(cost, NULL,
					m_rvecs[camid].data(), 
					m_tvecs[camid].data(), 
					m_points[pid].data()
				); 
			// }
		}
	}

	// add new markers 
	for(int i = 0; i < m_added_markers.size(); i++)
	{
		for(int camid = 0; camid < m_camNum; camid++)
		{
			double x = m_added_markers[i][camid](0); 
			double y = m_added_markers[i][camid](1);
			if(x<0) continue; 
			CostFunction *cost = ReprojectionError::Create(x,y);
			problem.AddResidualBlock(cost, NULL, 
				m_rvecs[camid].data(),
				m_tvecs[camid].data(), 
				m_added_points[i].data());
		}
	}

	Solver::Options options; 
	options.minimizer_progress_to_stdout = false; 
	options.minimizer_type = ceres::TRUST_REGION; 
	options.max_num_iterations = 200; 
	options.eta = 0.0001; 
	Solver::Summary summary; 
	Solve(options, &problem, &summary); 
}