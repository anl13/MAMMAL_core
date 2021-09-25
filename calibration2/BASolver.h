#pragma once

#include <vector> 
#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <string> 
#include <sstream> 
#include <fstream> 
#include <iomanip> 
#include "../utils/math_utils.h" 

using ceres::Problem; 
using std::vector; 

class BASolver {
public: 
	BASolver() {
	};
	~BASolver() {};

	void setObs(const std::vector<std::vector<Eigen::Vector3f> >& in_obs) {
		m_obs.resize(in_obs.size()); 
		for (int i = 0; i < in_obs.size(); i++)
		{
			m_obs[i].resize(in_obs[i].size()); 
			for (int j = 0; j < in_obs[i].size(); j++)
			{
				if (in_obs[i][j](2) > 0)
					m_obs[i][j] = in_obs[i][j].segment<2>(0).cast<double>();
				else
				{
					m_obs[i][j](0) = -1;
					m_obs[i][j](1) = -1; 
				}
			}
		}
	}

	void readInit(std::string _folder);
	
	std::vector<Eigen::Vector3d> getPoints() { return m_points;  }
	std::vector<Eigen::Vector3d> getRvecs() { return m_rvecs;  }
	std::vector<Eigen::Vector3d> getTvecs() { return m_tvecs;  }
	std::vector<Eigen::Vector3f> getPointsF();
	std::vector<Eigen::Vector3f> getRvecsF(); 
	std::vector<Eigen::Vector3f> getTvecsF(); 
	
	void solve_again(); 

	void setInit3DPoints(std::vector<Eigen::Vector3f> points); 
	void setCamIds(const std::vector<int>& camids) {
		m_camids = camids; 
		m_camNum = m_camids.size(); 
		m_rvecs.resize(m_camNum); 
		m_tvecs.resize(m_camNum); 
	}
private:
	/// observations 
	std::vector<int>  m_camids;

	std::string m_folder; 
	std::vector<std::vector<Eigen::Vector2d>> m_obs; // 2d points on image plane 
	/// variables to solve 
	int m_camNum; 
	int m_pointNum; 
	std::vector<Eigen::Vector3d> m_points; // points in global coordinate system
	std::vector<Eigen::Vector3d> m_rvecs;
	std::vector<Eigen::Vector3d> m_tvecs; 
};