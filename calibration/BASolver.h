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
		last_add_marker_id = -1; 
	};
	~BASolver() {};

	void setObs(const std::vector<std::vector<Eigen::Vector3f> >& in_obs) {
		m_obs.resize(in_obs.size()); 
		for (int i = 0; i < in_obs.size(); i++)
		{
			m_obs[i].resize(in_obs[i].size()); 
			for (int j = 0; j < in_obs[i].size(); j++)
				m_obs[i][j] = in_obs[i][j].segment<2>(0).cast<double>();
		}
	}
	void addMarker(const vector<Eigen::Vector3d>& marks, const Eigen::Vector3d& mark3d); 
	void addMarkerF(const vector<Eigen::Vector3f>& marks, const Eigen::Vector3f& mark3d); 

	void initMarkers(vector<int> camids, int pointNum); 
	void readInit(std::string _folder);
	void solve_init_calib(bool optim_points = true); 
	std::vector<Eigen::Vector3d> getPoints() { return m_points;  }
	std::vector<Eigen::Vector3d> getAddedPoints(){return m_added_points;}
	std::vector<Eigen::Vector3d> getRvecs() { return m_rvecs;  }
	std::vector<Eigen::Vector3d> getTvecs() { return m_tvecs;  }
	std::vector<Eigen::Vector3f> getPointsF();
	std::vector<Eigen::Vector3f> getAddedPointsF();
	std::vector<Eigen::Vector3f> getRvecsF(); 
	std::vector<Eigen::Vector3f> getTvecsF(); 
	
	double getRatio() {return m_ratio; }
	void solve_again(); 

	vector<int>  m_camids; 
private:
	/// observations 
	std::string m_folder; 
	std::vector<std::vector<Eigen::Vector2d>> m_obs; // 2d points on image plane 
	std::vector<std::vector<Eigen::Vector3d> > m_added_markers; // [pid, camid], is x < 0, then invisible. 
	int last_add_marker_id; 
	/// variables to solve 
	double m_ratio; 
	int m_camNum; 
	int m_pointNum; 
	std::vector<Eigen::Vector3d> m_points; // points in global coordinate system
	std::vector<Eigen::Vector3d> m_added_points; 
	std::vector<Eigen::Vector3d> m_rvecs;
	std::vector<Eigen::Vector3d> m_tvecs; 
};