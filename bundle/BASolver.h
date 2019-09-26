#pragma once

#include <vector> 
#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <string> 
#include <sstream> 
#include <fstream> 
#include <iomanip> 
#include "../associate/math_utils.h" 

using ceres::Problem; 
using std::vector; 

class BASolver {
public: 
	BASolver() {
		last_add_marker_id = -1; 
	};
	~BASolver() {};

	void setObs(const std::vector<std::vector<Eigen::Vector2d> >& in_obs) {
		m_obs = in_obs;
	}
	void addMarker(const vector<Vec3>& marks, const Vec3& mark3d); 

	void initMarkers(vector<int> camids, int pointNum); 
	void readInit();
	void solve_ratio(); // solve ratio 
	void solve_points(); 
	std::vector<Eigen::Vector3d> getPoints() { return m_points;  }
	std::vector<Eigen::Vector3d> getAddedPoints(){return m_added_points;}
	std::vector<Eigen::Vector3d> getRvecs() { return m_rvecs;  }
	std::vector<Eigen::Vector3d> getTvecs() { return m_tvecs;  }
	double getRatio() {return m_ratio; }
	void solve_again(); 

	vector<int>  m_camids; 
private:
	/// observations 
	std::vector<std::vector<Eigen::Vector2d>> m_obs; // 2d points on image plane 
	std::vector<std::vector<Vec3> > m_added_markers; // [pid, camid], is x < 0, then invisible. 
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