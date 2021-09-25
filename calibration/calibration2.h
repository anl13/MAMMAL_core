#pragma once

#include <iostream>
#include <iomanip>
#include <fstream> 
#include <vector>
#include <string> 
#include <sstream>
#include <boost/filesystem.hpp>

#include "../utils/camera.h"
#include "../utils/geometry.h"
#include "../utils/math_utils.h"
#include "../utils/image_utils.h"
#include "BASolver.h"

using std::vector;

class Calibrator2 {
public:
	Calibrator2();
	~Calibrator2() {}

	void setFolder(std::string _folder) { m_folder = _folder; }

	void readAllMarkers(std::string folder);
	void unprojectMarkers();

	// calibration for pig data1
	int calib_pipeline();
	void save_results(std::string result_folder);
	void read_results_rt(std::string result_folder);
	void evaluate();
	void draw_points();
	void test_epipolar();

	void readInitResult(); 
	void readInit3DPoints();

private:
	std::string m_folder;
	vector<Eigen::Vector3f> out_points;
	vector<Eigen::Vector3f> out_rvecs;
	vector<Eigen::Vector3f> out_tvecs;
	double                  out_ratio;
	vector<cv::Mat> m_imgsUndist;
	vector<cv::Mat> m_imgsDraw;

	BASolver ba;
	std::vector<int> m_camids;

	int m_draw_size;
	vector<vector<Eigen::Vector3f> > m_projs_markers;
	vector<vector<Eigen::Vector3f> > m_projs_added;
	vector<vector<Eigen::Vector3f> > m_added;
	vector<vector<Eigen::Vector3f> > m_m_dets;
	vector<vector<Eigen::Vector3f> > m_markers; // [camNum; pointNum]
	vector<vector<Eigen::Vector3f> > m_i_markers; // [camNum; pointNum] 
	Eigen::Matrix3f                  m_K;
	vector<int>                      m_camids;
	int                              m_camNum;
	std::vector<Camera>              m_camsUndist;
	std::vector<Eigen::Vector3i>     m_CM;

	vector<Eigen::Vector3f> readMarkers(std::string filename);
	void readImgs();
};
