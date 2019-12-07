#pragma once

#include <iostream>
#include <iomanip>
#include <fstream> 
#include <vector>
#include <string> 
#include <sstream>
#include <boost/filesystem.hpp>

#include "../associate/camera.h"
#include "../associate/geometry.h"
#include "../associate/math_utils.h"
#include "../associate/image_utils.h"
#include "BASolver.h"

using std::vector; 

class Calibrator{
public: 
    Calibrator(); 
    ~Calibrator(){}
    
    void readAllMarkers(std::string folder); 
    void readK(std::string filename);
    void unprojectMarkers(); 

    // calibration for pig data1
    int calib_pipeline(); 
    void save_results(std::string result_folder); 
    void read_results_rt(std::string result_folder); 
    void evaluate(); 
    void draw_points(); 
    void interactive_mark(); 
    void test_epipolar(); 

    void save_added();
    void reload_added(); 

private: 
    vector<Eigen::Vector3d> out_points; 
    vector<Eigen::Vector3d> out_points_new; 
    vector<Eigen::Vector3d> out_rvecs; 
    vector<Eigen::Vector3d> out_tvecs; 
    double                  out_ratio; 
    vector<cv::Mat> m_imgs;
    vector<cv::Mat> m_imgsUndist; 
    vector<cv::Mat> m_imgsDraw;

    BASolver ba;


    vector<vector<Vec3> >            m_projs_markers; 
    vector<vector<Vec3> >            m_projs_added; 
    vector<vector<Vec3> >            m_added; 
    vector<vector<Vec3> >            m_m_dets;  
    vector<vector<Eigen::Vector2d> > m_markers; // [camNum; pointNum]
    Eigen::Matrix3d                  m_K; 
    vector<vector<Eigen::Vector2d> > m_i_markers; // markers on image plane
    vector<int>                      m_camids; 
    int                              m_camNum; 
    std::vector<Camera>              m_cams;
    std::vector<Camera>              m_camsUndist; 
    std::vector<Eigen::Vector3i>     m_CM; 

    vector<Eigen::Vector2d> readMarkers(std::string filename); 
    void readImgs(); 
    void readCams(); 
};
