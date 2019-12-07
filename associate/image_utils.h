#pragma once 

#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <vector> 

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp> 

#include "camera.h" 
#include "colorterminal.h"

void my_undistort(const cv::Mat &input, cv::Mat &output, const Camera &camera, const Camera &newcam);
void my_undistort_points(const std::vector<Eigen::Vector3d>& points, 
    std::vector<Eigen::Vector3d>& out, const Camera &cam, const Camera &newcam); 

void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3d> &points);
void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3i &color);
void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3i &color, int radius);
void my_draw_point(cv::Mat& img, const Eigen::Vector3d &point, const Eigen::Vector3i& color, int radius); 

void draw_line(cv::Mat &img, Eigen::Vector3d ep, Eigen::Vector3i color_bgr); 
void packImgBlock(const std::vector<cv::Mat> &imgs, cv::Mat &output); 

void getColorMap(std::string cm_type, std::vector<Eigen::Vector3i> &colormap); 
void my_draw_segment(cv::Mat &img, const Vec3& s, const Vec3& e, const Eigen::Vector3i color); 
void my_draw_segment(cv::Mat &img, const Vec3& s, const Vec3& e, const Eigen::Vector3i color, int linewidth, int pointRadius=20); 

void cloneImgs(const std::vector<cv::Mat> & input, std::vector<cv::Mat> &output); 

void getLegend(cv::Mat& out);

std::vector<Eigen::Vector3d> read_points(std::string filename); 

Eigen::Vector4d my_undistort_box(Eigen::Vector4d box,const Camera &cam, const Camera &newcam); 
Eigen::Vector4d expand_box(Eigen::Vector4d box, double ratio = 0.15); 
void my_draw_boxes(cv::Mat& img, const std::vector<Eigen::Vector4d>& boxes); 
void my_draw_box(cv::Mat& img, const Eigen::Vector4d& box, Eigen::Vector3i c);
