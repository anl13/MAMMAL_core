#pragma once 

#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <vector> 

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp> 

#include "camera.h" 

void my_undistort(const cv::Mat &input, cv::Mat &output, const Camera &camera, const Camera &newcam);
void my_undistort_points(const std::vector<Eigen::Vector3d>& points, 
    std::vector<Eigen::Vector3d>& out, const Camera &cam, const Camera &newcam); 

void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3d> &points);
void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3i &color);
void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3i &color, int radius);

bool in_image(float w, float h, float x, float y); 
void draw_line(cv::Mat &img, Eigen::Vector3d ep); 
void packImgBlock(const std::vector<cv::Mat> &imgs, cv::Mat &output); 

void getColorMap(std::string cm_type, std::vector<Eigen::Vector3i> &colormap); 
void my_draw_segment(cv::Mat &img, const Vec3& s, const Vec3& e, const Eigen::Vector3i color); 
void my_draw_segment(cv::Mat &img, const Vec3& s, const Vec3& e, const Eigen::Vector3i color, int linewidth, int pointRadius=20); 

void cloneImgs(const std::vector<cv::Mat> & input, std::vector<cv::Mat> &output); 

extern std::vector<std::string> LABEL_NAMES;

void getLegend(cv::Mat& out);