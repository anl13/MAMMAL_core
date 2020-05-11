#pragma once 

#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <vector> 

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/eigen.hpp> 

#include "camera.h" 
#include "colorterminal.h"

using std::vector; 

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
void my_draw_mask(cv::Mat& img, vector<vector<Eigen::Vector2d> > contour_list, Eigen::Vector3i c, float alpha=0);
void my_draw_mask_gray(cv::Mat& img, vector<vector<Eigen::Vector2d> > contour_list, int c);

// outimg = img1 * alpha + img2 * (1-alpha)
cv::Mat blend_images(cv::Mat img1, cv::Mat img2, float alpha);
cv::Mat overlay_renders(cv::Mat rawimg, cv::Mat render, float alpha=0.5f); 

Eigen::Vector3f rgb2bgr(const Eigen::Vector3f& color); 

cv::Mat resizeAndPadding(cv::Mat img, const int width, const int height); 

cv::Mat get_dist_trans(cv::Mat input);



// This class is used as struct 
class ROIdescripter {
public:
	ROIdescripter() {}
	inline void setCam(const Camera& _cam) { cam = _cam; }
	inline void setId(const int& _id) { id = _id; }
	inline void setT(const int& _t) { t = _t; }

	int pid; 
	int idcode; 
	double area; 
	cv::Mat undist_mask; 
	cv::Mat chamfer; // <float>
	cv::Mat mask; // <uint8> including other body, to check visibility
	std::vector<std::vector<Eigen::Vector2d> > mask_list;
	std::vector<std::vector<Eigen::Vector2d> > mask_norm;
	cv::Mat gradx, grady;

	Camera cam; 
	int viewid; 
	Eigen::Vector4d box; // (x,y,x+w,y+h)
	int id;
	int t; 
	/*
	return: -1: outof image
	0: background
	1: yes 
	2: occluded by other pig
	*/
	int queryMask(const Eigen::Vector3d& point);
	/*
	return:
	-10000: outof image 
	other: chamfer value. <0 means outside contour, >0 means inside contour.
	*/
	float queryChamfer(const Eigen::Vector3d& point);
};

float queryPixel(const cv::Mat& img, const Eigen::Vector3d& point, const Camera& cam);

cv::Mat reverseChamfer(const cv::Mat& chamfer);
cv::Mat my_background_substraction(cv::Mat raw, cv::Mat bg);

std::vector<Eigen::Vector2d> computeContourNormal(const std::vector<Eigen::Vector2d>& points);
std::vector<std::vector<Eigen::Vector2d> >  computeContourNormalsAll(const std::vector<std::vector<Eigen::Vector2d> >&points);

cv::Mat computeSDF2d(const cv::Mat& mask, int thresh=-1);// compute signed distance function from mask image
cv::Mat visualizeSDF2d(cv::Mat tsdf, int thresh=-1);

// input and output are CV_32F images
void computeGradient(cv::Mat input, cv::Mat& outx, cv::Mat& outy);