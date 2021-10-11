#pragma once 

#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <vector> 

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/eigen.hpp> 

//#include <vector_functions.h>
//#include <nanogui/vector.h>

#include "camera.h" 
#include "colorterminal.h"

using std::vector; 

void my_undistort(const cv::Mat &input, cv::Mat &output, const Camera &camera, const Camera &newcam);
void my_undistort_points(const std::vector<Eigen::Vector3f>& points, 
    std::vector<Eigen::Vector3f>& out, const Camera &cam, const Camera &newcam); 
void my_undistort_points(const std::vector<Eigen::Vector2f>& points,
	std::vector<Eigen::Vector2f>& out, const Camera &cam, const Camera &newcam);

void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3f> &points);
void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3f> &points, const Eigen::Vector3i &color);
void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3f> &points, const Eigen::Vector3i &color, int radius);
void my_draw_point(cv::Mat& img, const Eigen::Vector3f &point, const Eigen::Vector3i& color, int radius); 

void draw_line(cv::Mat &img, Eigen::Vector3f ep, Eigen::Vector3i color_bgr); 
void packImgBlock(const std::vector<cv::Mat> &imgs, cv::Mat &output); 

void getColorMap(std::string cm_type, std::vector<Eigen::Vector3i> &colormap); 
//std::vector<float4> getColorMapFloat4(std::string cm_type); 
std::vector<Eigen::Vector3i> getColorMapEigen(std::string cm_type);
//std::vector<nanogui::Vector4f> getColorMapNano(std::string cm_type);
std::vector<Eigen::Vector3f> getColorMapEigenF(std::string cm_type); 

void my_draw_segment(cv::Mat &img, const Eigen::Vector3f& s, const Eigen::Vector3f& e, const Eigen::Vector3i color);
void my_draw_segment(cv::Mat &img, const Eigen::Vector3f& s, const Eigen::Vector3f& e, const Eigen::Vector3i color, int linewidth, int pointRadius=20);

void cloneImgs(const std::vector<cv::Mat> & input, std::vector<cv::Mat> &output); 

//void getLegend(cv::Mat& out);

std::vector<Eigen::Vector3f> read_points(std::string filename); 

Eigen::Vector4f my_undistort_box(Eigen::Vector4f box,const Camera &cam, const Camera &newcam); 
Eigen::Vector4f expand_box(Eigen::Vector4f box, float ratio = 0.15); 
void my_draw_boxes(cv::Mat& img, const std::vector<Eigen::Vector4f>& boxes); 
void my_draw_box(cv::Mat& img, const Eigen::Vector4f& box, Eigen::Vector3i c);
void my_draw_mask(cv::Mat& img, vector<vector<Eigen::Vector2f> > contour_list, Eigen::Vector3i c, float alpha=0, bool is_contour=false);
void my_draw_mask_gray(cv::Mat& img, vector<vector<Eigen::Vector2f> > contour_list, int c);
void my_draw_box_fill_gray(cv::Mat& img, const Eigen::Vector4f& box, unsigned int c); 

// outimg = img1 * alpha + img2 * (1-alpha)
cv::Mat blend_images(cv::Mat img1, cv::Mat img2, float alpha);
cv::Mat overlay_renders(cv::Mat rawimg, cv::Mat render, float alpha=0.5f); 

Eigen::Vector3f rgb2bgr(const Eigen::Vector3f& color); 

cv::Mat resizeAndPadding(cv::Mat img, const int width, const int height); 

cv::Mat get_dist_trans(cv::Mat input);

cv::Mat my_resize(const cv::Mat& input, float ratio);

// This class is used as struct 
class ROIdescripter {
public:
	ROIdescripter() {}
	inline void setCam(const Camera& _cam) { cam = _cam; }
	inline void setId(const int& _id) { id = _id; }
	inline void setT(const int& _t) { t = _t; }

	int pid; 
	int idcode; 
	float area; 
	cv::Mat undist_mask; 
	cv::Mat binary_mask; 
	std::vector<Eigen::Vector3f> keypoints; 
	cv::Mat scene_mask; 
	cv::Mat chamfer; // <float>
	cv::Mat mask; // <uint8> including other body, to check visibility
	std::vector<std::vector<Eigen::Vector2f> > mask_list;
	std::vector<std::vector<Eigen::Vector2f> > mask_norm;
	cv::Mat gradx, grady;

	Camera cam; 
	int viewid; 
	Eigen::Vector4f box; // (x,y,x+w,y+h)
	int id;
	int t; // timestamp
	/*
	return: -1: outof image
	0: background
	1: yes 
	2: occluded by other pig
	3: occluded by scene 
	*/
	int queryMask(const Eigen::Vector3f& point);
	/*
	return:
	-10000: outof image 
	other: chamfer value. <0 means outside contour, >0 means inside contour.
	*/
	float queryChamfer(const Eigen::Vector3f& point);
	float keypointsMaskOverlay(); 
	float valid; 
};

float checkKeypointsMaskOverlay(const cv::Mat& mask, const std::vector<Eigen::Vector3f>& keypoints,
	const int& idcode);

float queryPixel(const cv::Mat& img, const Eigen::Vector3f& point, const Camera& cam);
float queryDepth(const cv::Mat& img, float x, float y);

cv::Mat reverseChamfer(const cv::Mat& chamfer);

std::vector<Eigen::Vector2f> computeContourNormal(const std::vector<Eigen::Vector2f>& points);
std::vector<std::vector<Eigen::Vector2f> >  computeContourNormalsAll(const std::vector<std::vector<Eigen::Vector2f> >&points);

cv::Mat my_background_substraction(cv::Mat raw, cv::Mat bg); 

cv::Mat computeSDF2d(const cv::Mat& mask, int thresh=-1);// compute signed distance function from mask image
cv::Mat computeSDF2dFromDepthf(const cv::Mat& depth, int thresh = -1);
cv::Mat fromDepthToColorMask(cv::Mat depth);

cv::Mat visualizeSDF2d(cv::Mat tsdf, int thresh = -1);
cv::Mat pseudoColor(cv::Mat depth);

// input and output are CV_32F images
void computeGradient(cv::Mat input, cv::Mat& outx, cv::Mat& outy);

cv::Mat drawCVDepth(Eigen::MatrixXf vertices, Eigen::MatrixXu faces, Camera cam);


float silhouette_iou(const cv::Mat& mask1, const cv::Mat& mask2); 

// input: mesh vertices and mesh faces forms a mesh(watertight is best) 
//        point a and b forms a line. 
// output: whether the line is in mesh. 
bool inMeshTest_cpu(const std::vector<Eigen::Vector3f>& mesh_vertices,
	const std::vector<Eigen::Vector3u>& mesh_faces,
	const Eigen::Vector3f& a, const Eigen::Vector3f& b); 