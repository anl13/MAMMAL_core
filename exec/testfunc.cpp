#include "main.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "../utils/image_utils.h"

void testfunc()
{
	
	cv::Mat img = cv::imread("C:/Users/Liang/Pictures/face.PNG");
	
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::Mat mask;
	cv::threshold(gray, mask, 100, 255, cv::THRESH_BINARY);
	cv::Mat mask0;
	mask0.create(mask.size(), CV_8UC1);
	mask0.setTo(cv::Scalar(255));
	cv::Mat mask2 = mask0 - mask; 
	
	cv::Mat sdf = computeSDF2d(mask2, 64); 
	double min, max; 
	cv::minMaxLoc(sdf, &min, &max);
	std::cout << "min: " << min << "  max: " << max << std::endl;
	cv::Mat vis = visualizeSDF2d(sdf);

	cv::Mat gradx, grady; 
	int scale = 1; 
	int delta = 0;
	cv::Sobel(sdf, gradx, CV_32F, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
	cv::Sobel(sdf, grady, CV_32F, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
	cv::Mat visGradx = visualizeSDF2d(gradx);
	cv::Mat visGrady = visualizeSDF2d(grady);

	cv::namedWindow("vis", cv::WINDOW_NORMAL);
	cv::namedWindow("sdf", cv::WINDOW_NORMAL);
	cv::namedWindow("gradx", cv::WINDOW_NORMAL); 
	cv::namedWindow("grady", cv::WINDOW_NORMAL);
	cv::imshow("sdf", sdf);
	cv::imshow("vis", vis);
	cv::imshow("gradx", visGradx);
	cv::imshow("grady", visGrady);
	cv::waitKey();

	return; 
}