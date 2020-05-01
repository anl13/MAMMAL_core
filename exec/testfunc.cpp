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
	
	cv::Mat sdf = computeSDF2d(mask); 
	double min, max; 
	cv::minMaxLoc(sdf, &min, &max);
	std::cout << "min: " << min << "  max: " << max << std::endl;
	cv::Mat vis = visualizeSDF2d(sdf);

	cv::namedWindow("vis", cv::WINDOW_NORMAL);
	cv::namedWindow("sdf", cv::WINDOW_NORMAL);
	cv::imshow("sdf", sdf);
	cv::imshow("vis", vis);
	cv::waitKey();

	return; 
}