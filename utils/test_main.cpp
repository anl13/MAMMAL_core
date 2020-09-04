#include "test_main.h"
#include <iostream> 
#include "show_gpu_param.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h> 
#include "timer_util.h"
#include "image_utils.h"
#include "image_utils_gpu.h" 
#include "safe_call.hpp"
#include <vector_functions.h>
#include <vector_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>


int test_distance_transform()
{
	show_gpu_param();

	cv::Mat raw = cv::imread("../distance_transform/input.png");
	cv::Mat img;
	cv::resize(raw, img, cv::Size(1920, 1080));
	int W = img.cols;
	int H = img.rows;

	cv::Mat img_gray;
	cv::Mat img_gray2;
	img_gray2.create(cv::Size(W, H), CV_8UC1);
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	cv::Mat img_float(cv::Size(img.cols, img.rows), CV_32FC1);
	for (int i = 0; i < img_gray.rows; i++)
	{
		for (int j = 0; j < img_gray.cols; j++)
		{
			img_float.at<float>(i, j) = img_gray.at<uchar>(i, j) > 0 ? 1.f : 0;
			img_gray2.at<uchar>(i, j) = 255 - img_gray.at<uchar>(i, j);
		}
	}
	cv::Mat dt_cv;
	TimerUtil::Timer<std::chrono::microseconds> tt;
	tt.Start();
	dt_cv = get_dist_trans(img_gray2);
	std::cout << "Time used on opencv: " << tt.Elapsed() / 1 << " microseconds. " << std::endl;
	cv::Mat dt_vis;
	cv::normalize(dt_cv, dt_vis, 0, 1.0, cv::NORM_MINMAX);

	tt.Start();
	cv::Mat sdf = computeSDF2d(img_gray);
	std::cout << tt.Elapsed() << " microseconds for sdf computing. " << std::endl;


	float* d_depth;
	cudaMalloc((void**)&d_depth, W*H * sizeof(float));
	cudaMemcpy(d_depth, img_float.data, W*H * sizeof(float), cudaMemcpyHostToDevice);

	tt.Start();
	cv::Mat sdf2;
	computeSDF2d_device(d_depth, sdf2, W, H);
	std::cout << "non thing. " << tt.Elapsed() << std::endl; 

	tt.Start();
	cv::Mat dX, dY;
	computeGradient(sdf, dX, dY);
	std::cout << "sobel: " << tt.Elapsed() << std::endl; 

	cv::Mat visdf = visualizeSDF2d(sdf);
	cv::Mat visdf2 = visualizeSDF2d(sdf2); 

	cv::imshow("dis", visdf); 
	cv::imshow("dis2", visdf2); 
	cv::waitKey(); 
	cv::destroyAllWindows(); 

	return 0; 
}

int main()
{
	test_distance_transform(); 

	system("pause"); 
	return 0; 
}