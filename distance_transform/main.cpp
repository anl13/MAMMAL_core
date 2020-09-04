#include "distance_transform_cuda.h"


#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> 
#include <vector_types.h>
#include "../utils/image_utils.h"
#include <device_launch_parameters.h>
#include "../utils/timer_util.h"
#include "../utils/show_gpu_param.h"

int main()
{
	show_gpu_param(); 

	cv::Mat raw = cv::imread("input.png");
	cv::Mat img; 
	cv::resize(raw, img, cv::Size(1920, 1080)); 
	int W = img.cols;
	int H = img.rows;

	cv::Mat img_gray; 
	cv::Mat img_gray2;
	img_gray2.create(cv::Size(W, H), CV_8UC1); 
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

	cv::Mat img_1(cv::Size(img.cols, img.rows), CV_8UC1); 

	for (int i = 0; i < img_gray.rows; i++)
	{
		for (int j = 0; j < img_gray.cols; j++)
		{
			img_1.at<uchar>(i, j) = img_gray.at<uchar>(i, j) > 0 ? 1 : 0;
			img_gray2.at<uchar>(i, j) = 255 - img_gray.at<uchar>(i, j);
		}
	}
	cv::Mat dt_cv; 
	TimerUtil::Timer<std::chrono::microseconds> tt; 
	tt.Start(); 
	dt_cv =get_dist_trans(img_gray2); 
	std::cout << "Time used on opencv: " << tt.Elapsed() / 1 << " microseconds. " << std::endl; 
	cv::Mat dt_vis; 
	cv::normalize(dt_cv, dt_vis, 0, 1.0, cv::NORM_MINMAX);

	
	uchar* img_uchar_device;
	cudaMalloc((void**)&img_uchar_device, H*W * sizeof(uchar)); 
	float* img_out_device; 
	cudaMalloc((void**)&img_out_device, H*W * sizeof(float)); 
	uchar* d_middle; 
	cudaMalloc((void**)&d_middle, H*W * sizeof(uchar)); 
	cudaMemcpy(img_uchar_device, img_gray.data, H*W * sizeof(uchar), cudaMemcpyHostToDevice); 
	
	cv::Mat white(cv::Size(W, H), CV_32FC1); 
	white.setTo(0); 
	cudaMemcpy(img_out_device, white.data, H*W * sizeof(float), cudaMemcpyHostToDevice); 

	tt.Start(); 
	for(int i= 0; i < 1; i++)
		runCUDA(img_uchar_device,d_middle, img_out_device, W, H); 
	std::cout << "Time used on gpu: " << tt.Elapsed() / 1 << std::endl; 

	cudaMemcpy(white.data, img_out_device, H*W * sizeof(float), cudaMemcpyDeviceToHost); 

	cv::Mat dt_vis_gpu;
	cv::normalize(white, dt_vis_gpu, 0, 1, cv::NORM_MINMAX); 
	cv::Mat output(cv::Size(W, H), CV_8UC1); 
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			output.at<uchar>(i, j) = (uchar)(dt_vis_gpu.at<float>(i, j)*255); 
		}
	}

	cv::Mat sdf;

	
	cv::imwrite("transform.jpg", output); 
	cv::imshow("raw", img_gray);
	cv::imshow("dt_cv", dt_vis);
	cv::imshow("dt_cv_gpu", dt_vis_gpu); 
	cv::waitKey();
	cv::destroyAllWindows();
	

	system("pause"); 
	return 0; 
}
//
//
//int main()
//{
//	show_gpu_param();
//
//	cv::Mat raw = cv::imread("input.png");
//	cv::Mat img;
//	cv::resize(raw, img, cv::Size(512, 256));
//	int W = img.cols;
//	int H = img.rows;
//
//	cv::Mat img_gray;
//	cv::Mat img_gray_inv;
//	img_gray_inv.create(cv::Size(W, H), CV_8UC1);
//	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
//
//	img_gray_inv = 255 - img_gray;
//
//	cv::Mat dt_cv;
//	TimerUtil::Timer<std::chrono::microseconds> tt;
//	tt.Start();
//	for (int i = 0; i < 100; i++)
//		dt_cv = get_dist_trans(img_gray_inv);
//	std::cout << "Time used on opencv: " << tt.Elapsed() / 100.f << " microseconds. " << std::endl;
//	cv::Mat dt_vis;
//	cv::normalize(dt_cv, dt_vis, 0, 1.0, cv::NORM_MINMAX);
//
//	//cv::Mat white(cv::Size(W, H), CV_32FC1);
//	//white.setTo(0);
//
//	//tt.Start();
//	//for (int i = 0; i < 100; i++)
//	//	euclidian_distance_transform(img.data, (float*)white.data, W, H); 
//	//std::cout << "Time used on gpu: " << tt.Elapsed() / 100.f << std::endl;
//
//	//cv::Mat dt_vis_gpu;
//	//cv::normalize(white, dt_vis_gpu, 0, 1, cv::NORM_MINMAX);
//	//cv::Mat output(cv::Size(W, H), CV_8UC1);
//	//for (int i = 0; i < H; i++)
//	//{
//	//	for (int j = 0; j < W; j++)
//	//	{
//	//		output.at<uchar>(i, j) = (uchar)(dt_vis_gpu.at<float>(i, j) * 255);
//	//	}
//	//}
//
//	//cv::imwrite("transform.jpg", output);
//	cv::imshow("raw", img_gray);
//	cv::imshow("dt_cv", dt_vis);
//	//cv::imshow("dt_cv_gpu", dt_vis_gpu);
//	cv::waitKey();
//	cv::destroyAllWindows();
//
//	return 0;
//}