
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> 
#include <vector_types.h>
#include "../utils/image_utils.h"
#include <device_launch_parameters.h>
#include "../utils/timer_util.h"
#include "../utils/show_gpu_param.h"
#include "../utils/image_utils_gpu.h"

int test_distance_transform()
{
	show_gpu_param(); 

	cv::Mat raw = cv::imread("input.png");
	cv::Mat img; 
	cv::resize(raw, img, cv::Size(960, 540)); 
	int W = img.cols;
	int H = img.rows;

	cv::Mat img_gray; 
	cv::Mat img_gray2;
	
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	img_gray2 = 255 - img_gray;

	cv::Mat img_1(cv::Size(img.cols, img.rows), CV_8UC1); 
	img_1 = img_gray / 255;

	//for (int i = 0; i < img_gray.rows; i++)
	//{
	//	for (int j = 0; j < img_gray.cols; j++)
	//	{
	//		img_1.at<uchar>(i, j) = img_gray.at<uchar>(i, j) > 0 ? 1 : 0;
	//	}
	//}
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

	cudaMemcpy(img_uchar_device, img_1.data, H*W * sizeof(uchar), cudaMemcpyHostToDevice); 
	
	cv::Mat white(cv::Size(W, H), CV_32FC1); 
	white.setTo(0); 
	cudaMemcpy(img_out_device, white.data, H*W * sizeof(float), cudaMemcpyHostToDevice); 

	tt.Start(); 
	for(int i= 0; i < 1; i++)
		distanceTransform_device(img_uchar_device, img_out_device, W, H);
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

int visualize_sdf()
{
	show_gpu_param();

	cv::Mat raw = cv::imread("input.png");
	cv::Mat img;
	cv::resize(raw, img, cv::Size(960, 540));
	int W = img.cols;
	int H = img.rows;

	cv::Mat img_gray;
	cv::Mat img_gray2;

	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	img_gray2 = 255 - img_gray;

	cv::Mat img_1(cv::Size(img.cols, img.rows), CV_8UC1);
	img_1 = img_gray / 255;


	cv::Mat dt_cv;
	TimerUtil::Timer<std::chrono::microseconds> tt;
	
	dt_cv = computeSDF2d(img_gray2);
	cv::resize(dt_cv, dt_cv, cv::Size(960, 540)); 
	dt_cv = dt_cv / 2; 
	cv::Mat dx_cv, dy_cv; 
	tt.Start(); 
	computeGradient(dt_cv, dx_cv, dy_cv);
	std::cout << "Time used on opencv: " << tt.Elapsed() / 1 << " microseconds. " << std::endl;
	cv::Mat dt_vis = visualizeSDF2d(dx_cv);


	uchar* img_uchar_device;
	cudaMalloc((void**)&img_uchar_device, H*W * sizeof(uchar));
	float* img_out_device;
	cudaMalloc((void**)&img_out_device, H*W * sizeof(float));
	float* sobel_x, *sobel_y; 
	cudaMalloc((void**)&sobel_x, H*W * sizeof(float)); 
	cudaMalloc((void**)&sobel_y, H*W * sizeof(float));

	cudaMemcpy(img_uchar_device, img_1.data, H*W * sizeof(uchar), cudaMemcpyHostToDevice);

	cv::Mat white(cv::Size(W, H), CV_32FC1);
	white.setTo(0);
	cudaMemcpy(img_out_device, white.data, H*W * sizeof(float), cudaMemcpyHostToDevice);

	sdf2d_device(img_uchar_device, img_out_device, W, H);

	tt.Start(); 
	sobel_device(img_out_device, sobel_x, sobel_y, W, H);
	std::cout << "kernel: " << tt.Elapsed() << " mcs" << std::endl;
	cudaMemcpy(white.data, sobel_x, H*W * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "Time used on gpu: " << tt.Elapsed() << std::endl;


	cv::Mat dt_vis_gpu = visualizeSDF2d(white); 

	cv::imshow("raw", img_gray);
	cv::imshow("sdf_cv", dt_vis);
	cv::imshow("sdf_gpu", dt_vis_gpu);
	cv::waitKey();
	cv::destroyAllWindows();


	system("pause");
	return 0;
}

void test_iou()
{
	cv::Mat img1;
	img1.create(cv::Size(1920, 1080), CV_8UC3); 
	cv::rectangle(img1, cv::Rect(10, 10, 1000, 400), cv::Scalar(255,0,0), -1);

	cv::Mat img2; 
	img2.create(cv::Size(1920, 1080), CV_8UC3); 
	cv::rectangle(img2, cv::Rect(400, 400, 1200, 600), cv::Scalar(0, 255, 0), -1); 

	float iou1;
	TimerUtil::Timer<std::chrono::microseconds> tt;
	tt.Start(); 
	int I = 0; 
	int U = 0; 
	for (int i = 0; i < 1080; i++)
	{
		for (int j = 0; j < 1920; j++)
		{
			if (img1.at<cv::Vec3b>(i, j)[0] > 0 && img2.at<cv::Vec3b>(i, j)[1] > 0)
			{
				I += 1;
			}
			if (img1.at<cv::Vec3b>(i, j)[0] > 0 || img2.at<cv::Vec3b>(i, j)[1] > 0)
			{
				U += 1;
			}
		}
	}
	
	iou1 = I / (float)U;
	std::cout << "cpu iteration: " << tt.Elapsed() << " microseconds. " << std::endl;

	std::cout << "iou: " << iou1 << "(" << I << "/" << U << ")" <<   std::endl; 

	cv::imshow("iou", img1 + img2);
	cv::waitKey(); 
	cv::destroyAllWindows(); 
}


int main()
{
	test_iou(); 

	system("pause"); 
	return 0; 
}