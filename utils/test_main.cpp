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
#include <opencv2/xfeatures2d.hpp>


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
	//computeSDF2d_device(d_depth, sdf2, W, H);
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

void test_blend()
{
	cv::Mat render = cv::imread("data/render.png");
	cv::Mat raw = cv::imread("data/raw.png");
	cv::Mat blend; 

	TimerUtil::Timer<std::chrono::microseconds> tt; 
	tt.Start();
	overlay_render_on_raw_gpu(render, raw, blend); 
	std::cout << "blend gpu takes " << tt.Elapsed() << " microseconds" << std::endl; 

	tt.Start(); 
	cv::Mat blend2 = overlay_renders(raw, render, 0); 
	std::cout << "blend cpu takes " << tt.Elapsed() << " microseconds" << std::endl; 

	cv::namedWindow("blend", cv::WINDOW_NORMAL); 
	cv::namedWindow("blend2", cv::WINDOW_NORMAL); 
	
	cv::imshow("blend", blend); 
	cv::imshow("blend2", blend2); 
	cv::waitKey(); 
	cv::destroyAllWindows(); 


}

void test_sift()
{
	cv::Mat img = cv::imread("E:/sequences/20190704_morning/images/cam0/000000.jpg");

	cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create(); 
	std::vector<cv::KeyPoint> keypoints;
	siftPtr->detect(img, keypoints);

	cv::Mat output;
	cv::drawKeypoints(img, keypoints, output);
	cv::imshow("sift", output); 
	cv::waitKey(); 
	cv::destroyAllWindows(); 
	return; 
}

void test_rotation()
{
	std::vector<Eigen::Vector3f> gt_rot_vecs; 
	std::vector<Eigen::Matrix3f> gt_rot_mats; 
	std::vector<Eigen::Vector3f> gt_rot_eulers;

	// read data 
	std::ifstream input_rotvec("F:/projects/model_preprocess/designed_pig/pig_prior/c++/rot_vecs.txt");
	for (int i = 0; i < 100; i++)
	{
		Eigen::Vector3f a; 
		input_rotvec >> a(0) >> a(1) >> a(2); 
		gt_rot_vecs.push_back(a); 
	}
	input_rotvec.close(); 

	std::ifstream input_euler("F:/projects/model_preprocess/designed_pig/pig_prior/c++/rot_eulers.txt");
	for (int i = 0; i < 100; i++)
	{
		Eigen::Vector3f a; 
		input_euler >> a(0) >> a(1) >> a(2); 
		gt_rot_eulers.push_back(a); 
	}
	input_euler.close();

	std::ifstream input_mats("F:/projects/model_preprocess/designed_pig/pig_prior/c++/rot_mats.txt");
	for (int i = 0; i < 100; i++)
	{
		Eigen::Matrix3f a; 
		input_mats >> a(0, 0) >> a(0, 1) >> a(0, 2)
			>> a(1, 0) >> a(1, 1) >> a(1, 2)
			>> a(2, 0) >> a(2, 1) >> a(2, 2);
		gt_rot_mats.push_back(a); 
	}
	input_mats.close(); 

	float err_vec_to_mat = 0; 
	float err_mat_to_vec = 0; 
	float err_euler_to_mat = 0; 
	float err_mat_to_euler = 0; 
	
	// Rotvec--> Mat pass. 
	// Mat --> Rotvec pass. 
	// euler --> mat pass. 
	// mat --> euler pass. 
	for (int i = 0; i < 100; i++)
	{
		Eigen::Matrix<float, 3, 9, Eigen::ColMajor> dR1 = EulerJacobiF(gt_rot_eulers[i]); 
		Eigen::Matrix<float, 3, 9, Eigen::ColMajor> dR_num = EulerJacobiFNumeric(gt_rot_eulers[i]); 
		float err = (dR1 - dR_num).maxCoeff(); 
		if (err < 1e-6)
		{
			std::cout << "PASS. " << err << std::endl; 
		}
		else {
			std::cout << "NOT pass. " << i << " : " << err << std::endl; 
			std::cout << dR1 << std::endl; 
			std::cout << std::endl << dR_num << std::endl << std::endl; 
		}
	}
}

int main()
{
	test_rotation();

	system("pause"); 
	return 0; 
}