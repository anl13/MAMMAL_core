#pragma once

#include <vector> 
#include <Eigen/Eigen>
#include "../utils/camera.h"
#include <opencv2/opencv.hpp>

std::vector<Camera> readCameras(); 

std::vector<cv::Mat> readImgs(); 

void read_obj(std::string filename, Eigen::MatrixXf& vertices, Eigen::MatrixXu& faces);

int test_mean_pose(); 

int test_write_video(); 

int test_vae();

std::vector<Eigen::VectorXd> loadData();

void test_fitting();

int test_body_part(); 

// gpu 

void test_gpu(); 
