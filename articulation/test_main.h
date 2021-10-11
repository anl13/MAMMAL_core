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

std::vector<Eigen::VectorXf> loadData();

void test_fitting();

int test_body_part(); 

// gpu 

void test_gpu(); 

void test_compare_cpugpu(); 
void test_compare_cpugpu_jacobi(); 

void test_pointer(); 

void test_regressor(); 

int test_visdesigned(); 

int test_lay(); 
int test_leg();
int test_visanchors(); 

void test_texture(); 

void test_bone_var(); 

void test_inmeshtest(); 

int test_render_behavior(); 
void test_collision_render(); 