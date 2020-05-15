#pragma once
#include <string> 

// custom_pig.cpp
int run_on_sequence();

// run_shape.cpp
int run_shape();
int run_pose(); 

// run_render.cpp
int render_smal_test(); 
int renderScene(); 
int test_depth();
int test_nanogui();

// saml_fitting.cpp
int TestVerticesAlign();
int removeSmalTail();
int ComputeSymmetry();

// calibration 
void test_calib(std::string folder); 
void test_epipolar(std::string folder); 


void annotate(); 

void testfunc();