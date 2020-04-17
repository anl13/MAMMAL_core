#pragma once
#include <string> 

// custom_pig.cpp
int run_on_sequence();

// run_shape.cpp
int run_shape();

// run_render.cpp
int render_animal_skels();
int render_smal_test(); 
int renderScene(); 

// saml_fitting.cpp
int TestVerticesAlign();
int removeSmalTail();
int ComputeSymmetry();

// calibration 
void test_calib(std::string folder); 
void test_epipolar(std::string folder); 


void annotate(); 