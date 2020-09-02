#pragma once
#include <string> 

void run_pose_render(); 

// custom_pig.cpp
int run_on_sequence();

// run_shape.cpp
int run_shape();
int run_pose(); 

// saml_fitting.cpp
int removeSmalTail();
int ComputeSymmetry();

int ComputePigModelSym();


void annotate(); 

void testfunc();

int modify_model(); 

int write_video();