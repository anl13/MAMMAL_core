#pragma once
#include <string> 
#include <vector> 
#include <Eigen/Eigen> 

void run_demo_20211008(); 

int run_pose_render(); 
int run_pose_render_demo(); 

// custom_pig.cpp
int run_on_sequence();

// run_shape.cpp
int run_shape();
int run_pose_smooth(); 
int run_MAMMAL_main(); 
int run_tri();
// saml_fitting.cpp
int removeSmalTail();
int ComputeSymmetry();

int ComputePigModelSym();


void annotate(); 

void testfunc();

int modify_model(); 

int write_video();

void visualize_seq(); 

void run_trajectory();

std::string get_config(std::string name="main_config");

int run_pose_bone_length();
void run_trajectory2();

void run_demo_result(); 
void run_eval_seq(); 
void run_demo_visualize_depth(); 
void run_eval_seq_loadanchor_temp();

// for nature methods application demos. 
int nm_skelrender_for_comparison();
int nm_monocolor_singlebody();
int nm_video5_freeview(); 
int nm_fig_skel_rend_demo(); 
int nm_monocolor_44_clips(); 
void nm_trajectory(); 