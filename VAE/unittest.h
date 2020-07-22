#pragma once

#include <Eigen/Eigen> 

//Eigen::VectorXd readinput(int i);
//Eigen::MatrixXd readoutput(int sample_id); 
//void readrelu(Eigen::VectorXd& r1, Eigen::VectorXd& r2);
//bool compare_output(Eigen::MatrixXd pred, Eigen::MatrixXd gt);
//bool compare_reluout(Eigen::VectorXd pred, Eigen::VectorXd gt);
int unittest_forward(); 

int unittest_backward(); 