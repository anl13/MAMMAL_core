#pragma once 

#include <vector>
#include <Eigen/Eigen>
#include "../associate/framedata.h" 
#include "../associate/image_utils.h"

void GetBallsAndSticks(
    const PIG_SKEL& joints, 
    const std::vector<Eigen::Vector2i>& bones, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks 
    );

void GetBallsAndSticks(
    const Eigen::Matrix3Xf& joints, 
    const Eigen::VectorXi& parents, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > &sticks
); 

std::vector<Eigen::Vector3f> getColorMapEigen();
