#pragma once 

#include <vector>
#include <Eigen/Eigen>
#include "../associate/skel.h" 
#include "../associate/image_utils.h" 
#include "../associate/math_utils.h" 

using std::vector; 


// get balls and sticks with joints and bones 
void GetBallsAndSticks(
    const PIG_SKEL& joints, 
    const std::vector<Eigen::Vector2i>& bones, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks 
    );

// get balls and sticks with joints and parents 
void GetBallsAndSticks(
    const Eigen::Matrix3Xf& joints, 
    const Eigen::VectorXi& parents, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > &sticks
); 

// get balls with proposals 
void GetBalls(
    const vector<vector<Vec3> > &proposals, 
    const vector<vector<double> >& metric, 
    const vector<vector<int> >& concensus_num, 
    const vector<int>& kpt_color_id, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<float> & sizes, 
    std::vector<int>& color_ids
); 

std::vector<Eigen::Vector3f> getColorMapEigen(std::string cm_type, bool reorder=false);
