#pragma once 

#include <vector>
#include <Eigen/Eigen>
#include "../utils/image_utils.h" 
#include "../utils/math_utils.h" 

using std::vector; 


// get balls and sticks with joints and bones 
void GetBallsAndSticks(
    const vector<Eigen::Vector3d>& joints, 
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

// // get balls with proposals 
// void GetBalls(
//     const vector<vector<ConcensusData> > & data, 
//     const vector<int>& m_kpt_color_id, 
//     std::vector<Eigen::Vector3f>& balls, 
//     std::vector<float> & sizes, 
//     std::vector<int>& color_ids
// ); 


std::vector<Eigen::Vector3f> getColorMapEigen(std::string cm_type, bool reorder=false);