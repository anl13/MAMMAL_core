#pragma once 

#include <iostream> 
#include <fstream> 
#include <iomanip>
#include <Eigen/Eigen> 
#include <json/json.h> 
#include <vector> 
#include <opencv2/opencv.hpp> 
#include "math_utils.h"
#include "camera.h"
#include "image_utils.h"
#include "geometry.h" 
#include "clusterclique.h"
#include "Hungarian.h"
#include "skel.h" 

// cross view matching by RANSAC
class ConcensusData{
public: 
    ConcensusData() {
        cams.clear(); 
        ids.clear(); 
        joints2d.clear(); 
        num = 0; 
        errs.clear(); 
        metric = 0; 
        X = Eigen::Vector3d::Zero(); 
    }
    std::vector<Camera> cams; 
    std::vector<int> ids; 
    std::vector<Eigen::Vector3d> joints2d; 
    Eigen::Vector3d X; 
    int num; 
    std::vector<double> errs; 
    double metric; 
}; 

bool equal_concensus(const ConcensusData& data1, const ConcensusData& data2); 
bool equal_concensus_list(std::vector<ConcensusData> data1, std::vector<ConcensusData> data2); 
bool compare_concensus(ConcensusData data1, ConcensusData data2);