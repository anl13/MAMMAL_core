#pragma once
#include "NanoRenderer.h"
#include "../utils/camera.h"
#include <nanogui/nanogui.h>
#include <vector>
#include <vector_functions.hpp>
#include "../utils/image_utils.h"
#include "../utils/skel.h"


int multiview_annotator(); 

cv::Rect expand_box(const DetInstance& det); 
//
nanogui::Matrix4f eigen2nanoM4f(Eigen::Matrix4f mat);
//std::vector<Camera> readCameras();
std::vector<float4> getColorMapFloat4(std::string cm_type);