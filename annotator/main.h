#pragma once
#include "NanoRenderer.h"
#include "../utils/camera.h"
#include <nanogui/nanogui.h>
#include <vector>
#include <vector_functions.hpp>

int test_datatype();
int test_depth();

int multiview_annotator(); 

nanogui::Matrix4f eigen2nanoM4f(Eigen::Matrix4f mat);
std::vector<Camera> readCameras();
std::vector<float4> getColorMapFloat4(std::string cm_type);