#pragma once

#include "model.h"
#include "objloader.h"
#include <nanogui/vector.h>

void convert3CTo4C(
	const Model& in_m3c,
	ObjModel& out_m4c
);

void convert4CTo3C(
	const ObjModel& in_m4c,
	Model& out_m3c
);

nanogui::Matrix4f eigen2nanoM4f(const Eigen::Matrix4f& mat);
Eigen::Matrix4f nano2eigenM4f(const nanogui::Matrix4f& mat);