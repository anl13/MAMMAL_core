#include <opencv2/opencv.hpp> 
#include <string> 
#include <vector> 
#include "../bundle/calibration.h"
#include "main.h"

using std::vector; 


void test_calib(std::string folder)
{
	Calibrator calibrator(folder); 
	calibrator.calib_pipeline(); 
}

void test_epipolar(std::string folder)
{
	Calibrator calib(folder); 
	calib.test_epipolar(); 
}
