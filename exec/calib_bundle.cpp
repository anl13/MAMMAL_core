#include <opencv2/opencv.hpp> 
#include <string> 
#include <vector> 
#include "../bundle/calibration.h"

using std::vector; 


void test_calib()
{
	Calibrator calibrator; 
	calibrator.calib_pipeline(); 
}

void test_epipolar()
{
	Calibrator calib; 
	calib.test_epipolar(); 
}

int main()
{
	test_epipolar(); 

	return 0;
}