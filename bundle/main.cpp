#include <opencv2/opencv.hpp> 
#include <string> 
#include <vector> 
#include "calibration.h"

using std::vector; 


void test_calib()
{
	Calibrator calibrator; 
	calibrator.calib_pipeline(); 
	// calibrator.evaluate(); 
	// calibrator.interactive_mark(); 
}

int main()
{
	test_calib(); 

	return 0;
}