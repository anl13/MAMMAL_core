#include "decoder.h"
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen>  
#include <string> 
#include <sstream> 
#include "unittest.h"

void main()
{
	//unittest_forward();
	//unittest_backward();
	unittest_numeric(); 

	system("pause");
	return; 
}