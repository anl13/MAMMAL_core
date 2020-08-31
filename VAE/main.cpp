#include "decoder.h"
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen>  
#include <string> 
#include <sstream> 
#include "unittest.h"

/* 2020. 08. 28
In float mode, you need to set alpha=0.001 to pass unittest for numeric
jacobi. Because float has limited precision compared with double, so you 
could not use to small alpha. 
*/

void main()
{
	//unittest_forward();
	//unittest_backward();
	unittest_numeric(); 

	system("pause");
	return; 
}