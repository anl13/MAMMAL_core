#include "gmmsolver.h" 

#include <vector> 
#include <iostream> 
#include <string> 
#include <sstream> 
#include <fstream> 

std::vector<Eigen::VectorXd> loadData()
{
	std::string filename = "F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\data\\samples.txt"; 
	std::ifstream is(filename); 
	if (!is.is_open())
	{
		std::cout << "Not ok. " << std::endl; 
		exit(-1); 
	}
	int sample_num = 5452; 
	int dim = 183; 
	std::vector<Eigen::VectorXd> data(sample_num, Eigen::VectorXd::Zero(dim)); 
	for (int i = 0; i < sample_num; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			is >> data[i](j); 
		}
	}
	return data; 
}

int main()
{
	GmmSolver solver; 
	auto data = loadData(); 
	solver.Set(4, 183, data); 
	solver.Solve(100); 

	system("pause"); 
	return 0; 

}