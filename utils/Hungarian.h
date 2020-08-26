//https://github.com/Smorodov/Multitarget-tracker/tree/master/Tracker
// Modifier: AN Liang, 2018-07-22, al17@mails.tsinghua.edu.cn 
// Thanks to Smorodov for sharing the code! 
// Last modified: 2020-08-24, remove typedef 

#pragma once 

#include <vector>
#include <iostream>
#include <limits>
#include <time.h>
#include <Eigen/Eigen> 

// http://community.topcoder.com/tc?module=Static&d1=tutorials&d2=hungarianAlgorithm

class AssignmentProblemSolver
{
private:
	// --------------------------------------------------------------------------
	// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
	// --------------------------------------------------------------------------
	void assignmentoptimal(std::vector<int>& assignment, float& cost, const std::vector<float>& distMatrixIn, size_t nOfRows, size_t nOfColumns);
	void buildassignmentvector(std::vector<int>& assignment, bool *starMatrix, size_t nOfRows, size_t nOfColumns);
	void computeassignmentcost(const std::vector<int>& assignment, float& cost, const std::vector<float>& distMatrixIn, size_t nOfRows);
	void step2a(std::vector<int>& assignment, float *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
	void step2b(std::vector<int>& assignment, float *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
	void step3_5(std::vector<int>& assignment, float *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
	void step4(std::vector<int>& assignment, float *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim, size_t row, size_t col);
	// --------------------------------------------------------------------------
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	// --------------------------------------------------------------------------
	void assignmentsuboptimal1(std::vector<int>& assignment, float& cost, const std::vector<float>& distMatrixIn, size_t nOfRows, size_t nOfColumns);
	// --------------------------------------------------------------------------
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	// --------------------------------------------------------------------------
	void assignmentsuboptimal2(std::vector<int>& assignment, float& cost, const std::vector<float>& distMatrixIn, size_t nOfRows, size_t nOfColumns);

public:
	enum TMethod
	{
		optimal,
		many_forbidden_assignments,
		without_forbidden_assignments
	};

	AssignmentProblemSolver();
	~AssignmentProblemSolver();
    // assignment: N 
	float Solve(const std::vector<float>& distMatrixIn, size_t nOfRows, size_t nOfColumns, std::vector<int>& assignment, TMethod Method = optimal);
};

// AN Liang 2019-09-26: assign col to row 
std::vector<int> solveHungarian(const Eigen::MatrixXf &similarity); 
// AN Liang 2019-11-30: support double matrix
std::vector<int> solveHungarian(const Eigen::MatrixXd &similarity); 

