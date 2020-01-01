//https://github.com/Smorodov/Multitarget-tracker/tree/master/Tracker
// Modifier: AN Liang, 2018-07-22, al17@mails.tsinghua.edu.cn 
// Thanks to Smorodov for sharing the code! 

#ifndef __HUNGARIAN_H__
#define __HUNGARIAN_H__

#include <vector>
#include <iostream>
#include <limits>
#include <time.h>
#include <Eigen/Eigen> 

// http://community.topcoder.com/tc?module=Static&d1=tutorials&d2=hungarianAlgorithm

typedef std::vector<int> assignments_t;
typedef float track_t; 
typedef std::vector<track_t> distMatrix_t;

class AssignmentProblemSolver
{
private:
	// --------------------------------------------------------------------------
	// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
	// --------------------------------------------------------------------------
	void assignmentoptimal(assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns);
	void buildassignmentvector(assignments_t& assignment, bool *starMatrix, size_t nOfRows, size_t nOfColumns);
	void computeassignmentcost(const assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows);
	void step2a(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
	void step2b(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
	void step3_5(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim);
	void step4(assignments_t& assignment, track_t *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, size_t nOfRows, size_t nOfColumns, size_t minDim, size_t row, size_t col);
	// --------------------------------------------------------------------------
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	// --------------------------------------------------------------------------
	void assignmentsuboptimal1(assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns);
	// --------------------------------------------------------------------------
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	// --------------------------------------------------------------------------
	void assignmentsuboptimal2(assignments_t& assignment, track_t& cost, const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns);

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
	track_t Solve(const distMatrix_t& distMatrixIn, size_t nOfRows, size_t nOfColumns, assignments_t& assignment, TMethod Method = optimal);
};

// AN Liang 2019-09-26: assign col to row 
std::vector<int> solveHungarian(const Eigen::MatrixXf &similarity); 
// AN Liang 2019-11-30: support double matrix
std::vector<int> solveHungarian(const Eigen::MatrixXd &similarity); 

#endif 