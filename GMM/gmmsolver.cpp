#include "gmmsolver.h"


void GmmSolver::Set(const int& _gmmNum, const int& _dimNum, const std::vector<Eigen::VectorXd>& _data)
{
	gmmNum = _gmmNum;
	dimNum = _dimNum;
	data = _data;

	gammas.resize(data.size(), gmmNum);
	weights = Eigen::VectorXd::Constant(gmmNum, 1.0 / double(gmmNum));
	means.resize(gmmNum);
	covs.resize(gmmNum);
	dets.resize(gmmNum); 

	for (int i = 0; i < gmmNum; i++)
	{
		means[i].setRandom(dimNum);
		Eigen::MatrixXd t = Eigen::MatrixXd::Random(dimNum, dimNum);
		covs[i] = t.transpose()*t;

		for (int j = 0; j < dimNum; j++)
			covs[i](j, j) = 1;
	}
	covInvs.resize(gmmNum);
}


void GmmSolver::StepE()
{
	// #pragma omp parallel for
	for (int k = 0; k < gmmNum; k++) {
		bool invertible;
		covs[k].computeInverseAndDetWithCheck(covInvs[k], dets[k], invertible);
		covInvs[k] = covs[k].inverse();
	}

	for (int i = 0; i < data.size(); i++) {
		for (int k = 0; k < gmmNum; k++) {
			double det = covs[k].determinant();
			Eigen::VectorXd xNormalized = data[i] - means[k];
			gammas(i, k) = weights[k] * 1.0 / (pow(2 * EIGEN_PI, double(dimNum)*0.5)*sqrt(abs(det)))*exp(-0.5*xNormalized.transpose()*covInvs[k] * xNormalized);
		}
		gammas.row(i).normalize();
	}
}


void GmmSolver::StepM()
{
	// #pragma omp parallel for
	for (int k = 0; k < gmmNum; k++)
	{
		double nK = gammas.col(k).sum();

		weights[k] = nK / double(data.size());

		means[k].setZero();
		for (int i = 0; i < data.size(); i++)
			means[k] += gammas(i, k)*data[i];

		means[k] /= nK;

		covs[k].setZero();
		for (int i = 0; i < data.size(); i++)
		{
			Eigen::VectorXd xNormalized = data[i] - means[k];
			covs[k] += gammas(i, k)*xNormalized*xNormalized.transpose();
		}
		covs[k] /= nK;
	}
}


void GmmSolver::Solve(const int iterTime)
{
	for (int i = 0; i < iterTime; i++)
	{
		StepE();
		StepM();

		std::cout << gammas.row(0) << std::endl << std::endl;

		for (int i = 0; i < gmmNum; i++)
			std::cout << weights[i] << " ";
		std::cout << std::endl << std::endl;
	}

}

