#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Eigen>

#include "smal.h"
#include "../associate/math_utils.h"

SMAL::SMAL(const std::string &folder)
{
	// open files
	std::ifstream jointsFile(folder + "/J.txt");
	std::ifstream verticesFile(folder + "/v_template.txt");
	std::ifstream facesFile(folder + "/f.txt");
	std::ifstream weightsFile(folder + "/weights.txt");
	std::ifstream parentFile(folder + "/kintree_table.txt");
	std::ifstream jregressorFile(folder + "/J_regressor.txt");
	std::ifstream shapeblendFile(folder + "/shapedirs.txt");

	if (!(jointsFile.is_open() && verticesFile.is_open() && facesFile.is_open() && weightsFile.is_open()
		&& parentFile.is_open() && jregressorFile.is_open() && shapeblendFile.is_open()))
	{
		std::cout << "file not exist!" << std::endl; 
        exit(-1); 
	}

	// load joints
	{
		jointsOrigin.resize(3, jointNum);
		jointsShaped.resize(3, jointNum);
		jointsFinal.resize(3, jointNum);

		for (int i = 0; i < jointNum; i++)
		{
			jointsFile >> jointsOrigin(0, i) >> jointsOrigin(1, i) >> jointsOrigin(2, i);
		}
		jointsFile.close();
        std::cout << "...jointsLoaded." << std::endl; 
	}

	// load vertices
	{
		verticesOrigin.resize(3, vertexNum);
		verticesShaped.resize(3, vertexNum);
		verticesFinal.resize(3, vertexNum);

		for (int i = 0; i < vertexNum; i++)
		{
			verticesFile >> verticesOrigin(0, i) >> verticesOrigin(1, i) >> verticesOrigin(2, i);
		}
		verticesFile.close();
        std::cout << "...verticesLoaded." << std::endl;
	}

	// load faces
	{
		faces.resize(3, faceNum);
		for (int i = 0; i < faceNum; i++)
		{
            double f1, f2, f3;
			facesFile >> f1 >> f2 >> f3; 
            faces(0, i) = (int)f1; 
            faces(1, i) = (int)f2; 
            faces(2, i) = (int)f3;
		}
		facesFile.close();
        std::cout << "...facesLoaded." << std::endl; 
	}

	// load parent map
	{
		parent.resize(jointNum);
		for (int i = 0; i < jointNum; i++)
		{
            double p;
            parentFile >> p;
            if(p < 0 || p > 33) p = -1; 
			parent(i) = int(p);
		}
        std::cout << "...kintree_tableLoaded." << std::endl;
	}

	// load weights
	{
		weights.resize(vertexNum, jointNum);
		for (int i = 0; i < vertexNum; i++)
		{
			for (int j = 0; j < jointNum; j++)
			{
				weightsFile >> weights(i, j);
			}
		}
        std::cout << "...weightsLoaded." << std::endl; 
	}

	// load jregressor
	{
		jregressor.resize(vertexNum, jointNum);
		jregressor.setZero();
		
		int jregressorRow, jregressorCol;
		double jregressorValue;

		for(int i = 0; i < 440; i++)
		{
			jregressorFile >> jregressorCol; 
            jregressorFile >> jregressorRow >> jregressorValue;
			jregressor(jregressorRow, jregressorCol) = jregressorValue;
		}
		jregressorFile.close();
        std::cout << "...jointRegressorLoaded." << std::endl; 
	}

	// load shape blending param
	{
		shapeBlend.resize(3 * vertexNum, shapeNum);
		for (int i = 0; i < 3 * vertexNum; i++)
		{
			for (int j = 0; j < shapeNum; j++)
			{
				shapeblendFile >> shapeBlend(i, j);
			}
		}
        std::cout << "...shapedirsLoaded." << std::endl; 
	}

	// calculate shape2joint regressor
	shape2Joint.resize(3 * jointNum, shapeNum);
	for (int i = 0; i < shapeNum; i++)
	{
		const Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> shapeBlendCol(shapeBlend.col(i).data(), 3, vertexNum);
		Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> shape2JointCol(shape2Joint.col(i).data(), 3, jointNum);
		shape2JointCol = shapeBlendCol * jregressor;
	}

	poseParam.resize(3 * jointNum);
	shapeParam.resize(shapeNum);

	singleAffine.resize(4, 4 * jointNum);
	globalAffine.resize(4, 4 * jointNum);

	ResetPose();
	ResetShape();
	ResetTranslation();

	UpdateVertices();

    std::cout << "SMAL model prepared. " << std::endl; 
}


SMAL::~SMAL()
{
}


void SMAL::UpdateSingleAffine(const int jointCount)
{
	for (int jointId = 0; jointId < jointCount; jointId++)
	{
		const Eigen::Vector3d& pose = poseParam.block<3, 1>(jointId * 3, 0);
		Eigen::Matrix4d matrix;
		matrix.setIdentity();

		matrix.block<3, 3>(0, 0) = GetRodrigues(pose);
		if (jointId == 0)
			matrix.block<3, 1>(0, 3) = jointsShaped.col(jointId) + translation;
		else
			matrix.block<3, 1>(0, 3) = jointsShaped.col(jointId) - jointsShaped.col(parent(jointId));

		singleAffine.block<4,4>(0, 4 * jointId) = matrix; 
	}
}


void SMAL::UpdateGlobalAffine(const int jointCount)
{
	for (int jointId = 0; jointId < jointCount; jointId++)
	{
		if (jointId == 0)
			globalAffine.block<4, 4>(0, 4 * jointId) = singleAffine.block<4, 4>(0, 4 * jointId);
		else
			globalAffine.block<4, 4>(0, 4 * jointId) = globalAffine.block<4, 4>(0, 4 * parent(jointId))*singleAffine.block<4, 4>(0, 4 * jointId);
	}
}


void SMAL::UpdateJointsShaped()
{
	Eigen::VectorXd jointsOffset = shape2Joint * shapeParam;
	jointsShaped = jointsOrigin + Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(jointsOffset.data(), 3, jointNum);
}


void SMAL::UpdateJointsFinal(const int jointCount)
{
	for (int jointId = 0; jointId < jointCount; jointId++)
	{
		jointsFinal.col(jointId) = globalAffine.block<3, 1>(0, 4 * jointId + 3);
	}
}


void SMAL::UpdateJoints()
{
	UpdateJointsShaped();
	UpdateSingleAffine();
	UpdateGlobalAffine();
	UpdateJointsFinal();
}


void SMAL::UpdateVerticesShaped()
{
	Eigen::VectorXd verticesOffset = shapeBlend * shapeParam;
	verticesShaped = verticesOrigin + Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(verticesOffset.data(), 3, vertexNum);
}


void SMAL::UpdateVerticesFinal()
{
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> globalAffineNormalized = globalAffine;
	for (int jointId = 0; jointId < jointNum; jointId++)
	{
		globalAffineNormalized.block<3, 1>(0, jointId * 4 + 3) -= (globalAffine.block<3, 3>(0, jointId * 4)*jointsShaped.col(jointId));
	}

	for (int vertexId = 0; vertexId < vertexNum; vertexId++)
	{
		Eigen::Matrix<double, 4, 4, Eigen::ColMajor> globalAffineAverage;
		Eigen::Map<Eigen::VectorXd>(globalAffineAverage.data(), 16)
			= Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(globalAffineNormalized.data(), 16, jointNum) * (weights.row(vertexId).transpose());
		verticesFinal.col(vertexId) = globalAffineAverage.block<3, 4>(0, 0)*(verticesShaped.col(vertexId).homogeneous());
	}
}


void SMAL::UpdateVertices()
{
	UpdateJoints();
	UpdateVerticesShaped();
	UpdateVerticesFinal();
}


void SMAL::SaveObj(const std::string& filename) const
{
	std::ofstream f(filename);
	for (int i = 0; i < vertexNum; i++)
	{
		f << "v " << verticesFinal(0, i) << " " << verticesFinal(1, i) << " " << verticesFinal(2, i) << std::endl;
	}

	for (int i = 0; i < faceNum; i++)
	{
		f << "f " << faces(0, i) + 1 << " " << faces(1, i) + 1 << " " << faces(2, i) + 1 << std::endl;
	}
	f.close();
}


void SMAL::CalcPoseJacobi()
{
	// calculate affine and jointsFinal
	UpdateSingleAffine();
	UpdateGlobalAffine();
	UpdateJointsFinal();

	// calculate delta rodrigues
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> rodriguesDerivative(3, 3 * 3 * jointNum);
	for (int jointId = 0; jointId < jointNum; jointId++)
	{
		const Eigen::Vector3d& pose = poseParam.block<3, 1>(jointId * 3, 0);
		rodriguesDerivative.block<3, 9>(0, 9 * jointId) = RodriguesJacobiD(pose);
	}

	// set jacobi
	poseJacobi = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(3 * jointNum, 3 + 3 * jointNum);
	for (int jointDerivativeId = 0; jointDerivativeId < jointNum; jointDerivativeId++)
	{
		// update translation term
		poseJacobi.block<3, 3>(jointDerivativeId * 3, 0).setIdentity();

		// update poseParam term
		for (int axisDerivativeId = 0; axisDerivativeId < 3; axisDerivativeId++)
		{
			std::vector<std::pair<bool, Eigen::Matrix4d>> globalAffineDerivative(jointNum, std::make_pair(false, Eigen::Matrix4d::Zero()));
			globalAffineDerivative[jointDerivativeId].first = true;
			auto& affine = globalAffineDerivative[jointDerivativeId].second;
			affine.block<3, 3>(0, 0) = rodriguesDerivative.block<3, 3>(0, 3 * (3 * jointDerivativeId + axisDerivativeId));
			affine = jointDerivativeId == 0 ? affine : (globalAffine.block<4, 4>(0, 4 * parent(jointDerivativeId)) * affine);

			for (int jointId = jointDerivativeId + 1; jointId < jointNum; jointId++)
			{
				if (globalAffineDerivative[parent(jointId)].first)
				{
					globalAffineDerivative[jointId].first = true;
					globalAffineDerivative[jointId].second = globalAffineDerivative[parent(jointId)].second * singleAffine.block<4, 4>(0, 4 * jointId);
					// update jacobi for pose
					poseJacobi.block<3, 1>(jointId * 3, 3 + jointDerivativeId * 3 + axisDerivativeId) = globalAffineDerivative[jointId].second.block<3, 1>(0, 3);
				}
			}
		}
	}
}
