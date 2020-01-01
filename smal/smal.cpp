#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Eigen>

#include "smal.h"
#include "../utils/math_utils.h"
#include "../utils/colorterminal.h"

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
		m_jointsOrigin.resize(3, m_jointNum);
		m_jointsShaped.resize(3, m_jointNum);
		m_jointsFinal.resize(3, m_jointNum);

		for (int i = 0; i < m_jointNum; i++)
		{
			jointsFile >> m_jointsOrigin(0, i) >> m_jointsOrigin(1, i) >> m_jointsOrigin(2, i);
		}
		jointsFile.close();
        std::cout << "...jointsLoaded." << std::endl; 
	}

	// load vertices
	{
		m_verticesOrigin.resize(3, m_vertexNum);
		m_verticesShaped.resize(3, m_vertexNum);
		m_verticesFinal.resize(3, m_vertexNum);

		for (int i = 0; i < m_vertexNum; i++)
		{
			verticesFile >> m_verticesOrigin(0, i) >> m_verticesOrigin(1, i) >> m_verticesOrigin(2, i);
		}
		verticesFile.close();
        std::cout << "...verticesLoaded." << std::endl;
	}

	// load faces
	{
		m_faces.resize(3, m_faceNum);
		for (int i = 0; i < m_faceNum; i++)
		{
            double f1, f2, f3;
			facesFile >> f1 >> f2 >> f3; 
            m_faces(0, i) = (int)f1; 
            m_faces(1, i) = (int)f2; 
            m_faces(2, i) = (int)f3;
		}
		facesFile.close();
        std::cout << "...facesLoaded." << std::endl; 
	}

	// load m_parent map
	{
		m_parent.resize(m_jointNum);
		for (int i = 0; i < m_jointNum; i++)
		{
            double p;
            parentFile >> p;
            if(p < 0 || p > 33) p = -1; 
			m_parent(i) = int(p);
		}
        std::cout << "...kintree_tableLoaded." << std::endl;
	}

	// load weights
	{
		m_lbsweights.resize(m_jointNum, m_vertexNum);
		for (int i = 0; i < m_vertexNum; i++)
		{
			for (int j = 0; j < m_jointNum; j++)
			{
				weightsFile >> m_lbsweights(j, i);
			}
		}
        std::cout << "...weightsLoaded." << std::endl; 
	}

	// load jregressor
	{
		m_jregressor.resize(m_vertexNum, m_jointNum);
		m_jregressor.setZero();
		
		double jregressorRow, jregressorCol;
		double jregressorValue;

		while(true)
		{
			jregressorFile >> jregressorCol; 
			if(jregressorFile.eof()) break; 
            jregressorFile >> jregressorRow >> jregressorValue;
			m_jregressor(int(jregressorRow), int(jregressorCol)) = jregressorValue;
		}
		jregressorFile.close();
        std::cout << "...jointRegressorLoaded." << std::endl; 
	}

	// load shape blending param
	{
		m_shapeBlendV.resize(3 * m_vertexNum, m_shapeNum);
		for (int i = 0; i < 3 * m_vertexNum; i++)
		{
			for (int j = 0; j < m_shapeNum; j++)
			{
				shapeblendFile >> m_shapeBlendV(i, j);
			}
		}
        std::cout << "...shapedirsLoaded." << std::endl; 
	}

	// calculate shape2joint regressor
	m_shapeBlendJ.resize(3 * m_jointNum, m_shapeNum);
	for (int i = 0; i < m_shapeNum; i++)
	{
		const Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> shapeBlendCol(m_shapeBlendV.col(i).data(), 3, m_vertexNum);
		Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> shape2JointCol(m_shapeBlendJ.col(i).data(), 3, m_jointNum);
		shape2JointCol = shapeBlendCol * m_jregressor;
	}

	m_poseParam.resize(3 * m_jointNum);
	m_shapeParam.resize(m_shapeNum);

	m_singleAffine.resize(4, 4 * m_jointNum);
	m_globalAffine.resize(4, 4 * m_jointNum);

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
	m_singleAffine.setZero(); 
	for (int jointId = 0; jointId < jointCount; jointId++)
	{
		const Eigen::Vector3d& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
		Eigen::Matrix4d matrix;
		matrix.setIdentity();

		matrix.block<3, 3>(0, 0) = GetRodrigues(pose);
		if (jointId == 0)
			matrix.block<3, 1>(0, 3) = m_jointsShaped.col(jointId) + m_translation;
		else
			matrix.block<3, 1>(0, 3) = m_jointsShaped.col(jointId) - m_jointsShaped.col(m_parent(jointId));

		m_singleAffine.block<4,4>(0, 4 * jointId) = matrix; 
	}
}


void SMAL::UpdateGlobalAffine(const int jointCount)
{
	m_globalAffine.setZero(); 
	for (int jointId = 0; jointId < jointCount; jointId++)
	{
		if (jointId == 0)
			m_globalAffine.block<4, 4>(0, 4 * jointId) = m_singleAffine.block<4, 4>(0, 4 * jointId);
		else
			m_globalAffine.block<4, 4>(0, 4 * jointId) = m_globalAffine.block<4, 4>(0, 4 * m_parent(jointId))*m_singleAffine.block<4, 4>(0, 4 * jointId);
	}
}


void SMAL::UpdateJointsShaped()
{
	Eigen::VectorXd jointsOffset = m_shapeBlendJ * m_shapeParam;
	m_jointsShaped = m_jointsOrigin + Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(jointsOffset.data(), 3, m_jointNum);
}


void SMAL::UpdateJointsFinal(const int jointCount)
{
	for (int jointId = 0; jointId < jointCount; jointId++)
	{
		m_jointsFinal.col(jointId) = m_globalAffine.block<3, 1>(0, 4 * jointId + 3);
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
	Eigen::VectorXd verticesOffset = m_shapeBlendV * m_shapeParam;
	m_verticesShaped = m_verticesOrigin + Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(verticesOffset.data(), 3, m_vertexNum);
}


void SMAL::UpdateVerticesFinal()
{
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> globalAffineNormalized = m_globalAffine;
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		globalAffineNormalized.block<3, 1>(0, jointId * 4 + 3) -= (m_globalAffine.block<3, 3>(0, jointId * 4)*m_jointsShaped.col(jointId));
	}

	for (int vertexId = 0; vertexId < m_vertexNum; vertexId++)
	{
		Eigen::Matrix<double, 4, 4, Eigen::ColMajor> globalAffineAverage;
		Eigen::Map<Eigen::VectorXd>(globalAffineAverage.data(), 16)
			= Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(globalAffineNormalized.data(), 16, m_jointNum) * (m_lbsweights.col(vertexId) );
		m_verticesFinal.col(vertexId) = globalAffineAverage.block<3, 4>(0, 0)*(m_verticesShaped.col(vertexId).homogeneous());
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
	for (int i = 0; i < m_vertexNum; i++)
	{
		f << "v " << m_verticesFinal(0, i) << " " << m_verticesFinal(1, i) << " " << m_verticesFinal(2, i) << std::endl;
	}

	for (int i = 0; i < m_faceNum; i++)
	{
		f << "f " << m_faces(0, i) + 1 << " " << m_faces(1, i) + 1 << " " << m_faces(2, i) + 1 << std::endl;
	}
	f.close();
}


void SMAL::saveState(std::string state_file)
{
    std::ofstream os(state_file); 
    if(!os.is_open())
    {
        std::cout << "cant not open " << state_file << std::endl; 
        return; 
    }
    for(int i = 0; i < 3; i++) os << m_translation(i) << std::endl; 
    for(int i = 0; i < m_jointNum; i++) for(int k = 0; k < 3; k++) os << m_poseParam(3*i+k) << std::endl; 
    for(int i = 0; i < m_shapeNum; i++) os << m_shapeParam(i) << std::endl; 
    os.close(); 
}

void SMAL::readState(std::string state_file)
{
    std::ifstream is(state_file); 
    if(!is.is_open())
    {
        std::cout << "cant not open " << state_file << std::endl; 
        return; 
    }
    for(int i = 0; i < 3; i++) is >> m_translation(i);
    for(int i = 0; i < m_jointNum; i++) for(int k = 0; k < 3; k++) is >> m_poseParam(3*i+k); 
    for(int i = 0; i < m_shapeNum; i++) is >> m_shapeParam(i); 
    is.close(); 
}
