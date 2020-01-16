#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Eigen>

#include "pigmodel.h"
#include "../utils/math_utils.h"
#include "../utils/colorterminal.h"
#include "../utils/obj_reader.h"

PigModel::PigModel(const std::string &folder)
{
	// read vertices 
	std::ifstream vfile(folder + "vertices"); 
	if(!vfile.is_open()){
		std::cout << "vfile not open" << std::endl; exit(-1); 
	}
	m_verticesOrigin.resize(3,m_vertexNum);
	for(int i = 0; i < m_vertexNum; i++)
	{
		for(int j = 0; j < 3; j++) vfile >> m_verticesOrigin(j,i); 
	}
	vfile.close(); 

    // read parents 
	std::ifstream pfile(folder + "parents.txt"); 
	if(!pfile.is_open())
	{
		std::cout << "pfile not open" << std::endl; 
		exit(-1); 
	}
	m_parent.resize(m_jointNum); 
	for(int i = 0; i < m_jointNum; i++) 
	{
		pfile >> m_parent(i); 
	}
	pfile.close(); 

	// read faces 
	OBJReader reader; 
	reader.read(folder+"pig_tpose.obj"); 
	std::cout << "v num: " << reader.vertices.size() << std::endl; 
	std::cout << "f num: " << reader.faces_v.size() << std::endl;
	m_faces = reader.faces_v_eigen; 
    std::cout << "PigModel model prepared. " << std::endl; 

	// m_verticesFinal = m_verticesOrigin / 100; 
	m_verticesFinal = reader.vertices_eigen.cast<double>() / 100; 
	Eigen::Vector3d mean = m_verticesFinal.rowwise().mean(); 
	m_verticesFinal = m_verticesFinal.colwise() - mean; 

	// load pose 
	m_poseParam.resize(3 * m_jointNum); 
	std::ifstream rfile(folder + "rots.txt"); 
	for (int i = 0; i < 3 * m_jointNum; i++) rfile >> m_poseParam(i); 
	m_poseParam = m_poseParam * PI / 180; 

	// read translations ? joints ? 
	m_jointsOrigin.resize(3, m_jointNum); 
	std::ifstream jfile(folder + "joint.txt"); 
	if(!jfile.is_open()) 
	{
		std::cout << "can not open jfile " << std::endl; 
		exit(-1); 
	}
	for(int i = 0; i < m_jointNum; i++)
	{
		for(int j = 0; j < 3; j++)
		{
			jfile >> m_jointsOrigin(j,i); 
		}
	}
	jfile.close();
	m_jointsShaped = m_jointsOrigin / 100; 
	
	std::cout << "joints: " << std::endl << m_jointsOrigin.transpose() << std::endl; 
	m_translation = Eigen::Vector3d::Zero(); 
	m_singleAffine.resize(4, 4 * m_jointNum); 
	m_globalAffine.resize(4, 4 * m_jointNum); 
	m_jointsFinal.resize(3, m_jointNum); 
	UpdateSingleAffine(); 
	UpdateGlobalAffine(); 
	UpdateJointsFinal(); 
}


PigModel::~PigModel()
{
}


void PigModel::UpdateSingleAffine(const int jointCount)
{
	m_singleAffine.setZero(); 
	for (int jointId = 0; jointId < jointCount; jointId++)
	{
		const Eigen::Vector3d& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
		Eigen::Matrix4d matrix;
		matrix.setIdentity();

		//matrix.block<3, 3>(0, 0) = GetRodrigues(pose);
		matrix.block<3, 3>(0, 0) = EulerToRotRadD(pose); 
		if (jointId == 0)
			matrix.block<3, 1>(0, 3) = m_jointsShaped.col(jointId) + m_translation;
		else
		{
			
			matrix.block<3, 1>(0, 3) = m_jointsShaped.col(jointId); 
		}

		m_singleAffine.block<4,4>(0, 4 * jointId) = matrix; 
	}
}


void PigModel::UpdateGlobalAffine(const int jointCount)
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


void PigModel::UpdateJointsShaped()
{
	Eigen::VectorXd jointsOffset = m_shapeBlendJ * m_shapeParam;
	m_jointsShaped = m_jointsOrigin + Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(jointsOffset.data(), 3, m_jointNum);
}


void PigModel::UpdateJointsFinal(const int jointCount)
{
	for (int jointId = 0; jointId < jointCount; jointId++)
	{
		m_jointsFinal.col(jointId) = m_globalAffine.block<3, 1>(0, 4 * jointId + 3);
	}
}


void PigModel::UpdateJoints()
{
	UpdateJointsShaped();
	UpdateSingleAffine();
	UpdateGlobalAffine();
	UpdateJointsFinal();
}


void PigModel::UpdateVerticesShaped()
{
	Eigen::VectorXd verticesOffset = m_shapeBlendV * m_shapeParam;
	m_verticesShaped = m_verticesOrigin + Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(verticesOffset.data(), 3, m_vertexNum);
}


void PigModel::UpdateVerticesFinal()
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


void PigModel::UpdateVertices()
{
	UpdateJoints();
	UpdateVerticesShaped();
	UpdateVerticesFinal();
}


void PigModel::SaveObj(const std::string& filename) const
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


void PigModel::saveState(std::string state_file)
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

void PigModel::readState(std::string state_file)
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
