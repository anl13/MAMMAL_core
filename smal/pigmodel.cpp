#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Eigen>

#include "pigmodel.h"
#include "../utils/math_utils.h"
#include "../utils/colorterminal.h"
#include "../utils/obj_reader.h"
#include <json/json.h>

PigModel::PigModel(const std::string &_configfile)
{
	Json::Value root; 
	Json::CharReaderBuilder rbuilder; 
	std::string errs;
	std::ifstream instream(_configfile);
	if (!instream.is_open())
	{
		std::cout << "can not open " << _configfile << std::endl;
		exit(-1); 
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs); 
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}

	// read basic params 
	m_folder = root["folder"].asString(); 
	m_jointNum = root["joint_num"].asInt();
	m_vertexNum = root["vertex_num"].asInt(); 
	m_shapeNum = root["shape_num"].asInt(); 
	m_faceNum = root["face_num"].asInt(); 

	// read vertices 
	std::ifstream vfile(m_folder + "vertices.txt"); 
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
	std::ifstream pfile(m_folder + "parents.txt"); 
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
	m_faces.resize(3, m_faceNum); 
	std::ifstream facefile(m_folder + "faces.txt"); 
	if (!facefile.is_open())
	{
		std::cout << "facefile not open " << std::endl; 
		exit(-1); 
	}
	for (int i = 0; i < m_faceNum; i++)
	{
		facefile >> m_faces(0, i) >> m_faces(1, i) >> m_faces(2, i);
	}
	facefile.close(); 

	// read t pose joints
	m_jointsOrigin.resize(3, m_jointNum); 
	std::ifstream jfile(m_folder + "t_pose_joints.txt"); 
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

	// read skinning weights 
	std::ifstream weightsfile(m_folder + "skinning_weights.txt");
	if (!weightsfile.is_open())
	{
		std::cout << "weights file not open " << std::endl; 
		exit(-1); 
	}
	m_lbsweights.resize(m_jointNum, m_vertexNum);
	m_lbsweights.setZero(); 
	while (true)
	{
		if (weightsfile.eof()) break; 
		int row, col; 
		double value; 
		weightsfile >> row; 
		if (weightsfile.eof()) break; 
		weightsfile >> col >> value;
		m_lbsweights(row, col) = value; 
	}
	weightsfile.close(); 

	// read joint regressor 
	std::ifstream jregressorFile(m_folder + "/J_regressor.txt");
	if (!jregressorFile.is_open())
	{
		std::cout << "no regressor file" << std::endl; 
	}
	else {
		m_jregressor.resize(m_vertexNum, m_jointNum);
		m_jregressor.setZero();

		double jregressorRow, jregressorCol;
		double jregressorValue;

		while (true)
		{
			jregressorFile >> jregressorCol;
			if (jregressorFile.eof()) break;
			jregressorFile >> jregressorRow >> jregressorValue;
			m_jregressor(int(jregressorRow), int(jregressorCol)) = jregressorValue;
		}
		jregressorFile.close();
	}

	// read blendshape
	if (m_shapeNum > 0)
	{
		std::ifstream shapeblendFile(m_folder + "/shapedirs.txt");
		if (!shapeblendFile.is_open())
		{
			std::cout << "shape file not open" << std::endl;
		}
		else {
			m_shapeBlendV.resize(3 * m_vertexNum, m_shapeNum);
			for (int i = 0; i < 3 * m_vertexNum; i++)
			{
				for (int j = 0; j < m_shapeNum; j++)
				{
					shapeblendFile >> m_shapeBlendV(i, j);
				}
			}
			m_shapeBlendJ.resize(3 * m_jointNum, m_shapeNum);
			for (int i = 0; i < m_shapeNum; i++)
			{
				const Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> shapeBlendCol(m_shapeBlendV.col(i).data(), 3, m_vertexNum);
				Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> shape2JointCol(m_shapeBlendJ.col(i).data(), 3, m_jointNum);
				shape2JointCol = shapeBlendCol * m_jregressor;
			}
			shapeblendFile.close();
		}
	}


	// read texture coordinates 
	std::ifstream texfile(m_folder + "textures.txt"); 
	if (!texfile.is_open())
	{
		std::cout << "texture file not open " << std::endl; 
	}
	else {
		m_texcoords.resize(2, m_vertexNum);
		for (int i = 0; i < m_vertexNum; i++)
		{
			texfile >> m_texcoords(0, i) >> m_texcoords(1, i);
		}
		texfile.close();
	}

	// init 
	m_translation = Eigen::Vector3d::Zero(); 
	m_poseParam = Eigen::VectorXd::Zero(3 * m_jointNum); 
	if (m_shapeNum > 0) m_shapeParam = Eigen::VectorXd::Zero(m_shapeNum); 

	m_singleAffine.resize(4, 4 * m_jointNum); 
	m_globalAffine.resize(4, 4 * m_jointNum); 
	m_jointsFinal.resize(3, m_jointNum); 
	m_verticesFinal.resize(3, m_vertexNum); 
	
	m_normalOrigin.resize(3, m_vertexNum); 
	m_normalShaped.resize(3, m_vertexNum);
	m_normalFinal.resize(3, m_vertexNum); 
	m_normalOrigin.setZero();
	m_normalShaped.setZero(); 
	m_normalFinal.setZero(); 
	m_verticesDeformed.resize(3, m_vertexNum); 
	m_verticesDeformed.setZero(); 
	m_deform.resize(m_vertexNum); 
	m_deform.setZero();

	UpdateVertices();
	instream.close();
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

		matrix.block<3, 3>(0, 0) = GetRodrigues(pose);
		//matrix.block<3, 3>(0, 0) = EulerToRotRadD(pose);
		if (jointId == 0)
			matrix.block<3, 1>(0, 3) = m_jointsShaped.col(jointId) + m_translation;
		else
		{
			int p = m_parent(jointId); 
			matrix.block<3, 1>(0, 3) = m_jointsShaped.col(jointId) - m_jointsShaped.col(p); 
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
	if (m_shapeNum == 0)
	{
		m_jointsShaped = m_jointsOrigin; 
	}
	else
	{
		Eigen::VectorXd jointsOffset = m_shapeBlendJ * m_shapeParam;
		m_jointsShaped = m_jointsOrigin + Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(jointsOffset.data(), 3, m_jointNum);
	}
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
	if (m_shapeNum == 0)
	{
		m_verticesShaped = m_verticesOrigin; 
	}
	else
	{
		Eigen::VectorXd verticesOffset = m_shapeBlendV * m_shapeParam;
		m_verticesShaped = m_verticesOrigin + Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>>(verticesOffset.data(), 3, m_vertexNum);
	}
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
		m_verticesFinal.col(vertexId) = globalAffineAverage.block<3, 4>(0, 0)*(m_verticesDeformed.col(vertexId).homogeneous());
	}
}

void PigModel::UpdateVertices()
{
	UpdateJoints();
	UpdateVerticesShaped();
	UpdateVerticesDeformed(); 
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
	for (int i = 0; i < m_vertexNum; i++) os << m_deform(i) << std::endl; 
	os << m_scale;
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
	for (int i = 0; i < m_vertexNum; i++) is >> m_deform(i); 
	is >> m_scale; 
	is.close(); 
}

void PigModel::UpdateNormalOrigin()
{
	for (int fid = 0; fid < m_faceNum; fid++)
	{
		int x = m_faces(0, fid); 
		int y = m_faces(1, fid); 
		int z = m_faces(2, fid); 
		Eigen::Vector3d px = m_verticesOrigin.col(x); 
		Eigen::Vector3d py = m_verticesOrigin.col(y);
		Eigen::Vector3d pz = m_verticesOrigin.col(z); 
		Eigen::Vector3d norm = (py - px).cross(pz - px);
		m_normalOrigin.col(x) += norm;
		m_normalOrigin.col(y) += norm;
		m_normalOrigin.col(z) += norm; 
	}
	for (int i = 0; i < m_vertexNum; i++)
	{
		m_normalOrigin.col(i).normalize(); 
	}
}

void PigModel::UpdateNormalShaped()
{
	for (int fid = 0; fid < m_faceNum; fid++)
	{
		int x = m_faces(0, fid);
		int y = m_faces(1, fid);
		int z = m_faces(2, fid);
		Eigen::Vector3d px = m_verticesShaped.col(x);
		Eigen::Vector3d py = m_verticesShaped.col(y);
		Eigen::Vector3d pz = m_verticesShaped.col(z);
		Eigen::Vector3d norm = (py - px).cross(pz - px);
		m_normalShaped.col(x) += norm;
		m_normalShaped.col(y) += norm;
		m_normalShaped.col(z) += norm;
	}
	for (int i = 0; i < m_vertexNum; i++)
	{
		m_normalShaped.col(i).normalize();
	}
}

void PigModel::UpdateNormalFinal()
{
	for (int fid = 0; fid < m_faceNum; fid++)
	{
		int x = m_faces(0, fid);
		int y = m_faces(1, fid);
		int z = m_faces(2, fid);
		Eigen::Vector3d px = m_verticesFinal.col(x);
		Eigen::Vector3d py = m_verticesFinal.col(y);
		Eigen::Vector3d pz = m_verticesFinal.col(z);
		Eigen::Vector3d norm = (py - px).cross(pz - px);
		m_normalFinal.col(x) += norm;
		m_normalFinal.col(y) += norm;
		m_normalFinal.col(z) += norm;
	}
	for (int i = 0; i < m_vertexNum; i++)
	{
		m_normalFinal.col(i).normalize();
	}
}

void PigModel::UpdateVerticesDeformed()
{
	for (int i = 0; i < m_vertexNum; i++)
	{
		m_verticesDeformed.col(i) =
			m_verticesShaped.col(i) +
			m_normalShaped.col(i) * m_deform(i); 
	}
}

void PigModel::RescaleOriginVertices()
{
	if (m_scale == 0) return; 
	m_verticesOrigin = m_verticesOrigin * m_scale;
	m_jointsOrigin = m_jointsOrigin * m_scale; 
	m_shapeBlendJ = m_shapeBlendJ * m_scale; 
	m_shapeBlendV = m_shapeBlendV * m_scale;
}

void PigModel::readTexImg(std::string filename)
{
	m_texImgBody = cv::imread(filename);
	if (m_texImgBody.empty())
	{
		std::cout << filename << "  is empty!" << std::endl;
		exit(-1); 
	}
}

void PigModel::determineBodyParts()
{
	m_bodyParts.resize(m_vertexNum);
	std::string teximgfile = m_folder + "/body_red.png";
	m_texImgBody = cv::imread(teximgfile);
	if (m_texImgBody.empty())
	{
		std::cout << "texture img is empty!";
		exit(-1);
	}

	if (m_texImgBody.empty() || m_texcoords.cols() == 0)
	{
		std::cout << "no valid data!" << std::endl; 
		exit(-1);
	}
	std::cout << "determin body parts: " << std::endl; 
	int texW = m_texImgBody.cols;
	int texH = m_texImgBody.rows; 

	std::cout << "img type: " << m_texImgBody.type() << std::endl; 
	cv::Mat temp(cv::Size(texW, texH), CV_8UC3);
	for (int i = 0; i < m_vertexNum; i++)
	{
		Eigen::Vector2d t = m_texcoords.col(i); 
		int x = int(round(t(0) * texW)); 
		int y = texH - int(round(t(1) * texH));

		if (m_texImgBody.at<cv::Vec3b>(y, x) == cv::Vec3b(0,0,255))
		{
			m_bodyParts[i] = MAIN_BODY;
		}
		else
		{
			m_bodyParts[i] = OTHERS;
		}
	}
}