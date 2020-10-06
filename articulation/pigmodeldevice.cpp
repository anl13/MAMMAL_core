#include "pigmodeldevice.h"
#include <json/json.h>

PigModelDevice::PigModelDevice(const std::string&_configFile)
{
	// load data 
	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(_configFile);
	if (!instream.is_open())
	{
		std::cout << "can not open " << _configFile << std::endl;
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
	m_texNum = root["tex_num"].asInt();
	instream.close();
	/*
	necessary: vertices, parents, faces_vert, t_pose_joints, skinning_weights
	body parts
	optional : textures, faces_tex, tex_to_vert, J_regressor, shapedir
	*/
	// read vertices 
	std::ifstream vfile(m_folder + "vertices.txt");
	if (!vfile.is_open()) {
		std::cout << "vfile not open" << std::endl; exit(-1);
	}
	m_host_verticesOrigin.resize(m_vertexNum); 
	for (int i = 0; i < m_vertexNum; i++)
	{
		for (int j = 0; j < 3; j++) vfile >> m_host_verticesOrigin[i](j);
	}
	vfile.close();

	// read parents 
	std::ifstream pfile(m_folder + "parents.txt");
	if (!pfile.is_open())
	{
		std::cout << "pfile not open" << std::endl;
		exit(-1);
	}
	m_host_parents.resize(m_jointNum);
	for (int i = 0; i < m_jointNum; i++)
	{
		pfile >> m_host_parents[i];
	}
	pfile.close();

	// read faces tex
	if (m_texNum > 0)
	{
		m_host_facesTex.resize(m_faceNum);
		std::ifstream facetfile(m_folder + "faces_tex.txt");
		if (!facetfile.is_open())
		{
			std::cout << "face tex file not open " << std::endl;
			exit(-1);
		}
		for (int i = 0; i < m_faceNum; i++)
		{
			facetfile >> m_host_facesTex[i](0) >> m_host_facesTex[i](1) >> m_host_facesTex[i](2);
		}
		facetfile.close();
	}

	// read faces vert 
	m_host_facesVert.resize(m_faceNum);
	std::ifstream facevfile(m_folder + "faces_vert.txt");
	if (!facevfile.is_open())
	{
		std::cout << "face vert not open " << std::endl;
		exit(-1);
	}
	for (int i = 0; i < m_faceNum; i++)
	{
		facevfile >> m_host_facesVert[i](0) >> m_host_facesVert[i](1) >> m_host_facesVert[i](2);
	}
	facevfile.close();

	// read t pose joints
	m_host_jointsOrigin.resize(m_jointNum);
	std::ifstream jfile(m_folder + "t_pose_joints.txt");
	if (!jfile.is_open())
	{
		std::cout << "can not open jfile " << std::endl;
		exit(-1);
	}
	for (int i = 0; i < m_jointNum; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			jfile >> m_host_jointsOrigin[i](j);
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
	m_host_lbsweights.resize(m_jointNum, m_vertexNum);
	m_host_lbsweights.setZero();
	while (true)
	{
		if (weightsfile.eof()) break;
		int row, col;
		double value;
		weightsfile >> row;
		if (weightsfile.eof()) break;
		weightsfile >> col >> value;
		m_host_lbsweights(row, col) = value;
	}
	weightsfile.close();

	// read joint regressor 
	std::ifstream jregressorFile(m_folder + "/J_regressor.txt");
	if (!jregressorFile.is_open())
	{
		std::cout << "no regressor file" << std::endl;
	}
	else {
		m_host_jregressor.resize(m_vertexNum, m_jointNum);
		m_host_jregressor.setZero();
		m_host_jregressor_list.resize(m_jointNum); 
		float jregressorRow, jregressorCol;
		float jregressorValue;

		while (true)
		{
			jregressorFile >> jregressorCol; // joint id 
			if (jregressorFile.eof()) break;
			jregressorFile >> jregressorRow >> jregressorValue;
			m_host_jregressor(int(jregressorRow), int(jregressorCol)) = jregressorValue;
			m_host_jregressor_list[int(jregressorCol)].push_back({ int(jregressorRow), jregressorValue });
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
			m_host_shapeBlendV.resize(3 * m_vertexNum, m_shapeNum);
			for (int i = 0; i < 3 * m_vertexNum; i++)
			{
				for (int j = 0; j < m_shapeNum; j++)
				{
					shapeblendFile >> m_host_shapeBlendV(i, j);
				}
			}
			shapeblendFile.close();

			m_host_shapeBlendJ.resize(3 * m_jointNum, m_shapeNum);
			for (int i = 0; i < m_shapeNum; i++)
			{
				const Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::ColMajor>> shapeBlendCol(m_host_shapeBlendV.col(i).data(), 3, m_vertexNum);
				Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::ColMajor>> shape2JointCol(m_host_shapeBlendJ.col(i).data(), 3, m_jointNum);
				shape2JointCol = shapeBlendCol * m_host_jregressor;
			}
		}
	}

	// read texture coordinates 
	if (m_texNum > 0)
	{
		std::ifstream texfile(m_folder + "textures.txt");
		if (!texfile.is_open())
		{
			std::cout << "texture file not open " << std::endl;
		}
		else {
			m_host_texcoords.resize(m_texNum);
			for (int i = 0; i < m_texNum; i++)
			{
				texfile >> m_host_texcoords[i](0) >> m_host_texcoords[i](1);
			}
			texfile.close();
		}
	}

	// read body parts 
	std::ifstream partFile(m_folder + "/body_parts.txt");
	if (!partFile.is_open())
	{
		std::cout << "no body parts file! Please run determineBodyPartsByWeight()" << std::endl;
		//determineBodyPartsByWeight2();
	}
	else
	{
		m_host_bodyParts.resize(m_vertexNum, NOT_BODY);
		for (int i = 0; i < m_vertexNum; i++)
		{
			int temp;
			partFile >> temp;
			m_host_bodyParts[i] = (BODY_PART)temp;
		}
	}
	partFile.close();

	// init driven params
	m_host_translation = Eigen::Vector3f::Zero();
	m_host_poseParam.resize(m_jointNum, Eigen::Vector3f::Zero()); 
	if (m_shapeNum > 0) m_host_shapeParam = Eigen::VectorXf::Zero(m_shapeNum);
	m_host_scale = 1.0f;

	m_host_localSE3.resize(m_jointNum, Eigen::Matrix4f::Identity()); 
	m_host_globalSE3.resize(m_jointNum, Eigen::Matrix4f::Identity()); 
	m_host_normalizedSE3.resize(m_jointNum, Eigen::Matrix4f::Identity()); 
	m_host_jointsPosed = m_host_jointsDeformed = m_host_jointsShaped = m_host_jointsScaled = m_host_jointsOrigin;
	m_host_verticesPosed = m_host_verticesDeformed = m_host_verticesShaped = m_host_verticesScaled = m_host_verticesOrigin; 

	// for gpu 
	m_device_lbsweights.upload(
		m_host_lbsweights.data(), m_jointNum * sizeof(float), m_vertexNum, m_jointNum
	);
	m_device_parents.upload(m_host_parents);
	m_device_verticesOrigin.upload(m_host_verticesOrigin); 
	m_device_faces.upload(m_host_facesVert); 
	m_device_bodyParts.upload(m_host_bodyParts);

	m_device_verticesPosed.upload(m_host_verticesPosed); 
	m_device_verticesDeformed.upload(m_host_verticesDeformed); 

	UpdateVertices(); 
	
}

void PigModelDevice::UpdateLocalSE3_host()
{
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		const Eigen::Vector3f& pose = m_host_poseParam[jointId];
		Eigen::Matrix4f matrix;
		matrix.setIdentity();

		matrix.block<3, 3>(0, 0) = GetRodrigues(pose);
		if (jointId == 0)
			matrix.block<3, 1>(0, 3) = m_host_jointsDeformed[jointId] + m_host_translation;
		else
		{
			int p = m_host_parents[jointId];
			matrix.block<3, 1>(0, 3) = m_host_jointsDeformed[jointId] - m_host_jointsDeformed[p];
		}
		m_host_localSE3[jointId] = matrix;
	}
}

void PigModelDevice::UpdateGlobalSE3_host()
{
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		if (jointId == 0) m_host_globalSE3[jointId] = m_host_localSE3[jointId];
		else m_host_globalSE3[jointId] =
			m_host_globalSE3[m_host_parents[jointId]] * m_host_localSE3[jointId];
	}
}

void PigModelDevice::UpdateNormalizedSE3_host()
{
	m_host_normalizedSE3 = m_host_globalSE3; 
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		m_host_normalizedSE3[jointId].block<3, 1>(0, 3) -=
			m_host_globalSE3[jointId].block<3, 3>(0, 0) * m_host_jointsDeformed[jointId]; 
	}
}

void PigModelDevice::UpdateVerticesShaped()
{
	// TODO:
	if (m_shapeNum == 0) m_host_verticesShaped = m_host_verticesScaled; 
	else
	{
		// TODO on gpu 
	}
}

void PigModelDevice::UpdateJointsPosed_host()
{
	for (int jointId = 0; jointId < m_jointNum; jointId++)
	{
		m_host_jointsPosed[jointId] = m_host_globalSE3[jointId].block<3, 1>(0, 3); 
	}
}

void PigModelDevice::UpdateJointsShaped()
{
	if (m_shapeNum == 0) m_host_jointsShaped = m_host_jointsScaled; 
	else
	{
		// TODO: update shape on gpu 
	}
}

void PigModelDevice::UpdateVerticesDeformed()
{
	m_host_verticesDeformed = m_host_verticesShaped; 
	m_device_verticesDeformed.upload(m_host_verticesDeformed); 
	// TODO: add surface deformation 
}

void PigModelDevice::UpdateJointsDeformed()
{
	m_host_jointsDeformed = m_host_jointsShaped; 
	m_device_jointsDeformed.upload(m_host_jointsDeformed);
	// TODO: add surface deformation
}

void PigModelDevice::UpdateJoints()
{
	UpdateScaled_device(); 
	UpdateJointsShaped(); 
	UpdateJointsDeformed(); 

	UpdateLocalSE3_host(); 
	UpdateGlobalSE3_host(); 
	
	UpdateJointsPosed_host(); 
}

void PigModelDevice::UpdateVertices()
{
	UpdateJoints(); 
	UpdateNormalizedSE3_host(); 
	UpdateVerticesShaped(); 
	UpdateVerticesDeformed(); 

	UpdateVerticesPosed_device(); 
}

void PigModelDevice::UpdateNormalFinal()
{
	UpdateNormalsFinal_device(); 
}

void PigModelDevice::saveObj(const std::string& filename) const
{
	std::ofstream f(filename);
	for (int i = 0; i < m_vertexNum; i++)
	{
		f << "v " << m_host_verticesPosed[i].transpose() << std::endl;
	}

	for (int i = 0; i < m_faceNum; i++)
	{
		f << "f " << m_host_facesVert[i](0) + 1 << " " << m_host_facesVert[i](1) + 1 << " " << m_host_facesVert[i](2) + 1 << std::endl;
	}
	f.close();
}


void PigModelDevice::saveState(std::string state_file)
{
	std::ofstream os(state_file);
	if (!os.is_open())
	{
		std::cout << "cant not open " << state_file << std::endl;
		return;
	}
	for (int i = 0; i < 3; i++) os << m_host_translation(i) << std::endl;
	for (int i = 0; i < m_jointNum; i++) for (int k = 0; k < 3; k++) os << m_host_poseParam[i](k) << std::endl;
	os << m_host_scale;
	os.close();
}

void PigModelDevice::readState(std::string state_file)
{
	std::ifstream is(state_file);
	if (!is.is_open())
	{
		std::cout << "cant not open " << state_file << std::endl;
		return;
	}
	for (int i = 0; i < 3; i++) is >> m_host_translation(i);
	for (int i = 0; i < m_jointNum; i++) 
		for (int k = 0; k < 3; k++) 
			is >> m_host_poseParam[i](k);
	is >> m_host_scale;
	is.close();
}

std::vector<Eigen::Vector3f> PigModelDevice::RegressJointsPosed()
{
	std::vector<Eigen::Vector3f> joints; 
	joints.resize(m_jointNum, Eigen::Vector3f::Zero()); 
	for (int i = 0; i < m_jointNum; i++)
	{
		for (int j = 0; j < m_host_jregressor_list[i].size(); j++)
		{
			int vid = m_host_jregressor_list[i][j].first; 
			float value = m_host_jregressor_list[i][j].second; 
			joints[i] += m_host_verticesPosed[vid] * value; 
		}
	}
	return joints; 
}