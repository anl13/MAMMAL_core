#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Eigen>

#include "pigmodel.h"
#include "../utils/math_utils.h"
#include "../utils/colorterminal.h"
#include "../utils/obj_reader.h"
#include "../utils/model.h"
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
	m_texNum = root["tex_num"].asInt(); 
	m_isLatent = root["latent"].asBool(); 
	instream.close();
	/*
	necessary: vertices, parents, faces_vert, t_pose_joints, skinning_weights
	           body parts
	optional : textures, faces_tex, tex_to_vert, J_regressor, shapedir
	*/
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

	// read tex to vert 
	if (m_texNum > 0)
	{
		m_texToVert.resize(m_texNum);
		std::ifstream ttvfile(m_folder + "/tex_to_vert.txt");
		if (!ttvfile.is_open())
		{
			std::cout << "ttv file is not opened" << std::endl;
		}
		else {
			for (int i = 0; i < m_texNum; i++) ttvfile >> m_texToVert[i];
			ttvfile.close();
		}
	}

	// read faces tex
	if (m_texNum > 0)
	{
		m_facesTex.resize(3, m_faceNum);
		std::ifstream facetfile(m_folder + "faces_tex.txt");
		if (!facetfile.is_open())
		{
			std::cout << "face tex file not open " << std::endl; 
			exit(-1);
		}
		for (int i = 0; i < m_faceNum; i++)
		{
			facetfile >> m_facesTex(0, i) >> m_facesTex(1, i) >> m_facesTex(2, i);
		}
		facetfile.close();
	}

	// read faces vert 
	m_facesVert.resize(3, m_faceNum);
	std::ifstream facevfile(m_folder + "faces_vert.txt");
	if (!facevfile.is_open())
	{
		std::cout << "face vert not open " << std::endl; 
		exit(-1); 
	}
	for (int i = 0; i < m_faceNum; i++)
	{
		facevfile >> m_facesVert(0, i) >> m_facesVert(1, i) >> m_facesVert(2, i);
	}
	facevfile.close(); 

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
	m_weightsNoneZero.resize(m_jointNum);
	while (true)
	{
		if (weightsfile.eof()) break; 
		int row, col; 
		double value; 
		weightsfile >> row; 
		if (weightsfile.eof()) break; 
		weightsfile >> col >> value;
		m_lbsweights(row, col) = value; 
		m_weightsNoneZero[row].push_back(col); 
	}
	weightsfile.close(); 

	// read joint regressor 
	m_regressorNoneZero.clear();
	std::ifstream jregressorFile(m_folder + "/J_regressor.txt");
	if (!jregressorFile.is_open())
	{
		std::cout << "no regressor file" << std::endl; 
	}
	else {
		m_jregressor.resize(m_vertexNum, m_jointNum);
		m_jregressor.setZero();
		m_regressorNoneZero.resize(m_jointNum); 

		double jregressorRow, jregressorCol;
		double jregressorValue;

		while (true)
		{
			jregressorFile >> jregressorCol;
			if (jregressorFile.eof()) break;
			jregressorFile >> jregressorRow >> jregressorValue;
			m_jregressor(int(jregressorRow), int(jregressorCol)) = jregressorValue;
			m_regressorNoneZero[jregressorCol].push_back(jregressorRow);
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
			shapeblendFile.close();

			m_shapeBlendJ.resize(3 * m_jointNum, m_shapeNum);
			for (int i = 0; i < m_shapeNum; i++)
			{
				const Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> shapeBlendCol(m_shapeBlendV.col(i).data(), 3, m_vertexNum);
				Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> shape2JointCol(m_shapeBlendJ.col(i).data(), 3, m_jointNum);
				shape2JointCol = shapeBlendCol * m_jregressor;
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
			m_texcoords.resize(2, m_texNum);
			for (int i = 0; i < m_texNum; i++)
			{
				texfile >> m_texcoords(0, i) >> m_texcoords(1, i);
			}
			texfile.close();
		}
	}

	// read body parts 
	std::ifstream partFile(m_folder + "/body_parts.txt");
	if (!partFile.is_open())
	{
		std::cout << "no body parts file! Please run determineBodyPartsByWeight()" << std::endl;
		//determineBodyPartsByWeight();
	}
	else
	{
		m_bodyParts.resize(m_vertexNum, NOT_BODY);
		for (int i = 0; i < m_vertexNum; i++)
		{
			int temp;
			partFile >> temp;
			m_bodyParts[i] = (BODY_PART)temp;
		}
	}
	partFile.close();

	// init params
	m_translation = Eigen::Vector3d::Zero(); 
	m_poseParam = Eigen::VectorXd::Zero(3 * m_jointNum); 
	if (m_shapeNum > 0) m_shapeParam = Eigen::VectorXd::Zero(m_shapeNum); 
	m_scale = 1; 

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

	m_verticesTex.resize(3, m_texNum); 
	m_verticesTex.setZero(); 
	m_latentCode = Eigen::VectorXd::Zero(32); 

	UpdateVertices();
}


PigModel::~PigModel()
{
}


void PigModel::UpdateSingleAffine()
{
	m_singleAffine.setZero(); 
	if (m_isLatent) // use latent code to get local rotation 
	{
		m_decoder.latent = m_latentCode;
		m_decoder.forward(); 
		for (int jointid = 0; jointid < m_jointNum; jointid++)
		{
			Eigen::Matrix4d matrix; 
			matrix.setIdentity();
			for (int i = 0; i < 9; i++)
			{
				int row = i % 3; 
				int col = i / 3; 
				matrix(col, row) = m_decoder.output(9*jointid + i);
			}
			if (jointid == 0)
			{
				matrix.block<3, 1>(0, 3) = m_jointsDeformed.col(jointid) + m_translation;
			}
			else
			{
				int p = m_parent(jointid);
				matrix.block<3, 1>(0, 3) = m_jointsDeformed.col(jointid) - m_jointsDeformed.col(p);
			}
			m_singleAffine.block<4, 4>(0, 4 * jointid) = matrix;
			if (jointid == 0)
			{
				Eigen::Vector3d pose = m_poseParam.segment<3>(0); // global rotation 
				m_singleAffine.block<3, 3>(0, 0) = GetRodrigues(pose); 
			}
		}
	}
	else 
	{ // use axis-angle to get local rotation 
		for (int jointId = 0; jointId < m_jointNum; jointId++)
		{
			const Eigen::Vector3d& pose = m_poseParam.block<3, 1>(jointId * 3, 0);
			Eigen::Matrix4d matrix;
			matrix.setIdentity();

			matrix.block<3, 3>(0, 0) = GetRodrigues(pose);
			//matrix.block<3, 3>(0, 0) = EulerToRotRadD(pose);
			if (jointId == 0)
				matrix.block<3, 1>(0, 3) = m_jointsDeformed.col(jointId) + m_translation;
			else
			{
				int p = m_parent(jointId);
				matrix.block<3, 1>(0, 3) = m_jointsDeformed.col(jointId) - m_jointsDeformed.col(p);
			}
			m_singleAffine.block<4, 4>(0, 4 * jointId) = matrix;
		}
	}
}


void PigModel::UpdateGlobalAffine()
{
	m_globalAffine.setZero(); 
	for (int jointId = 0; jointId < m_jointNum; jointId++)
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
		m_jointsFinal.col(jointId) = 
			m_globalAffine.block<3, 1>(0, 4 * jointId + 3);
	}
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
	UpdateJointsShaped();
	UpdateVerticesShaped();
	if (mp_nodeGraph)
	{
		UpdateModelShapedByKNN();
	}
	else
	{
		m_verticesDeformed = m_verticesShaped;
	}
	UpdateJointsDeformed();
	UpdateSingleAffine();
	UpdateGlobalAffine();
	UpdateJointsFinal();
	UpdateVerticesFinal();
}

void PigModel::UpdateJoints()
{
	UpdateJointsShaped();
	UpdateJointsDeformed(); 
	UpdateSingleAffine(); 
	UpdateGlobalAffine(); 
	UpdateJointsFinal(); 
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
		f << "f " << m_facesVert(0, i) + 1 << " " << m_facesVert(1, i) + 1 << " " << m_facesVert(2, i) + 1 << std::endl;
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
	os << m_scale;
	for (int i = 0; i < 32; i++) os << m_latentCode(i) << std::endl; 
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
	is >> m_scale; 
	for (int i = 0; i < 32; i++) is >> m_latentCode(i);
	is.close(); 
}

void PigModel::saveShapeParam(std::string state_file)
{
	std::ofstream os(state_file);
	if (!os.is_open())
	{
		std::cout << "cant not open " << state_file << std::endl;
		return;
	}
	for (int i = 0; i < m_shapeNum; i++) os << m_shapeParam(i) << std::endl;
	os.close();
}

void PigModel::readShapeParam(std::string state_file)
{
	std::ifstream is(state_file);
	if (!is.is_open())
	{
		std::cout << "cant not open " << state_file << std::endl;
		return;
	}
	for (int i = 0; i < m_shapeNum; i++) is >> m_shapeParam(i);
	is.close();
}

void PigModel::saveScale(std::string state_file)
{
	std::ofstream os(state_file);
	if (!os.is_open())
	{
		std::cout << "cant not open " << state_file << std::endl;
		return;
	}
	os << m_scale;
	os.close();
}

void PigModel::readScale(std::string state_file)
{
	std::ifstream is(state_file);
	if (!is.is_open())
	{
		std::cout << "cant not open " << state_file << std::endl;
		return;
	}
	is >> m_scale; 
	is.close();
}

void PigModel::UpdateNormalOrigin()
{
	for (int fid = 0; fid < m_faceNum; fid++)
	{
		int x = m_facesVert(0, fid); 
		int y = m_facesVert(1, fid); 
		int z = m_facesVert(2, fid); 
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
		int x = m_facesVert(0, fid);
		int y = m_facesVert(1, fid);
		int z = m_facesVert(2, fid);
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
		int x = m_facesVert(0, fid);
		int y = m_facesVert(1, fid);
		int z = m_facesVert(2, fid);
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

void PigModel::UpdateNormals()
{
	UpdateNormalOrigin();
	UpdateNormalShaped();
	UpdateNormalFinal();
}

void PigModel::RescaleOriginVertices(double alpha)
{
	if (alpha == 0 || alpha == 1) return; 
	m_verticesOrigin = m_verticesOrigin * alpha;
	m_jointsOrigin = m_jointsOrigin * alpha; 
	if (m_shapeNum > 0)
	{
		m_shapeBlendJ = m_shapeBlendJ * alpha;
		m_shapeBlendV = m_shapeBlendV * alpha;
	}
}

void PigModel::UpdateVerticesTex()
{
	for (int i = 0; i < m_texNum; i++)
	{
		m_verticesTex.col(i) =
			m_verticesFinal.col(m_texToVert[i]);
	}
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

#if 0
void PigModel::debugStitchModel()
{
	// find stitches
	m_stitchMaps.clear(); 
	for (int i = 0; i < m_vertexNum; i++)
	{
		Eigen::Vector3d p1 = m_verticesOrigin.col(i); 
		for (int j = i + 1; j < m_vertexNum; j++)
		{
			Eigen::Vector3d p2 = m_verticesOrigin.col(j);
			if ((p1 - p2).norm() < 0.0001)
			{
				m_stitchMaps.push_back(i);
				m_stitchMaps.push_back(j); 
			}
		}
	} // 327 size
	// remvoe them 
	m_texNum = m_vertexNum;
	m_texToVert.resize(m_texNum);
	m_vertToTex.clear(); 

	int v_count = 0; 
	for (int i = 0; i < m_texNum; i++)
	{
		int corr = find_in_list(i, m_stitchMaps);
		if (corr < 0) {
			m_texToVert[i] = v_count;
			v_count++;
			m_vertToTex.push_back(i);
		}
		else
		{
			int left = corr % 2;
			int vid = m_stitchMaps[corr - left];
			if (left == 0)
			{
				m_texToVert[i] = v_count;
				v_count++;
				m_vertToTex.push_back(i); 
			}
			else {
				m_texToVert[i] = m_texToVert[vid];
			}
		}
	}
	std::ofstream os_verts(m_folder + "/vertices_stitched.txt");
	for (int i = 0; i < m_vertToTex.size(); i++)
	{
		int t_vid = m_vertToTex[i];
		os_verts << m_verticesOrigin.col(t_vid).transpose() << std::endl;
	}
	os_verts.close(); 

	std::ofstream os_map(m_folder + "/tex_to_vert.txt");
	for (int i = 0; i < m_texNum; i++)
	{
		os_map << m_texToVert[i] << "\n";
	}
	os_map.close(); 

	m_facesVert.resize(3, m_faceNum);
	for (int i = 0; i < m_faceNum; i++)
	{
		m_facesVert(0, i) = m_texToVert[m_facesTex(0, i)];
		m_facesVert(1, i) = m_texToVert[m_facesTex(1, i)];
		m_facesVert(2, i) = m_texToVert[m_facesTex(2, i)];
	}
	std::ofstream os_face_vert(m_folder + "/faces_vert.txt");
	for (int i = 0; i < m_faceNum; i++)
	{
		os_face_vert << m_facesVert(0, i) << " "
			<< m_facesVert(1, i) << " "
			<< m_facesVert(2, i) << "\n";
	}
	os_face_vert.close(); 

	// re-compute skinning weights
	std::ofstream os_skinweight(m_folder + "/skinning_weights_stitched.txt");
	for (int i = 0; i < m_jointNum; i++)
	{
		for (int j = 0; j < m_vertexNum; j++)
		{
			if (m_lbsweights(i, j) > 0)
			{
				int stitched = find_in_list(j, m_stitchMaps);
				if (stitched >= 0 && stitched%2==1)
				{
					continue;
				}
				else
				{
					os_skinweight << i << " "
						<< m_texToVert[j] << " "
						<< m_lbsweights(i, j) << "\n";
				}

			}
		}
	}
	os_skinweight.close();
}

#endif 


void PigModel::determineBodyPartsByWeight()
{
	m_bodyParts.resize(m_vertexNum, NOT_BODY);

	// artist designed model 
	std::vector<int> head = { 23 };
	std::vector<int> l_f_leg = { 13, 14, 15, 16,17,18,19,20};
	std::vector<int> r_f_leg = { 5,6,7,8,9,10,11,12 };
	std::vector<int> l_b_leg = { 55, 56, 57, 58, 59, 60, 61};
	std::vector<int> r_b_leg = { 39, 40, 41, 42, 43, 44, 45};
	std::vector<int> main_body = { 1,2,3,4, 5, 13, 38, 54};
	std::vector<int> tail = {  48, 49, 50, 51, 52, 53 };
	std::vector<int> l_ear = {31,32,33,34,35};
	std::vector<int> r_ear = {26, 27, 28, 29, 30};
	std::vector<int> jaw = { 36 };
	std::vector<int> neck = {};

	// priority 1
	std::vector<std::vector<int>> prior1 = {
		l_ear,r_ear,tail,l_f_leg,r_f_leg,jaw,main_body,head,l_b_leg,r_b_leg
	};
	std::vector<BODY_PART> prior1_name = {
		L_EAR,R_EAR,TAIL,L_F_LEG,R_F_LEG,JAW,MAIN_BODY,HEAD,L_B_LEG,R_B_LEG
	};
	for (int list_id = 0;list_id<prior1.size();list_id++)
	{
		auto list = prior1[list_id];
		for (int i = 0; i < list.size(); i++)
		{
			int part_id = list[i];
			for (int j = 0; j < m_weightsNoneZero[part_id].size(); j++)
			{
				if (m_bodyParts[m_weightsNoneZero[part_id][j]] != NOT_BODY)continue;
				m_bodyParts[m_weightsNoneZero[part_id][j]] = prior1_name[list_id];
			}
		}
	}
	for (int i = 0; i < m_vertexNum; i++)
	{
		if (m_bodyParts[i] == NOT_BODY) m_bodyParts[i] = MAIN_BODY;
	}
	
	std::ofstream os_part(m_folder + "/body_parts.txt");
	for (int i = 0; i < m_vertexNum; i++)
	{
		os_part << (int)m_bodyParts[i] << "\n";
	}
	os_part.close();
}

void PigModel::determineBodyPartsByWeight2()
{
	m_bodyParts.resize(m_vertexNum, NOT_BODY);
	std::vector<int> head = { 21, 22, 23, 24,25 };
	std::vector<int> l_f_leg = {  14, 15, 16,17,18,19,20 };
	std::vector<int> r_f_leg = { 6,7,8,9,10,11,12 };
	std::vector<int> l_b_leg = { 55, 56, 57, 58, 59, 60, 61 };
	std::vector<int> r_b_leg = { 39, 40, 41, 42, 43, 44, 45 };
	std::vector<int> main_body = { 0, 1,2,3,4, 5, 13, 38, 54 };
	std::vector<int> tail = { 46, 47, 48, 49, 50, 51, 52, 53 };
	std::vector<int> l_ear = { 31,32,33,34,35 };
	std::vector<int> r_ear = { 26, 27, 28, 29, 30 };
	std::vector<int> jaw = { 36, 37};
	std::vector<int> neck = {};

	std::vector<BODY_PART> joint_parts(62, NOT_BODY); 
	for (const int u : head) joint_parts[u] = HEAD; 
	for (const int u : l_f_leg) joint_parts[u] = L_F_LEG;
	for (const int u : r_f_leg) joint_parts[u] = R_F_LEG; 
	for (const int u : l_b_leg) joint_parts[u] = L_B_LEG; 
	for (const int u : r_b_leg) joint_parts[u] = R_B_LEG; 
	for (const int u : main_body) joint_parts[u] = MAIN_BODY; 
	for (const int u : tail) joint_parts[u] = TAIL; 
	for (const int u : l_ear) joint_parts[u] = L_EAR; 
	for (const int u : r_ear) joint_parts[u] = R_EAR; 
	for (const int u : jaw) joint_parts[u] = JAW; 
	for (const int u : neck) joint_parts[u] = NECK; 

	for (int i = 0; i < m_vertexNum; i++)
	{
		int maxid = -1; 
		double maxvalue = 0; 
		for (int jid = 0; jid < m_jointNum; jid++)
		{
			if (m_lbsweights(jid, i) > maxvalue) {
				maxid = jid; 
				maxvalue = m_lbsweights(jid, i); 
			}
		}
		if (maxid < 0) m_bodyParts[i] = NOT_BODY; 
		else m_bodyParts[i] = joint_parts[maxid];
	}
}

void PigModel::UpdateModelShapedByKNN()
{
	for (int sIdx = 0; sIdx < m_vertexNum; sIdx++)
	{
		Eigen::Matrix4d T = Eigen::Matrix4d::Zero();

		for (int i = 0; i < mp_nodeGraph->knn.rows(); i++) {
			const int ni = mp_nodeGraph->knn(i, sIdx);
			if (ni != -1)
				T += mp_nodeGraph->weight(i, sIdx) * m_warpField.middleCols(4 * ni, 4);
		}
		m_verticesDeformed.col(sIdx) = T.topLeftCorner(3, 4) * m_verticesShaped.col(sIdx).homogeneous();
	}
}

void PigModel::UpdateJointsDeformed()
{
	if (m_jregressor.cols() > 0 && !mp_nodeGraph)
	{
		m_jointsDeformed = m_verticesDeformed * m_jregressor;
	}
	else {
		m_jointsDeformed = m_jointsShaped;
	}
}

void PigModel::InitNodeAndWarpField()
{
	if (!mp_nodeGraph)
	{
		mp_nodeGraph = std::make_shared<NodeGraph>();
		mp_nodeGraph->Load(m_folder + "/node_graph.txt");
	}
	m_warpField.resize(4, 4 * mp_nodeGraph->nodeIdx.size());
	for (int i = 0; i < mp_nodeGraph->nodeIdx.size(); i++)
		m_warpField.middleCols(4 * i, 4).setIdentity();	
}

void PigModel::SaveWarpField()
{
	std::string filename = m_folder + "/warpfield.txt";
	std::ofstream os(filename);
	if (os.is_open())
	{
		os << m_warpField.transpose();
		os.close();
	}
}

void PigModel::LoadWarpField()
{
	std::string filename = m_folder + "/warpfield.txt";
	std::ifstream is(filename);
	if (is.is_open())
	{
		if (m_warpField.cols() == 0)
		{
			m_warpField.resize(4, 4 * mp_nodeGraph->nodeIdx.size());
		}

		for (int j = 0; j < 4*mp_nodeGraph->nodeIdx.size(); j++)
		{
			for (int i = 0; i < 4; i++)
			{
				is >> m_warpField(i, j);
			}
		}
	}
	UpdateVertices();
}

void PigModel::testReadJoint(std::string filename)
{
	std::ifstream jfile(filename);
	if (!jfile.is_open())
	{
		std::cout << "can not open jfile " << std::endl;
		exit(-1);
	}
	for (int i = 0; i < m_jointNum; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			jfile >> m_jointsOrigin(j, i);
		}
	}
	jfile.close();

	m_jointsFinal = m_jointsOrigin; 
	m_jointsDeformed = m_jointsOrigin;
	m_jointsShaped = m_jointsOrigin; 
}