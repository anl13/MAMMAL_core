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
	m_texNum = root["tex_num"].asInt(); 
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
			exit(-1);
		}
		for (int i = 0; i < m_texNum; i++) ttvfile >> m_texToVert[i];
		ttvfile.close();
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
	m_deform = Eigen::VectorXd::Zero(m_vertexNum);
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

	UpdateVertices();
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
	if (mp_nodeGraph)
	{
		UpdateModelShapedByKNN();
	}
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


void PigModel::saveDeform(std::string state_file)
{
	std::ofstream os(state_file);
	if (!os.is_open())
	{
		std::cout << "cant not open " << state_file << std::endl;
		return;
	}
	for (int i = 0; i < m_vertexNum; i++) os << m_deform(i) << std::endl;
	os.close();
}

void PigModel::readDeform(std::string state_file)
{
	std::ifstream is(state_file);
	if (!is.is_open())
	{
		std::cout << "cant not open " << state_file << std::endl;
		return;
	}
	for (int i = 0; i < m_vertexNum; i++) is >> m_deform(i);
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

void PigModel::UpdateVerticesDeformed()
{
	UpdateNormalShaped(); 
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
	if (m_shapeNum > 0)
	{
		m_shapeBlendJ = m_shapeBlendJ * m_scale;
		m_shapeBlendV = m_shapeBlendV * m_scale;
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

void PigModel::determineBodyPartsByTex()
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
	for (int i = 0; i < m_texNum; i++)
	{
		Eigen::Vector2d t = m_texcoords.col(i); 
		int vid = m_texToVert[i];
		int x = int(round(t(0) * texW)); 
		int y = texH - int(round(t(1) * texH));

		if (m_texImgBody.at<cv::Vec3b>(y, x) == cv::Vec3b(0,0,255))
		{
			m_bodyParts[vid] = MAIN_BODY;
		}
		else
		{
			m_bodyParts[vid] = OTHERS;
		}
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



void PigModel::debugRemoveEye()
{
	OBJReader reader;
	reader.read("D:/Projects/animal_calib/data/pig_model_noeye/pig_noeye.obj");
	int vnum = reader.vertices.size(); 
	std::cout << "vnum: " << vnum << std::endl;

	std::vector<int> mapping;
	mapping.resize(vnum);
	std::vector<int> reverse(m_vertexNum, -1);
	for (int i = 0; i < vnum; i++)
	{
		Eigen::Vector3d x = reader.vertices[i].cast<double>();
		for (int j = 0; j < m_vertexNum; j++)
		{
			Eigen::Vector3d y = m_verticesOrigin.col(j);
			double dist = (x - y).norm(); 
			if (dist < 0.00001)
			{
				mapping[i] = j;
				reverse[j] = i;
			}
		}
	}
	// check which points are removed 
	for (int i = 0; i < m_vertexNum; i++)
	{
		if (reverse[i] < 0) std::cout << i << " ";
	}
	std::cout << std::endl; 
	// save 
	std::string folder = "D:/Projects/animal_calib/data/pig_model_noeye/";
	std::ofstream os_v(folder + "vertices_stitched.txt");
	for (int i = 0; i < vnum; i++)
	{
		os_v << reader.vertices[i].transpose() << "\n";
	}
	os_v.close(); 

	std::ofstream os_fv(folder + "faces_vert.txt");
	for (int i = 0; i < m_faceNum; i++)
	{
		int v1 = m_facesVert(0, i);
		int v2 = m_facesVert(1, i);
		int v3 = m_facesVert(2, i);
		if (reverse[v1] < 0 || reverse[v2] < 0 || reverse[v3] < 0)continue;
		os_fv << reverse[v1] << " " << reverse[v2] << " " << reverse[v3] << "\n";
	}
	os_fv.close(); 

	std::vector<int> t_to_v; 
	std::vector<int> t_map;
	std::vector<int> t_rev(m_texNum, -1);
	for (int i = 0; i < m_texNum; i++)
	{
		int left = m_texToVert[i];
		if (reverse[left] < 0) continue;
		t_rev[i] = t_map.size();
		t_map.push_back(i);
	}
	int tnum = t_map.size();
	t_to_v.resize(tnum, -1);
	int count = vnum;
	for (int i = 0; i < tnum; i++)
	{
		int t1 = t_map[i];
		int v1 = m_texToVert[t1];
		int n1 = reverse[v1];
		t_to_v[i] = n1;
	}
	std::ofstream os_t_to_v(folder + "tex_to_vert.txt");
	for (int i = 0; i < tnum; i++)os_t_to_v << t_to_v[i] << "\n";
	os_t_to_v.close();
	std::ofstream os_tex(folder + "textures.txt");
	for (int i = 0; i < tnum; i++)
	{
		int tn = t_map[i];
		Eigen::Vector2d tex = m_texcoords.col(tn);
		os_tex << tex(0) << " " << tex(1) << "\n";
	}
	os_tex.close();
	std::ofstream os_ft(folder + "faces_tex.txt");
	for (int i = 0; i < m_faceNum; i++)
	{
		int t1 = m_facesTex(0, i);
		int t2 = m_facesTex(1, i);
		int t3 = m_facesTex(2, i);
		if (t_rev[t1] < 0 || t_rev[t2] < 0 || t_rev[t3] < 0)continue;
		os_ft << t_rev[t1] << " " << t_rev[t2] << " " << t_rev[t3] << "\n";
	}
	os_ft.close();
	std::ofstream os_skin(folder + "skinning_weights_stitched.txt");
	for (int i = 0; i < m_jointNum; i++)
	{
		for (int j = 0; j < m_vertexNum; j++)
		{
			if (reverse[j] < 0)continue;
			if (m_lbsweights(i, j) == 0)continue;
			int vid = reverse[j];
			
			os_skin << i << " " << vid << " " << m_lbsweights(i, j) << "\n";
		}
	}
	os_skin.close();
	for (int i = 0; i < m_vertexNum; i++)
	{
		if (reverse[i] < 0)continue;
		Eigen::VectorXd x = m_lbsweights.col(i);
		double w = x.sum();
		if (w > 0)std::cout << w << std::endl; 
	}
}

#endif  

void PigModel::determineBodyPartsByWeight()
{
	m_bodyParts.resize(m_vertexNum, NOT_BODY);
	// pig model
	//std::vector<int> head = { 14 };
	//std::vector<int> l_f_leg = { 21,22,23,24,25 };
	//std::vector<int> r_f_leg = { 6, 7,8,9,10 };
	//std::vector<int> l_b_leg = { 40,41,42 };
	//std::vector<int> r_b_leg = { 28,29,30 };
	//std::vector<int> main_body = { 0,1,2,3,4 };
	//std::vector<int> tail = { 31,32,33,34,35,36,37 };
	//std::vector<int> l_ear = { 19 };
	//std::vector<int> r_ear = { 17 };
	//std::vector<int> jaw = { 15 };
	//std::vector<int> neck = { 11 };
	std::vector<int> head = { 16 };
	std::vector<int> l_f_leg = { 9,10 };
	std::vector<int> r_f_leg = { 13,14 };
	std::vector<int> l_b_leg = { 18,19,20 };
	std::vector<int> r_b_leg = { 22,23,24 };
	std::vector<int> main_body = { 1,2,3,4,5,6 };
	std::vector<int> tail = { 26,27,28,29,30,31 };
	std::vector<int> l_ear = {  };
	std::vector<int> r_ear = {  };
	std::vector<int> jaw = { 32 };
	std::vector<int> neck = { };
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
		m_verticesShaped.col(sIdx) = T.topLeftCorner(3, 4) * m_verticesShaped.col(sIdx).homogeneous();
		m_normalShaped.col(sIdx) = T.topLeftCorner(3, 3) * m_normalShaped.col(sIdx);
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