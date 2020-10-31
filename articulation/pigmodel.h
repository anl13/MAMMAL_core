#pragma once
/*
Pure cpu version of articulation model. 
For algorithm demonstration. 

News: 
2020.08.28: transfer to float version. 
*/
#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "../utils/node_graph.h"
#include "../VAE/decoder.h"
#include "common.h"

class PigModel
{
public:
	PigModel(const std::string &_configfile);
	~PigModel();
	PigModel() = delete;
	PigModel(const PigModel& _) = delete;
	PigModel& operator=(const PigModel& _) = delete;

	void saveState(std::string state_file="state.txt"); 
    void readState(std::string state_file="state.txt");

	void SetPose(Eigen::VectorXf _poseParam) { m_poseParam = _poseParam; }
	void SetShape(Eigen::VectorXf _shapeParam) { m_shapeParam = _shapeParam; }
	void SetTranslation(Eigen::VectorXf _translation) { m_translation = _translation; }
	void SetScale(float _scale) { m_scale = _scale; }
	void ResetPose() { m_poseParam.setZero(); m_translation.setZero(); }
	void ResetShape() { m_shapeParam.setZero(); }
	void ResetTranslation() { m_translation.setZero(); }

	Eigen::MatrixXf GetJoints() const { return m_jointsFinal; }
	Eigen::MatrixXf GetVertices() const { return m_verticesFinal; }
	Eigen::Matrix<unsigned int,-1,-1,Eigen::ColMajor> GetFacesTex() { return m_facesTex; }
	Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> GetFacesVert() { return m_facesVert; }
	Eigen::MatrixXf GetVerticesTex()const { return m_verticesTex; }
	Eigen::Vector3f GetTranslation() { return m_translation; }
	Eigen::VectorXf GetShape() { return m_shapeParam; }
	Eigen::VectorXf GetPose() { return m_poseParam; }
	Eigen::VectorXi GetParents() {return m_parent; }
	float GetScale() { return m_scale; }
	cv::Mat getTexImg() { return m_texImgBody; }
	Eigen::MatrixXf GetTexcoords() { return m_texcoords; }
	std::string GetFolder() { return m_folder; }
	int GetVertexNum() { return m_vertexNum; }
	int GetJointNum() { return m_jointNum; }
	int GetFaceNum() { return m_faceNum; }
	Eigen::MatrixXf GetNormals() { return m_normalFinal; }
	Eigen::MatrixXf GetShapeBlendV() { return m_shapeBlendV; }
	Eigen::MatrixXf GetJRegressor() { return m_jregressor; }
	Eigen::MatrixXf GetLBSWeights() { return m_lbsweights; }
	std::vector<BODY_PART> GetBodyPart() { return m_bodyParts; }
	void UpdateVertices();
	void UpdateJoints(); 
	void UpdateNormals();

	void UpdateNormalOrigin();
	void UpdateNormalShaped();
	void UpdateNormalFinal();

	void UpdateVerticesTex(); 

	void SaveObj(const std::string& filename) const;
	
	// texture
	void readTexImg(std::string filename);

	/// only used for standalone processing
	void determineBodyPartsByWeight2(); 
		
	void InitNodeAndWarpField(); 
	void UpdateModelShapedByKNN(); // updateverticesdeformed
	void SaveWarpField(std::string filename);
	void LoadWarpField(std::string filename);

	// public methods for latent code
	void setLatent(Eigen::VectorXf _l) { m_latentCode = _l; }
	void setIsLatent(bool _is_latent) { m_isLatent = _is_latent;  }
	
public:
	Eigen::VectorXf m_latentCode; 
	Decoder m_decoder; 
	bool m_isLatent; 


	// traditional articulation model 
	int m_jointNum;// 43 / 33
	int m_vertexNum; // 1879 / 3889
	int m_shapeNum; // 0 / 41
	int m_faceNum; // 3718 / 7774
	int m_texNum; // 2176 

	cv::Mat m_texImgBody; 
	std::vector<BODY_PART> m_bodyParts; // body part label of each vertex
	std::vector<int> m_texToVert; // [texNum, vertNum], map tex indices to vert indices
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_verticesTex; 

	std::shared_ptr<NodeGraph> mp_nodeGraph; 
	Eigen::Matrix4Xf m_warpField; 

	// bone increment. defined on each joint relative to its parent. 
	std::vector<Eigen::Vector3f> m_bone_extend; 

	// shape deformation
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_normalOrigin; 
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_normalShaped;
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_normalFinal;

	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_jointsOrigin;
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_jointsShaped;
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_jointsDeformed;
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_jointsFinal;

	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_verticesOrigin;
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_verticesShaped;
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_verticesDeformed; 
	Eigen::Matrix<float, 3, -1, Eigen::ColMajor> m_verticesFinal;

	Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor> m_facesTex;
	Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor> m_facesVert;
	Eigen::Matrix<float, 2, -1, Eigen::ColMajor> m_texcoords; 

	Eigen::VectorXi m_parent;

	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> m_lbsweights;     // jointnum * vertexnum
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> m_jregressor;  // vertexnum * jointnum
	std::vector<std::vector<std::pair<int, float>>> m_jregressor_list;

	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> m_shapeBlendV;  // (vertexnum*3) * shapenum
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> m_shapeBlendJ; // (jointnum*3) * shapenum
	

	Eigen::VectorXf m_poseParam;
	Eigen::VectorXf m_shapeParam;
	Eigen::Vector3f m_translation;

	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> m_singleAffine;
	Eigen::Matrix<float, -1, -1, Eigen::ColMajor> m_globalAffine;
	float m_scale; 
	std::string m_folder; 


	void UpdateSingleAffine();
	void UpdateGlobalAffine();

	void UpdateVerticesShaped();
	void UpdateVerticesFinal();

	void UpdateJointsShaped();
	void UpdateJointsDeformed();
	inline void UpdateJointsFinal() { UpdateJointsFinal(m_jointNum); }
	void UpdateJointsFinal(const int jointCount);

};
