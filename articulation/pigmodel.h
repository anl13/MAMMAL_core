#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "../utils/node_graph.h"
#include "../VAE/decoder.h"

enum BODY_PART
{
	NOT_BODY = 0,
	MAIN_BODY,
	HEAD,
	L_EAR,
	R_EAR,
	L_F_LEG,
	R_F_LEG,
	L_B_LEG,
	R_B_LEG,
	TAIL,
	JAW,
	NECK,
	OTHERS
};

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
	void saveShapeParam(std::string state_file = "shape.txt");
	void readShapeParam(std::string state_file = "shape.txt");
	void saveScale(std::string state_file = "scale.txt");
	void readScale(std::string state_file = "scale.txt");

	void SetPose(Eigen::VectorXd _poseParam) { m_poseParam = _poseParam; }
	void SetShape(Eigen::VectorXd _shapeParam) { m_shapeParam = _shapeParam; }
	void SetTranslation(Eigen::VectorXd _translation) { m_translation = _translation; }
	void SetScale(double _scale) { m_scale = _scale; }
	void ResetPose() { m_poseParam.setZero();}
	void ResetShape() { m_shapeParam.setZero(); }
	void ResetTranslation() { m_translation.setZero(); }

	Eigen::MatrixXd GetJoints() const { return m_jointsFinal; }
	Eigen::MatrixXd GetVertices() const { return m_verticesFinal; }
	Eigen::Matrix<unsigned int,-1,-1,Eigen::ColMajor> GetFacesTex() { return m_facesTex; }
	Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> GetFacesVert() { return m_facesVert; }
	Eigen::MatrixXd GetVerticesTex()const { return m_verticesTex; }
	Eigen::Vector3d GetTranslation() { return m_translation; }
	Eigen::VectorXd GetShape() { return m_shapeParam; }
	Eigen::VectorXd GetPose() { return m_poseParam; }
	Eigen::VectorXi GetParents() {return m_parent; }
	double GetScale() { return m_scale; }
	cv::Mat getTexImg() { return m_texImgBody; }
	Eigen::MatrixXd GetTexcoords() { return m_texcoords; }
	std::string GetFolder() { return m_folder; }
	int GetVertexNum() { return m_vertexNum; }
	int GetJointNum() { return m_jointNum; }
	int GetFaceNum() { return m_faceNum; }
	Eigen::MatrixXd GetShapeBlendV() { return m_shapeBlendV; }
	Eigen::MatrixXd GetJRegressor() { return m_jregressor; }
	Eigen::MatrixXd GetLBSWeights() { return m_lbsweights; }
	std::vector<std::vector<int> > GetWeightsNoneZero() { return m_weightsNoneZero; }
	std::vector<std::vector<int> > GetRegressorNoneZero() { return m_regressorNoneZero; }
	std::vector<BODY_PART> GetBodyPart() { return m_bodyParts; }
	void UpdateVertices();
	void UpdateNormals();

	void UpdateNormalOrigin();
	void UpdateNormalShaped();
	void UpdateNormalFinal();

	void RescaleOriginVertices(double alpha);
	void UpdateVerticesTex(); 

	void SaveObj(const std::string& filename) const;
	
	// texture
	void readTexImg(std::string filename);

	/// only used for standalone processing
	void determineBodyPartsByWeight(); 
	
	// for debug:stitching
#if 0
	void debugStitchModel();
	std::vector<int> m_stitchMaps;
	std::vector<int> m_texToVert;
	std::vector<int> m_vertToTex;
	void debugRemoveEye();
#endif 
	void remapmodel();
	
	void InitNodeAndWarpField(); 
	void UpdateModelShapedByKNN();
	void SaveWarpField();
	void LoadWarpField();

	void testReadJoint(std::string filename);

	// public methods for latent code
	void setLatent(Eigen::VectorXd _l) { m_latentCode = _l; }
	void setIsLatent(bool _is_latent) { m_isLatent = _is_latent;  }

protected:
	Eigen::VectorXd m_latentCode; 
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
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesTex; 
	std::shared_ptr<NodeGraph> mp_nodeGraph; 
	Eigen::Matrix4Xd m_warpField; 

	// shape deformation
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_normalOrigin; 
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_normalShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_normalFinal;

	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsOrigin;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsDeformed;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsFinal;

	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesOrigin;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesDeformed; 
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesFinal;

	Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor> m_facesTex;
	Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor> m_facesVert;
	Eigen::Matrix<double, 2, -1, Eigen::ColMajor> m_texcoords; 

	Eigen::VectorXi m_parent;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_lbsweights;     // jointnum * vertexnum
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jregressor;  // vertexnum * jointnum
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_shapeBlendV;  // (vertexnum*3) * shapenum
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_shapeBlendJ; // (jointnum*3) * shapenum
	std::vector<std::vector<int> > m_weightsNoneZero;
	std::vector<std::vector<int> > m_regressorNoneZero;

	Eigen::VectorXd m_poseParam;
	Eigen::VectorXd m_shapeParam;
	Eigen::Vector3d m_translation;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_singleAffine;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_globalAffine;
	double m_scale; 
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
