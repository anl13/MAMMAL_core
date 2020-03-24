#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

enum BODY_PART
{
	NOT_BODY = 0,
	MAIN_BODY,
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

	void SetPose(Eigen::VectorXd _poseParam) { m_poseParam = _poseParam; }
	void SetShape(Eigen::VectorXd _shapeParam) { m_shapeParam = _shapeParam; }
	void SetTranslation(Eigen::VectorXd _translation) { m_translation = _translation; }
	void SetScale(double _scale) { m_scale = _scale; }
	void ResetPose() { m_poseParam.setZero();}
	void ResetShape() { m_shapeParam.setZero(); }
	void ResetTranslation() { m_translation.setZero(); }

	Eigen::MatrixXd GetJoints() const { return m_jointsFinal; }
	Eigen::MatrixXd GetVertices() const { return m_verticesFinal; }
	Eigen::Matrix<unsigned int,-1,-1,Eigen::ColMajor> GetFaces() const { return m_faces; }
	Eigen::Vector3d GetTranslation() { return m_translation; }
	Eigen::VectorXd GetShape() { return m_shapeParam; }
	Eigen::VectorXd GetPose() { return m_poseParam; }
	Eigen::VectorXi GetParents() {return m_parent; }
	double GetScale() { return m_scale; }
	cv::Mat getTexImg() { return m_texImgBody; }
	Eigen::MatrixXd GetTexcoords() { return m_texcoords; }

	void UpdateJoints();
	void UpdateVertices();

	void UpdateNormalOrigin();
	void UpdateNormalShaped();
	void UpdateNormalFinal();
	void RescaleOriginVertices();

	void SaveObj(const std::string& filename) const;
	
	// texture
	void readTexImg(std::string filename);
	void determineBodyParts();

protected:
	int m_jointNum;// 43 / 33
	int m_vertexNum; // 2176 / 3889
	int m_shapeNum; // 0 / 41
	int m_faceNum; // 3719 / 7774

	cv::Mat m_texImgBody; 
	std::vector<BODY_PART> m_bodyParts; // body part label of each vertex

	// shape deformation
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_normalOrigin; 
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_normalShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_normalFinal;
	Eigen::VectorXd m_deform; // deformation distance 
	

	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsOrigin;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsFinal;

	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesOrigin;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesDeformed; 
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesFinal;

	Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor> m_faces;
	Eigen::Matrix<double, 2, -1, Eigen::ColMajor> m_texcoords; 

	Eigen::VectorXi m_parent;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_lbsweights;     // jointnum * vertexnum
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_jregressor;  // vertexnum * jointnum
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_shapeBlendV;  // (vertexnum*3) * shapenum
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_shapeBlendJ; // (jointnum*3) * shapenum

	Eigen::VectorXd m_poseParam;
	Eigen::VectorXd m_shapeParam;
	Eigen::Vector3d m_translation;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_singleAffine;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m_globalAffine;
	double m_scale; 
	std::string m_folder; 


	void UpdateSingleAffine() {UpdateSingleAffine(m_jointNum); }
	void UpdateSingleAffine(const int jointCount);
	void UpdateGlobalAffine() { UpdateGlobalAffine(m_jointNum); }
	void UpdateGlobalAffine(const int jointCount);

	void UpdateVerticesShaped();
	void UpdateVerticesFinal();

	void UpdateJointsShaped();
	inline void UpdateJointsFinal() { UpdateJointsFinal(m_jointNum); }
	void UpdateJointsFinal(const int jointCount);

	void UpdateVerticesDeformed(); 
};
