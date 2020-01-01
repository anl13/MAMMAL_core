#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>

class SMAL
{
public:
	SMAL(const std::string &folder);
	~SMAL();
	SMAL() = delete;
	SMAL(const SMAL& _) = delete;
	SMAL& operator=(const SMAL& _) = delete;

	void saveState(std::string state_file="state.txt"); 
    void readState(std::string state_file="state.txt");

	void SetPose(Eigen::VectorXd _poseParam) { m_poseParam = _poseParam; }
	void SetShape(Eigen::VectorXd _shapeParam) { m_shapeParam = _shapeParam; }
	void SetTranslation(Eigen::VectorXd _translation) { m_translation = _translation; }
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

	void UpdateJoints();
	void UpdateVertices();

	void SaveObj(const std::string& filename) const;

protected:
	const int m_jointNum = 33;
	const int m_vertexNum = 3889;
	const int m_shapeNum = 41;
	const int m_faceNum = 7774;

	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsOrigin;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_jointsFinal;

	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesOrigin;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> m_verticesFinal;

	Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor> m_faces;

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



	void UpdateSingleAffine() {UpdateSingleAffine(m_jointNum); }
	void UpdateSingleAffine(const int jointCount);
	void UpdateGlobalAffine() { UpdateGlobalAffine(m_jointNum); }
	void UpdateGlobalAffine(const int jointCount);

	void UpdateVerticesShaped();
	void UpdateVerticesFinal();

	void UpdateJointsShaped();
	inline void UpdateJointsFinal() { UpdateJointsFinal(m_jointNum); }
	void UpdateJointsFinal(const int jointCount);

	
};
