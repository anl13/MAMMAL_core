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

	void SetPose(Eigen::VectorXd _poseParam) { poseParam = _poseParam; }
	void SetShape(Eigen::VectorXd _shapeParam) { shapeParam = _shapeParam; }
	void SetTranslation(Eigen::VectorXd _translation) { translation = _translation; }
	void ResetPose() { poseParam.setZero();}
	void ResetShape() { shapeParam.setZero(); }
	void ResetTranslation() { translation.setZero(); }

	Eigen::MatrixXd GetJoints() const { return jointsFinal; }
	Eigen::MatrixXd GetVertices() const { return verticesFinal; }
	Eigen::Matrix<unsigned int,-1,-1,Eigen::ColMajor> GetFaces() const { return faces; }
	Eigen::Vector3d GetTranslation() { return translation; }
	Eigen::VectorXd GetShape() { return shapeParam; }
	Eigen::VectorXd GetPose() { return poseParam; }
	Eigen::VectorXi GetParents() {return parent; }


	void UpdateJoints();
	void UpdateVertices();

	void SaveObj(const std::string& filename) const;

protected:
	const int jointNum = 33;
	const int vertexNum = 3889;
	const int shapeNum = 41;
	const int faceNum = 7774;

	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> jointsOrigin;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> jointsShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> jointsFinal;

	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> verticesOrigin;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> verticesShaped;
	Eigen::Matrix<double, 3, -1, Eigen::ColMajor> verticesFinal;

	Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor> faces;

	Eigen::VectorXi parent;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> weights;     // vertexnum * jointnum
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> jregressor;  // vertexnum * jointnum
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> shapeBlend;  // (vertexnum*3) * shapenum
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> shape2Joint; // (jointnum*3) * shapenum

	Eigen::VectorXd poseParam;
	Eigen::VectorXd shapeParam;
	Eigen::Vector3d translation;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> singleAffine;
	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> globalAffine;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> poseJacobi;

	void UpdateSingleAffine() {UpdateSingleAffine(jointNum); }
	void UpdateSingleAffine(const int jointCount);
	void UpdateGlobalAffine() { UpdateGlobalAffine(jointNum); }
	void UpdateGlobalAffine(const int jointCount);

	void UpdateVerticesShaped();
	void UpdateVerticesFinal();

	void UpdateJointsShaped();
	inline void UpdateJointsFinal() { UpdateJointsFinal(jointNum); }
	void UpdateJointsFinal(const int jointCount);

	void CalcPoseJacobi(); // Attention! calculate jacobi will update global affine and jointsfinal

};
