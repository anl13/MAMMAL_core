#pragma once 

#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "../utils/node_graph.h"
#include "../utils/math_utils.h"
#include "../utils/image_utils.h"
#include "../utils/timer_util.h"
#include "../VAE/decoder.h"
#include "common.h" 

// gpu include 
#include "vector_operations.hpp"
#include <vector_functions.h>
#include <vector_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>

class PigModelDevice
{
public:
	PigModelDevice(const std::string &_configfile);
	~PigModelDevice() {};
	PigModelDevice() = delete;
	PigModelDevice(const PigModelDevice& _) = delete;
	PigModelDevice& operator=(const PigModelDevice& _) = delete;

	void saveState(std::string state_file = "state.txt");
	void readState(std::string state_file = "state.txt");
	void saveObj(const std::string& filename) const; 

	// all data interfaces are on host 
	void SetPose(Eigen::VectorXf _poseParam) {
		for (int i = 0; i < m_jointNum; i++) m_host_poseParam[i] = _poseParam.segment<3>(3 * i); 
	}
	void SetPose(const std::vector<Eigen::Vector3f>& _poseParam) { m_host_poseParam = _poseParam; }
	void SetShape(const Eigen::VectorXf& _shapeParam) { m_host_shapeParam = _shapeParam; }
	void SetTranslation(const Eigen::Vector3f& _translation) { m_host_translation = _translation; }
	void SetScale(const float &_scale) { m_host_scale = _scale; }
	void ResetPose() { for (int i = 0; i < m_jointNum; i++)m_host_poseParam[i].setZero(); }
	void ResetShape() { m_host_shapeParam.setZero(); }
	void ResetTranslation() { m_host_translation.setZero(); }

	std::vector<Eigen::Vector3f> GetJoints() const { return m_host_jointsPosed; }
	std::vector<Eigen::Vector3f> GetVertices() const { return m_host_verticesPosed; }
	std::vector<Eigen::Vector3u> GetFacesTex() { return m_host_facesTex; }
	std::vector<Eigen::Vector3u> GetFacesVert() { return m_host_facesVert; }
	Eigen::Vector3f GetTranslation() { return m_host_translation; }
	Eigen::VectorXf GetShape() { return m_host_shapeParam; }
	std::vector<Eigen::Vector3f> GetPose() { return m_host_poseParam; }
	std::vector<Eigen::Vector3f> GetNormals() { return m_host_normalsFinal; }
	std::vector<int> GetParents() { return m_host_parents; }
	float GetScale() { return m_host_scale; }
	cv::Mat getTexImg() { return m_host_texImg; }
	std::vector<Eigen::Vector2f> GetTexcoords() { return m_host_texcoords; }
	std::string GetFolder() { return m_folder; }
	int GetVertexNum() { return m_vertexNum; }
	int GetJointNum() { return m_jointNum; }
	int GetFaceNum() { return m_faceNum; }
	Eigen::MatrixXf GetShapeBlendV() { return m_host_shapeBlendV; }
	Eigen::MatrixXf GetJRegressor() { return m_host_jregressor; }
	Eigen::MatrixXf GetLBSWeights() { return m_host_lbsweights; }
	std::vector<BODY_PART> GetBodyPart() { return m_host_bodyParts; }

	void UpdateVertices();
	void UpdateJoints();
	void UpdateNormalFinal(); 
	//void UpdateNormals();

	// texture
	//void ReadTexImg(std::string filename);
	//void SaveTexImg(std::string filename);

	//void InitNodeAndWarpField();
	//void UpdateModelShapedByKNN();
	//void SaveWarpField();
	//void LoadWarpField();

protected:
	// basic parameter 
	int m_jointNum;
	int m_vertexNum; 
	int m_shapeNum;
	int m_faceNum; 
	int m_texNum;
	std::string m_folder;

	// model state data 
	std::vector<Eigen::Vector3f> m_host_verticesOrigin;
	std::vector<Eigen::Vector3f> m_host_verticesScaled; 
	std::vector<Eigen::Vector3f> m_host_verticesShaped;
	std::vector<Eigen::Vector3f> m_host_verticesDeformed;
	std::vector<Eigen::Vector3f> m_host_verticesPosed;

	std::vector<Eigen::Vector3f> m_host_jointsOrigin;
	std::vector<Eigen::Vector3f> m_host_jointsScaled; 
	std::vector<Eigen::Vector3f> m_host_jointsShaped; 
	std::vector<Eigen::Vector3f> m_host_jointsDeformed; 
	std::vector<Eigen::Vector3f> m_host_jointsPosed; 

	std::vector<Eigen::Vector3f> m_host_normalsFinal; 

	// model fixed data 
	std::vector<Eigen::Vector3u> m_host_facesTex;
	std::vector<Eigen::Vector3u> m_host_facesVert;
	std::vector<Eigen::Vector2f> m_host_texcoords;
	std::vector<int>           m_host_parents;
	Eigen::MatrixXf m_host_lbsweights;     // jointnum * vertexnum
	Eigen::MatrixXf m_host_jregressor;  // vertexnum * jointnum
	Eigen::MatrixXf m_host_shapeBlendV;  // (vertexnum*3) * shapenum
	Eigen::MatrixXf m_host_shapeBlendJ; // (jointnum*3) * shapenum
	std::vector<BODY_PART> m_host_bodyParts; // body part label of each vertex

	// embeded driven data 
	std::vector<Eigen::Vector3f> m_host_poseParam;
	Eigen::VectorXf m_host_shapeParam;
	Eigen::Vector3f m_host_translation;
	float m_host_scale;
	cv::Mat m_host_texImg; 

	// middle data for skinning 
	std::vector<Eigen::Matrix4f> m_host_localSE3;
	std::vector<Eigen::Matrix4f> m_host_globalSE3; 
	std::vector<Eigen::Matrix4f> m_host_normalizedSE3;

	pcl::gpu::DeviceArray<int> m_device_parents; 
	pcl::gpu::DeviceArray2D<float> m_device_lbsweights; // vertexnum * jointnum, row major
	pcl::gpu::DeviceArray<Eigen::Vector3u> m_device_faces; 
	pcl::gpu::DeviceArray<BODY_PART> m_device_bodyParts;
	pcl::gpu::DeviceArray<Eigen::Vector3f> m_device_verticesOrigin;
	pcl::gpu::DeviceArray<Eigen::Vector3f> m_device_verticesScaled; 
	pcl::gpu::DeviceArray<Eigen::Vector3f> m_device_normals; 
	
	
	pcl::gpu::DeviceArray<Eigen::Vector3f> m_device_verticesDeformed; 
	pcl::gpu::DeviceArray<Eigen::Vector3f> m_device_verticesPosed;
	pcl::gpu::DeviceArray<Eigen::Vector3f> m_device_jointsDeformed; 
	pcl::gpu::DeviceArray<Eigen::Vector3f> m_device_jointsPosed;
		

	void UpdateNormalsFinal_device(); 

	void UpdateLocalSE3_host();
	void UpdateGlobalSE3_host();
	void UpdateNormalizedSE3_host(); 

	void UpdateVerticesShaped();
	void UpdateVerticesDeformed(); 
	void UpdateVerticesPosed_device();

	void UpdateJointsShaped();
	void UpdateJointsDeformed();
	void UpdateJointsPosed_host(); 

	void UpdateScaled_device(); 

};
