#pragma once

#include <vector>
#include <iostream> 
#include <fstream> 

#include <Eigen/Eigen> 

#include "pigmodeldevice.h"
#include "../render/renderer.h"
#include "../utils/skel.h"
#include "../utils/camera.h"

#include "../utils/image_utils.h"
#include "../utils/math_utils.h" 
#include "gpuutils.h"

//#define DEBUG_SIL

class PigSolverDevice : public PigModelDevice
{
public: 
	PigSolverDevice() = delete; 
	PigSolverDevice(const std::string &_configfile); 
	~PigSolverDevice();
	PigSolverDevice(const PigSolverDevice&) = delete; 
	PigSolverDevice& operator=(const PigSolverDevice&) = delete; 

	void setSource(const MatchedInstance& _source) { 
		m_source.view_ids.clear(); 
		m_source.dets.clear(); 
		m_source.view_ids = _source.view_ids;
		m_source.dets = _source.dets; 
	}
	void setCameras(const std::vector<Camera>& _cams) { m_cameras = _cams;  }
	void setRenderer(Renderer * _p_render) { mp_renderEngine = _p_render; }
	void setROIs(std::vector<ROIdescripter> _rois) { m_rois = _rois; }

	void directTriangulationHost(); // calc m_skel3d using direct triangulation
	std::vector<Eigen::Vector3f> getRegressedSkel(); 
	void globalAlign(); // compute scale and global R,T

	void normalizeCamera(); 
	void normalizeSource(); 
	void optimizePose(); 
	void optimizePoseSilhouette(
		int maxIter);

	void debug(); 


	void fitPoseToVSameTopo(const std::vector<Eigen::Vector3f> &_tv);

	std::vector<Eigen::Vector3f> getRegressedSkel_host(); 


	// J_joint: [3*jontnum+3, 3*jointnum]
// J_vert: [3*jointnum+3, 3*vertexnum]
// 3*jointnum+3:  pose parameter to solve 
// 2d array on gpu are row major 
	void calcPoseJacobiFullTheta_device(pcl::gpu::DeviceArray2D<float> &J_joint,
		pcl::gpu::DeviceArray2D<float> &J_vertices);
	void calcPoseJacobiPartTheta_device(pcl::gpu::DeviceArray2D<float> &J_joint,
		pcl::gpu::DeviceArray2D<float> &J_vert);
	void calcSkelJacobiPartTheta_host(Eigen::MatrixXf& J);
	void calcPose2DTerm_host(const DetInstance& det, const Camera& cam, const std::vector<Eigen::Vector3f>& skel2d,
		const Eigen::MatrixXf& Jacobi3d, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);

	void Calc2dJointProjectionTerm(
		const MatchedInstance& source,
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);

	void CalcPoseJacobiFullTheta_cpu(Eigen::MatrixXf& jointJacobiPose, Eigen::MatrixXf& J_vert,
		bool with_vert);
	void CalcPoseJacobiPartTheta_cpu(Eigen::MatrixXf& J_joint, Eigen::MatrixXf& J_vert,
		bool with_vert);

	// constant gpu attributes 
	std::vector<int> m_host_paramLines;
	pcl::gpu::DeviceArray<int> m_device_paramLines;

	bool init_backgrounds; 
private:

	
	// state indicator 
	bool m_initScale;

	// config info, read from json file 
	std::vector<CorrPair> m_skelCorr; 
	SkelTopology m_skelTopo; 
	std::vector<int> m_poseToOptimize;

	// optimization source
	MatchedInstance m_source; 
	std::vector<Camera> m_cameras;
	std::vector<ROIdescripter> m_rois; 

	// output 
	std::vector<Eigen::Vector3f> m_skel3d; 

	// render engine 
	Renderer* mp_renderEngine;

	// tmp data pre-allocated at construction stage
	std::vector<float*> d_depth_renders;
	pcl::gpu::DeviceArray2D<uchar> d_det_mask; 
	pcl::gpu::DeviceArray2D<float> d_det_sdf; 
	pcl::gpu::DeviceArray2D<float> d_rend_sdf;
	pcl::gpu::DeviceArray2D<float> d_det_gradx;
	pcl::gpu::DeviceArray2D<float> d_det_grady;
	pcl::gpu::DeviceArray2D<uchar> d_const_distort_mask;
	pcl::gpu::DeviceArray2D<uchar> d_const_scene_mask;
	pcl::gpu::DeviceArray2D<float> d_J_vert; 
	pcl::gpu::DeviceArray2D<float> d_J_joint; 
	pcl::gpu::DeviceArray2D<float> d_JT_sil;
	pcl::gpu::DeviceArray2D<float> d_J_joint_full, d_J_vert_full;
	pcl::gpu::DeviceArray<float> d_r_sil;
	pcl::gpu::DeviceArray2D<float> d_ATA_sil; 
	pcl::gpu::DeviceArray<float> d_ATb_sil; 
	pcl::gpu::DeviceArray2D<float> d_RP;
	pcl::gpu::DeviceArray2D<float> d_LP; 
	uchar* d_middle_mask; 

	Eigen::MatrixXf h_J_joint; 
	Eigen::MatrixXf h_J_vert; 

	// sil term constructor 
	void CalcSilhouettePoseTerm(
		const std::vector<float*>& renders,
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	void calcSilhouetteJacobi_device(
		Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Vector3f T,
		float* d_depth, int idcode, int paramNum
	);

	void CalcSilouettePoseTerm_cpu(
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);

	void renderDepths();

	void calcPoseJacobiFullTheta_V_device(
		pcl::gpu::DeviceArray2D<float> J_vert,
		pcl::gpu::DeviceArray2D<float> J_joint,
		pcl::gpu::DeviceArray2D<float> d_RP,
		pcl::gpu::DeviceArray2D<float> d_LP
	);
};