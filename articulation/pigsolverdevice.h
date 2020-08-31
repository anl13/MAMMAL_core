#pragma once

#include <vector>
#include <iostream> 
#include <fstream> 

#include <Eigen/Eigen> 

#include "pigmodeldevice.h"
#include "../render/render_object.h"
#include "../render/renderer.h"
#include "../utils/skel.h"
#include "../utils/camera.h"

#include "../utils/image_utils.h"
#include "../utils/math_utils.h" 


class PigSolverDevice : public PigModelDevice
{
public: 
	PigSolverDevice() = delete; 
	PigSolverDevice(const std::string &_configfile); 
	~PigSolverDevice() {}
	PigSolverDevice(const PigSolverDevice&) = delete; 
	PigSolverDevice& operator=(const PigSolverDevice&) = delete; 

	void setSource(const MatchedInstance& _source) { m_source = _source; }
	void setCameras(const std::vector<Camera>& _cams) { m_cameras = _cams;  }
	void setRenderer(Renderer * _p_render) { mp_renderer = _p_render; }

	void directTriangulationHost(); // calc m_skel3d using direct triangulation
	std::vector<Eigen::Vector3f> getRegressedSkel(); 
	void globalAlign(); // compute scale and global R,T

	// J_joint: [3*jontnum+3, 3*jointnum]
	// J_vert: [3*jointnum+3, 3*vertexnum]
	// 3*jointnum+3:  pose parameter to solve 
	// 2d array on gpu are row major 
	void calcPoseJacobiFullTheta_device(pcl::gpu::DeviceArray2D<float> &J_joint, 
		pcl::gpu::DeviceArray2D<float> &J_vertices);
	void calcPoseJacobiPartTheta_device(pcl::gpu::DeviceArray2D<float> &J_joint,
		pcl::gpu::DeviceArray2D<float> &J_vert); 

private:
	// constant gpu attributes 
	std::vector<int> m_host_paramLines; 
	pcl::gpu::DeviceArray<int> m_device_paramLines; 

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

	Renderer* mp_renderer; 

	std::vector<Eigen::Vector3f> m_skel3d; 
};