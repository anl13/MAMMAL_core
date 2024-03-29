#pragma once

#include <vector>
#include <iostream> 
#include <fstream> 

#include <Eigen/Eigen> 
#include <json/json.h>
#include "pigmodeldevice.h"
#include "../render/renderer.h"
#include "../utils/skel.h"
#include "../utils/camera.h"

#include "../utils/image_utils.h"
#include "../utils/math_utils.h" 
#include "../utils/gpuutils.h"

//#define DEBUG_SOLVER

//#define USE_GPU_SOLVER 
#define SHOW_FITTING_INFO

class ParamSet {
public: 
	ParamSet() {} 
	void loadParams(const Json::Value& data); 
	float m_valid_threshold;
	float m_lambda;
	float m_w_data_term;
	float m_w_sil_term;
	float m_w_reg_term;
	float m_w_temp_term;
	float m_w_floor_term;
	float m_w_on_floor_term;
	float m_kpt_track_dist;
	float m_w_anchor_term;
	float m_iou_thres;
	bool m_use_bodyonly_reg;
	bool m_use_height_enhanced_temp;
	float m_w_collision_term;
	bool m_use_given_scale; 
	int m_sil_step; 
	int m_collision_step; 
};

typedef struct
{
	std::vector<Eigen::Vector3f> pose; // pose in rotation 
	Eigen::Vector3f translation; 
	float scale; 
	std::vector<Eigen::Vector3f> joint_positions_62; // root relative 3D joint positions 
	std::vector<Eigen::Vector3f> joint_positions_23; // root relative 3D joint positions 
}AnchorPoseType;

typedef struct {
	std::vector<AnchorPoseType> anchors; 
	void load(std::string folder); 
}AnchorPoseLib;

class PigSolverDevice : public PigModelDevice
{
public:
	PigSolverDevice() = delete;
	PigSolverDevice(const std::string &_configfile, bool _use_gpu = false, int _view_num = 10);
	~PigSolverDevice();
	PigSolverDevice(const PigSolverDevice&) = delete;
	PigSolverDevice& operator=(const PigSolverDevice&) = delete;

	ParamSet m_params;
	std::string m_anchor_folder; 
	void setSource(const MatchedInstance& _source) {
		m_source = _source;
	}

	// state marker
	bool m_isUpdated;
	bool m_isPostprocessed;
	void resetStateMarker();
	float m_trackConf;
	bool m_use_gpu;
	float m_gtscale;  // This gt scale is given from outside
	bool m_isReAssoc;

	float getAvgHeight();
	float computeProjectionError();
	float computeScale();

	void getTheta(Eigen::VectorXf& theta);
	void setTheta(const Eigen::VectorXf& theta);
	void getThetaAnchor(Eigen::VectorXf& theta);
	void setThetaAnchor(const Eigen::VectorXf& theta);

	void setParams(const ParamSet& _params); 
	void setCameras(const std::vector<Camera>& _cams) { m_cameras = _cams; }
	void setRenderer(Renderer * _p_render) { mp_renderEngine = _p_render; }
	void setROIs(std::vector<ROIdescripter> _rois) { m_rois = _rois; }
	std::vector<Eigen::Vector3f> getSkel3D() { return m_skel3d; }
	std::vector<int> getPoseToOptimize() { return m_poseToOptimize; }
	std::vector<std::vector<Eigen::Vector3f> > getSkelsProj() { return m_skelProjs; }

	// detection confidence for each joint, used for adapt 
	// joint joint fitting weights 
	std::vector<float> m_det_confs;

	std::vector<Eigen::Vector3f>  directTriangulationHost(int validViewThresh = 2); // calc m_skel3d using direct triangulation
	void globalAlign(); // compute scale and global R,T
	void optimizeTri();


	void optimizePose();
	void optimizePoseSilhouette(
		int maxIter);

	cv::Mat debug_source_visualize();
	std::vector<float> computeValidObservation();

	void fitPoseToVSameTopo(const std::vector<Eigen::Vector3f> &_tv);
	void fitPoseToJointSameTopo(const std::vector<Eigen::Vector3f> &_joints);


	//=================TERMS====================
	// J_joint: [3*jontnum+3, 3*jointnum]
// J_vert: [3*jointnum+3, 3*vertexnum]
// 3*jointnum+3:  pose parameter to solve 
// 2d array on gpu are row major 
	void calcPoseJacobiFullTheta_device(pcl::gpu::DeviceArray2D<float> &J_joint,
		pcl::gpu::DeviceArray2D<float> &J_vertices, bool with_vert = true);
	void calcPoseJacobiPartTheta_device(pcl::gpu::DeviceArray2D<float> &J_joint,
		pcl::gpu::DeviceArray2D<float> &J_vert, bool with_vert = true);
	void calcSkelJacobiPartTheta_host(Eigen::MatrixXf& J);
	void calcPose2DTerm_host(const DetInstance& det, int camid, const std::vector<Eigen::Vector3f>& skel2d,
		const Eigen::MatrixXf& Jacobi3d, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb,
		float radius, bool is_converge_detect = false, std::vector<int> high_conf_views = {});
	void calcJoint3DTerm_host(const Eigen::MatrixXf& Jacobi3d, const std::vector<Eigen::Vector3f>& skel3d, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	void calcSkel3DTerm_host(const Eigen::MatrixXf& Jacobi3d, const std::vector<Eigen::Vector3f>& joints3d, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	void calcAnchorTerm_host(int anchorid, const Eigen::VectorXf& theta, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	void calcAnchorTermHeight_host(int anchorid, const Eigen::MatrixXf& Jacobi3d, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);

	void Calc2dJointProjectionTerm(
		const MatchedInstance& source,
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb,
		float track_radius = 80,
		bool with_depth_weight = false, bool is_converge_detect = false);

	void Calc2dSkelProjectionTermReassoc(
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, bool with_depth_weight = false);
	void Calc2DSkelTermReassoc_host(const std::vector<Eigen::Vector3f>& skel_det, const std::vector<Eigen::Vector3f>& skel3d, int camid,
		const Eigen::MatrixXf& Jacobi3d, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);

	void CalcJointFloorTerm(
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb
	);
	void CalcJointBidirectFloorTerm( // foot_contact: 4 * bool , [9,10,15,16] left front, right font, left back, right back
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, std::vector<bool> foot_contact
	);
	void CalcJointOnFloorTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);

	void CalcRegTerm(const Eigen::VectorXf& theta, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, bool adaptive_weight = false);
	void CalcRegTermBodyOnly(const Eigen::VectorXf& theta, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);

	void CalcJointTempTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, const Eigen::VectorXf& last_theta, const Eigen::VectorXf& theta);
	void CalcJointTempTerm2(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, const Eigen::MatrixXf& skelJ,
		const std::vector<Eigen::Vector3f>& last_regressed_skel);


	void CalcPoseJacobiFullTheta_cpu(Eigen::MatrixXf& jointJacobiPose, Eigen::MatrixXf& J_vert,
		bool with_vert);
	void CalcPoseJacobiPartTheta_cpu(Eigen::MatrixXf& J_joint, Eigen::MatrixXf& J_vert,
		bool with_vert);

	// sil term constructor 
	void CalcSilhouettePoseTerm(
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, int iter = 0);
	void calcSilhouetteJacobi_device(
		Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Vector3f T,
		float* d_depth, float* d_depth_interact, int idcode, int paramNum, int view
	);

	void CalcSilouettePoseTerm_cpu(
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, int iter = 0);

	// 2021.09.26 
	void CalcCollisionJointTerm_cpu(
		Eigen::MatrixXf& J_joint, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb
	);
	void CalcCollisionSurfaceTerm_cpu(
		Eigen::MatrixXf& J_vert, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb
	);

	void renderDepths();

	void calcPoseJacobiFullTheta_V_device(
		pcl::gpu::DeviceArray2D<float> J_vert,
		pcl::gpu::DeviceArray2D<float> J_joint,
		pcl::gpu::DeviceArray2D<float> d_RP,
		pcl::gpu::DeviceArray2D<float> d_LP
	);

	// constant gpu attributes 
	std::vector<int> m_host_paramLines;
	pcl::gpu::DeviceArray<int> m_device_paramLines;

	// feed by outer data maintainer 
	void generateDataForSilSolver(); 
	bool init_backgrounds; 
	std::vector<cv::Mat> m_det_masks;
	std::vector<cv::Mat> m_det_masks_binary; 
	std::vector<int> m_viewids; 
	std::vector<float> m_valid_keypoint_ratio; 
	std::vector<float> m_mask_areas; 
	std::vector<uchar*> d_det_mask; // full size
	std::vector<float*> d_det_sdf; // half size
	float* d_rend_sdf; // half size
	std::vector<float*> d_det_gradx; // half size
	std::vector<float*> d_det_grady; // half size
	uchar* d_const_distort_mask; // full size
	std::vector<uchar*> d_const_scene_mask; // full size 
	uchar* d_middle_mask; // half size 
	int m_pig_id; // 0-3
	std::vector<cv::Mat> c_const_scene_mask; 
	cv::Mat c_const_distort_mask; 
	std::vector<cv::Mat> m_rawimgs; 
	std::vector<cv::Mat> m_scene_mask_chamfer; 
	cv::Mat m_undist_mask_chamfer;

	void postProcessing(); // post process: project skel, determine model visibility 
	std::vector<std::vector<bool> > m_det_ignore; // [camid, jointid]
	std::vector<std::vector<int> > m_visRegressorList; //[jointid]

	// use anchor to optimize 
	void optimizeAnchor(int anchor_id);
	void optimizePoseWithAnchor(); 
	int searchAnchorSpace(); 
	int m_anchor_id; 
	AnchorPoseLib m_anchor_lib;
	void calcSkel2DTermAnchor_host(const DetInstance& det, int camid, const std::vector<Eigen::Vector3f>& skel2d,
		const Eigen::MatrixXf& Jacobi3d, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	void CalcSkelProjectionTermAnchor(
		const MatchedInstance& source,
		Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, bool with_depth_weight = false);
	float evaluate_error(); 
	float evaluate_mask_error(); 
	void CalcLambdaTerm(Eigen::MatrixXf& ATA); 
	void CalcAnchorRegTerm(const Eigen::VectorXf& theta, Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb, int anchor_id, bool adaptive_weight = false);
	void CalcSurfaceFloorTerm(Eigen::MatrixXf& ATA, Eigen::VectorXf& ATb);
	
	void computeDepthWeight(); 
	
	std::vector<float* > d_depth_renders_interact; 
	float optimizePoseSilWithAnchorOneStep(int iter); 

	float approxIOU(int view); 

	std::vector < std::vector<Eigen::Vector3f> > m_skelProjs; // [viewid, jointid] (u,v,visibility)
	std::vector<float> regressSkelVisibility(int camid); // assume d_depth_render_interact exist. 
	void computeAllSkelVisibility(); 
	std::vector<std::vector<float> > m_skel_vis;
	void projectSkels(); 

	std::vector<std::vector<Eigen::Vector3f> > m_keypoints_reassociated; // camnum, jointnum
	cv::Mat debug_vis_reassoc_swap();

	std::vector<float> o_ious; 

	// 2021/2/20 add: fitting for clicked points  
	std::vector<std::vector<Eigen::Vector3f> > clicked_points; 
	void optimizePoseWithClickedPoints();

	Eigen::VectorXf m_optimMask; 
	std::vector<int> m_currentHierarchy; 
	
	void map_reduced_vertices(); 
	vector<vector<Eigen::Vector3f> >  m_other_pigs_reduced_vertices; 
	vector<Eigen::Vector3f> m_other_centers; 
	std::vector<Eigen::Vector3f> m_reduced_vertices;
	std::vector<Eigen::Vector3u> m_reduced_faces;
	std::vector<Eigen::Vector3f> m_reduced_normals;

protected:
	// 2021/09/25
	// reduced model for collision detection and social 
	std::vector<int> m_reduced_ids; 
	
	void load_reduced();

	// state indicator 
	bool m_initScale;
	float m_scaleCount; 

	// config info, read from json file 
	
	std::vector<int> m_poseToOptimize;


	// optimization source
	MatchedInstance m_source; 
	std::vector<Camera> m_cameras;
	std::vector<ROIdescripter> m_rois; 
	Eigen::VectorXf m_last_thetas; 
	std::vector<Eigen::Vector3f> m_last_regressed_skel3d; 

	// output 
	std::vector<Eigen::Vector3f> m_skel3d; 
	std::vector<float> m_depth_weight; // center depth for each view 
	std::vector<float> m_param_reg_weight; // weight to regularize different joints 
	Eigen::VectorXf m_param_temp_weight; // temporal weight per joint
	Eigen::VectorXf m_param_observe_num; // joint observation for each joint

	// render engine 
	Renderer* mp_renderEngine;

	// tmp data pre-allocated at construction stage
	std::vector<float*> d_depth_renders; // full size

	pcl::gpu::DeviceArray2D<float> d_J_vert; 
	pcl::gpu::DeviceArray2D<float> d_J_joint; 
	pcl::gpu::DeviceArray2D<float> d_J_joint_full, d_J_vert_full;
	pcl::gpu::DeviceArray2D<float> d_ATA_sil; 
	pcl::gpu::DeviceArray<float> d_ATb_sil; 
	pcl::gpu::DeviceArray2D<float> d_RP;
	pcl::gpu::DeviceArray2D<float> d_LP; 

	pcl::gpu::DeviceArray2D<float> d_AT_sil; 
	pcl::gpu::DeviceArray<float> d_b_sil; 

	Eigen::MatrixXf h_J_joint; // [jointnum * paramNum]
	Eigen::MatrixXf h_J_vert;  // [vertexnum * paramNum]
	Eigen::MatrixXf h_J_skel;
};