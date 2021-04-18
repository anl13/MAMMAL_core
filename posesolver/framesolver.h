#pragma once

#include "framedata.h"
#include "scenedata.h"
#include "../articulation/pigsolverdevice.h"
#include "../tracking/sift_matcher.h" 

class FrameSolver : public FrameData
{
public: 
	FrameSolver();

	~FrameSolver(); 


	void configByJson(std::string jsonfile) override;
	void pureTracking();

	// load manual annotation 
	std::string m_annotation_folder; 
	bool try_load_anno(); 
	void resetSolverStateMarker(); 

	// last frame data 
	vector<MatchedInstance> m_last_matched; // used for tracking
	vector<vector<Eigen::Vector3f> > m_skels3d_last;

    // sift flows 
	vector<vector<cv::KeyPoint> > m_siftKeypointsLast; // [camid, index]
	vector<cv::Mat> m_siftDescriptionLast; //[camid]
	cv::Ptr<cv::SIFT> p_sift;
	vector<vector<cv::KeyPoint> > m_siftKeypointsCurrent; // [camid, idnex]
	vector<cv::Mat> m_siftDescriptionCurrent; // [camid]
	cv::FlannBasedMatcher m_siftMatcher; 
	vector<vector<cv::DMatch> > m_siftMatches; 
	vector<vector<cv::DMatch> > m_siftMatchesCleaned; 
	void buildSIFTMapToSurface(); 
	vector<vector<SurfaceCorr> > m_siftToFaceIds; //[camid, index]
	vector<vector<vector<SIFTCorr> > > m_siftCorrs; // [pid, camid, index]
	void renderFaceIndex();
	void renderMaskColor(); 
	void detectSIFTandTrack();
	void readSIFTandTrack(); 
	cv::Mat m_faceIndexTexImg; 
	Mesh m_objForTex; 
	vector<cv::Mat> m_faceIndexImg; 

	// scene rendering
	std::shared_ptr<SceneData> mp_sceneData;
	Renderer* mp_renderEngine;
	vector<std::shared_ptr<PigSolverDevice> >       mp_bodysolverdevice;
	std::vector<cv::Mat> m_rawMaskImgs;
	void drawRawMaskImgs();
	std::string m_result_folder;
	bool is_smth;
	int m_startid;
	int m_framenum;

	vector<MatchedInstance>                   m_matched; // matched raw data after matching()
	vector<vector<DetInstance> >              m_unmatched; // [camnum, candnum]
	std::vector<cv::Mat> m_masksMatched; 

		// matching & 3d data 
	vector<vector<int> > m_clusters; // pigid, camid [candid]
	vector<vector<Eigen::Vector3f> > m_skels3d; // pigid, kptid
	bool m_use_gpu;
	int m_solve_sil_iters;
	int m_solve_sil_iters_2nd_phase; 
	float       m_epi_thres;
	std::string m_epi_type;
	std::string m_anchor_folder; 
	bool m_use_reassoc; 
	float m_terminal_thresh;

	vector<vector<vector<Eigen::Vector3f> > > m_projs; // [viewid, pigid, kptid]

		// shape solver 
	void getROI(vector<ROIdescripter>& rois, int id = 0);
	void setConstDataToSolver(); // direct set some data to solver
	void drawMaskMatched();  // can only be used after association 
	std::string m_pigConfig;
	std::string m_match_alg;
	

		// top-down matching
	void tracking();
	void matching_by_tracking();

	void reproject_skels();
	void read_parametric_data();
	void save_parametric_data();
	void solve_scales(); 

	//void load_labeled_data();
	void save_clusters();
	void load_clusters();

	void save_joints(); 
	void save_skels();

	// overall depth rendering. 
	void init_parametric_solver(); 
	void renderInteractDepth(bool withMask=false);
	std::vector<cv::Mat> m_interMask;
	std::vector<float*> d_interDepth; // depth rendering of all object
	std::vector<uchar*> d_interMask; // render mask with occlusion 
	void splitDetKeypoints(); 
	void splitDetKeypointsWithoutTracked();
	void nmsKeypointCands(std::vector<Eigen::Vector3f>& list);
	void reAssociateKeypoints(); // post-priori motion refinement
	std::vector< std::vector< std::vector<Eigen::Vector3f> > > m_keypoints_associated; // idnum, camnum, jointnum
	std::vector< std::vector< std::vector<Eigen::Vector3f> > > m_keypoints_pool; // camnum, jointnum, candnum
	std::vector< std::vector<std::vector<float> > > m_skelVis; // idnum, camnum, jointnum
	cv::Mat visualizeReassociation(); 
	cv::Mat visualizeVisibility(); 
	cv::Mat visualizeSwap(); 
	cv::Mat visualizeRawAssoc();
	cv::Mat visualizeSIFT(); 

	// 20201108: assoc with 
	void reAssocProcessStep1();
	// method 2 
	void reAssocWithoutTracked(); 
	void reAssocKeypointsWithoutTracked(); 
	void determineTracked(); 
	cv::Mat debug_visDetTracked(); 
	void nms2(std::vector<Eigen::Vector3f>& pool, int jointid,
		const std::vector<std::vector<Eigen::Vector3f> >& ref);
	
	// pipeline controlling 
	void DARKOV_Step0_topdownassoc(bool isLoad); // matching by tracking / puretracking 
	void DARKOV_Step1_setsource();  // set source data to solvers 
	void DARKOV_Step2_loadanchor(); // only load and set anchor id, without any fitting or align 
	void DARKOV_Step2_searchanchor(int pid); 
	void DARKOV_Step2_optimanchor(int pid); 
	void DARKOV_Step3_reassoc_type2(); // type2 contains three small steps: find tracked, assign untracked, solve mix-up
	void DARKOV_Step3_reassoc_type1(); 
	void DARKOV_Step4_fitrawsource();  // fit model to raw source 
	void DARKOV_Step4_fitreassoc();    // fit model to reassociated keypoints and silhouettes 
	void DARKOV_Step5_postprocess();   // some postprocessing step 
	
	// 20210418 use triangulation only; 
	// all steps are here. 
	void DirectTriangulation(); 


	// camid, candid, jointid, correspond to m_detUndist
	// each value means tracked pig id
	vector<vector<vector<int> > > m_detTracked; 
	
	// pidid, camid, jointid
	// each value means tracked candid in m_detUndist
	vector<vector<vector<int> > > m_modelTracked; 

	vector<vector<float> > m_ious; // [id, camid]
	void computeIOUs(); // return: [id, camid]

	void optimizeSilWithAnchor(int maxIterTime); 
	void saveAnchors(std::string folder);
	void loadAnchors(std::string folder, bool andsolve=false); 
	// visualization function  
	
	cv::Mat visualizeIdentity2D(int viewid = -1, int id = -1);
	cv::Mat visualizeProj();

	void writeSkel3DtoJson(std::string savepath);
	void readSkel3DfromJson(std::string jsonfile);


	void set_start_id(int _startid) { m_startid = _startid; }
	void set_frame_num(int _framenum) { m_framenum = _framenum; }
	int get_start_id() { return m_startid; }
	int get_frame_num() { return m_framenum; }

	SkelTopology get_topo() { return m_topo; }
	vector<vector<Eigen::Vector3f> >  get_skels3d() { return m_skels3d; }
	vector<MatchedInstance> get_matched() { return m_matched; }
	vector<std::shared_ptr<PigSolverDevice> > get_solvers() { return mp_bodysolverdevice; }
	vector<vector<DetInstance> > get_unmatched() { return m_unmatched; }

	int _compareSkel(const vector<Eigen::Vector3f>& skel1, const vector<Eigen::Vector3f>& skel2);
	int _countValid(const vector<Eigen::Vector3f>& skel);

	// anliang 2021/03/30
	std::vector<float> m_given_scales;
	bool m_use_given_scale;
	bool m_use_init_cluster; 
	bool m_try_load_anno;
	bool m_use_triangulation_only;

	std::string m_scenedata_path; 
	std::string m_background_folder; 
};