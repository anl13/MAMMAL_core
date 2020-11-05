#pragma once

#include "framedata.h"
#include "scenedata.h"
#include "../articulation/pigsolverdevice.h"

class FrameSolver : public FrameData
{
public: 
	FrameSolver() {
		m_epi_thres = -1;
		m_epi_type = "p2l";
	}

	~FrameSolver(); 

	void configByJson(std::string jsonfile) override;

	std::shared_ptr<SceneData> mp_sceneData;

	SkelTopology get_topo() { return m_topo; }
	vector<vector<Eigen::Vector3f> >  get_skels3d() { return m_skels3d; }
	vector<MatchedInstance> get_matched() { return m_matched; }
	vector<std::shared_ptr<PigSolverDevice> > get_solvers() { return mp_bodysolverdevice; }
	vector<vector<DetInstance> > get_unmatched() { return m_unmatched; }

	void pureTracking();

	Renderer* mp_renderEngine;
	vector<std::shared_ptr<PigSolverDevice> >       mp_bodysolverdevice;
	std::vector<cv::Mat> m_rawMaskImgs;
	void drawRawMaskImgs();
	std::string result_folder;
	bool is_smth;

	int _compareSkel(const vector<Eigen::Vector3f>& skel1, const vector<Eigen::Vector3f>& skel2);
	int _countValid(const vector<Eigen::Vector3f>& skel);

	vector<MatchedInstance>                   m_matched; // matched raw data after matching()
	vector<vector<DetInstance> >              m_unmatched; // [camnum, candnum]

	vector<MatchedInstance> m_last_matched; // used for tracking
		// matching & 3d data 
	vector<vector<int> > m_clusters; // pigid, camid [candid]
	vector<vector<Eigen::Vector3f> > m_skels3d;
	vector<vector<Eigen::Vector3f> > m_skels3d_last;
	void detNMS();
	bool m_use_gpu;
	int m_solve_sil_iters;
	float       m_epi_thres;
	std::string m_epi_type;

	vector<vector<vector<Eigen::Vector3f> > > m_projs; // [viewid, pigid, kptid]

		// shape solver 
	void getROI(vector<ROIdescripter>& rois, int id = 0);
	void setConstDataToSolver(int id); // direct set some data to solver
	vector<cv::Mat> drawMask();  // can only be used after association 
	std::string m_pigConfig;
	std::string m_match_alg;


		// top-down matching
	void tracking();
	void matching_by_tracking();
	void reproject_skels();
	void solve_parametric_model();
	void solve_parametric_model_pipeline2();
	void solve_parametric_model_pipeline3(); 
	void solve_parametric_model_optimonly();
	void read_parametric_data();
	void save_parametric_data();
	void solve_parametric_model_cpu(); 

	//void load_labeled_data();
	void save_clusters();
	void load_clusters();

	// overall depth rendering. 
	void init_parametric_solver(); 
	void renderInteractDepth(bool withMask=false);
	std::vector<cv::Mat> m_interMask;
	std::vector<float*> d_interDepth; // depth rendering of all object
	std::vector<uchar*> d_interMask; // render mask with occlusion 
	void splitDetKeypoints(); 
	void nmsKeypointCands(std::vector<Eigen::Vector3f>& list);
	void reAssociateKeypoints(); // post-priori motion refinement
	std::vector< std::vector< std::vector<Eigen::Vector3f> > > m_keypoints_associated; // idnum, camnum, jointnum
	std::vector< std::vector< std::vector<Eigen::Vector3f> > > m_keypoints_pool; // camnum, jointnum, candnum
	std::vector< std::vector<std::vector<float> > > m_skelVis; // idnum, camnum, jointnum
	cv::Mat visualizeReassociation(); 
	cv::Mat visualizeKeypointsPool(); 
	cv::Mat visualizeVisibility(); 
	cv::Mat visualizeSwap(); 
	cv::Mat visualizeRawAssoc();
	void reAssocProcessStep1();
	void reAssocProcessStep2();

	void optimizeSil(int maxIterTime); 
	void optimizeSilWithAnchor(int maxIterTime); 
	void saveAnchors(std::string folder);
	void loadAnchors(std::string folder, bool andsolve=false); 
	// visualization function  
	
	cv::Mat visualizeIdentity2D(int viewid = -1, int id = -1);
	cv::Mat visualizeProj();

	void writeSkel3DtoJson(std::string savepath);
	void readSkel3DfromJson(std::string jsonfile);

	int m_startid;
	int m_framenum;
	void set_start_id(int _startid) { m_startid = _startid; }
	void set_frame_num(int _framenum) { m_framenum = _framenum; }
	int get_start_id() { return m_startid; }
	int get_frame_num() { return m_framenum; }
};