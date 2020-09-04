#pragma once 

#include <iostream> 
#include <fstream> 
#include <iomanip>
#include <Eigen/Eigen> 
#include <json/json.h> 
#include <vector> 
#include <opencv2/opencv.hpp> 
#include "../utils/math_utils.h"
#include "../utils/camera.h"
#include "../utils/image_utils.h"
#include "../utils/geometry.h" 
#include "../utils/Hungarian.h"
#include "../articulation/pigsolverdevice.h"
#include "clusterclique.h"
#include "../utils/skel.h" 
#include "../articulation/pigsolver.h"

using std::vector; 

// data of a frame, together with matching process
class FrameData{
public: 
    FrameData(){
        m_camids = {0,1,2,5,6,7,8,9,10,11};
        m_camNum = 10; 
        m_imw = 1920; 
        m_imh = 1080; 
        getColorMap("anliang_rgb", m_CM); 
        m_epi_thres = -1; 
        m_epi_type = "p2l";
    }
    ~FrameData(){} 

    // attributes 
    void set_frame_id(int _frameid){m_frameid = _frameid;}
    void set_start_id(int _startid){m_startid = _startid;}
    void set_frame_num(int _framenum){m_framenum = _framenum;}
    int get_frame_id() {return m_frameid;}
    int get_start_id(){return m_startid;}
    int get_frame_num(){return m_framenum;}
	vector<cv::Mat> get_imgs_undist() { return m_imgsUndist; }
    SkelTopology get_topo(){return m_topo;}
	vector<vector<Eigen::Vector3f> >  get_skels3d(){return m_skels3d;}
    vector<MatchedInstance> get_matched() {return m_matched; }
    vector<Camera> get_cameras(){return m_camsUndist; }
	vector<std::shared_ptr<PigSolverDevice> > get_solvers() { return mp_bodysolverdevice; }
	vector<vector<DetInstance> > get_unmatched() { return m_unmatched; }
    void configByJson(std::string jsonfile); 
    void fetchData(); 
    cv::Mat test(); 

	// debug 
	void debug_fitting(int pid=0); 
	void visualizeDebug(int id = -1);
	void debug_chamfer(int pid = 0); 
	void view_dependent_clean();
	void top_view_clean(DetInstance& det);
	void side_view_clean(DetInstance& det);
	void clean_step1();

    // top-down matching
    void matching(); 
    void tracking(); 
	void matching_by_tracking(); 
    void reproject_skels(); 
	void solve_parametric_model(); 
	void read_parametric_data(); 
	void save_parametric_data(); 
	void solve_parametric_model_cpu(); 

	//void load_labeled_data();
	void save_clusters();
	void load_clusters(); 

    // visualization function  
    cv::Mat visualizeSkels2D(); 
    cv::Mat visualizeIdentity2D(int viewid=-1, int id=-1);
    cv::Mat visualizeProj(); 

    void writeSkel3DtoJson(std::string savepath); 
    void readSkel3DfromJson(std::string jsonfile); 

	// shape solver 
	void getROI(vector<ROIdescripter>& rois, int id = 0);
	cv::Mat m_undist_mask; // mask for image undistortion valid area 
	std::vector<cv::Mat> m_scene_masks; // mask for scene
	vector<cv::Mat> m_backgrounds; 
	std::vector<cv::Mat> m_foreground;

	void readSceneMask(); 
	void extractFG(); 

	void pureTracking(); 

	Renderer* mp_renderEngine; 

	vector<std::shared_ptr<PigSolverDevice> >       mp_bodysolverdevice;
	vector<std::shared_ptr<PigSolver> > mp_bodysolver; 
	vector<vector<Eigen::Vector4f> > m_projectedBoxesLast; // pigid, camid
	std::vector<cv::Mat> m_rawMaskImgs;
	void drawRawMaskImgs();
	std::string result_folder; 
	bool is_smth;
protected:
    // io functions 
    void setCamIds(std::vector<int> _camids); 
    void assembleDets(); 
    void detNMS(); 
    int _compareSkel(const vector<Eigen::Vector3f>& skel1, const vector<Eigen::Vector3f>& skel2); 
    int _countValid(const vector<Eigen::Vector3f>& skel); 

    void drawSkel(cv::Mat& img, const vector<Eigen::Vector3f>& _skel2d, int colorid);
	void drawSkelDebug(cv::Mat& img, const vector<Eigen::Vector3f>& _skel2d);
	vector<cv::Mat> drawMask();  // can only be used after association 
	void getChamferMap(int pid, int viewid, cv::Mat& chamfer);

	int m_imh; 
    int m_imw; 
    int m_frameid; 
    int m_camNum; 
    std::vector<int>                          m_camids;
    std::vector<Eigen::Vector3i>              m_CM;
    vector<vector<vector<Eigen::Vector3f> > > m_projs; // [viewid, candid, kptid]

    std::vector<Camera>                       m_cams; 
    std::vector<cv::Mat>                      m_imgs; 
    std::vector<Camera>                       m_camsUndist; 
    std::vector<cv::Mat>                      m_imgsUndist; 

	std::vector<cv::Mat>                      m_imgsDetect;
	std::vector<cv::Mat>                      m_imgsOverlay; 

    vector<vector<DetInstance> >              m_detUndist; // [camnum, candnum]
    vector<MatchedInstance>                   m_matched; // matched raw data after matching()
	vector<vector<DetInstance> >              m_unmatched; // [camnum, candnum]
	

    // matching & 3d data 
    vector<vector<int> > m_clusters; // pigid, camid [candid]
    vector<vector<Eigen::Vector3f> > m_skels3d; 
    vector<vector<Eigen::Vector3f> > m_skels3d_last;

    // config data, set by confByJson() 
    std::string m_sequence; 
    float       m_epi_thres; 
    std::string m_epi_type;
    float       m_boxExpandRatio; 
    std::string m_skelType; 
    SkelTopology m_topo; 
    int m_startid;
    int m_framenum; 

    // io function and tmp data
    vector<vector<vector<Eigen::Vector3f> > > m_keypoints; // [viewid, candid, kptid]
    vector<vector<Eigen::Vector4f> >          m_boxes_raw; // xyxy
    vector<vector<vector<vector<Eigen::Vector2f> > > > m_masks; // mask in contours 
    vector<vector<vector<Eigen::Vector3f> > > m_keypoints_undist; 
    vector<vector<Eigen::Vector4f> >          m_boxes_processed; // camid, candid
    vector<vector<vector<vector<Eigen::Vector2f> > > > m_masksUndist; 
    std::string m_boxDir; 
    std::string m_maskDir;  
    std::string m_keypointsDir; 
    std::string m_imgExtension; 
    std::string m_camDir; 
    std::string m_imgDir; 
	std::string m_pigConfig; 
	std::string m_match_alg; 
    void readImages(); 
    void readCameras(); 
    void readKeypoints(); 
    void readBoxes(); 
    void readMask(); 
    void processBoxes(); 
    void undistKeypoints(); 
    void undistImgs(); 
    void undistMask(); 
    
	void to_left_clean(DetInstance& det);
};

