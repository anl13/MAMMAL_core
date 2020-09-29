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
//#include "../articulation/pigsolver.h"

using std::vector; 

// data of a frame, together with matching process
class FrameData{
public: 
    FrameData(){
        m_camids = {0,1,2,5,6,7,8,9,10,11};
        m_camNum = 10; 
        m_imw = 1920; 
        m_imh = 1080; 
        getColorMap("anliang_render", m_CM); 

    }
    ~FrameData(){} 

    // attributes 
    void set_frame_id(int _frameid){m_frameid = _frameid;}
    int get_frame_id() {return m_frameid;}

	vector<cv::Mat> get_imgs_undist() { return m_imgsUndist; }
    vector<Camera> get_cameras(){return m_camsUndist; }

    virtual void configByJson(std::string jsonfile); 
    void fetchData();

    // io functions 
    
	SkelTopology m_topo;

	int m_imh; 
    int m_imw; 
    int m_frameid; 
    int m_camNum; 
    std::vector<int>                          m_camids;
    std::vector<Eigen::Vector3i>              m_CM;

    std::vector<Camera>                       m_cams; 
    std::vector<cv::Mat>                      m_imgs; 
    std::vector<Camera>                       m_camsUndist; 
    std::vector<cv::Mat>                      m_imgsUndist; 

	std::vector<cv::Mat>                      m_imgsDetect;
	std::vector<cv::Mat>                      m_imgsOverlay; 
    vector<vector<DetInstance> >              m_detUndist; // [camnum, candnum]
	std::string m_skelType;
    std::string m_sequence; 
    float       m_boxExpandRatio; 

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
    void readImages(); 
    void readCameras(); 
    void readKeypoints(); 
    void readBoxes(); 
    void readMask(); 
    void processBoxes(); 
    void undistKeypoints(); 
    void undistImgs(); 
    void undistMask(); 

protected:
	void assembleDets();

};

