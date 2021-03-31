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
#include "../utils/skel.h" 

using std::vector; 

// data of a frame, together with matching process
class FrameData{
public: 
    FrameData(){
        m_imw = 1920; 
        m_imh = 1080; 
        getColorMap("anliang_render", m_CM); 
    }
    ~FrameData(){} 

    // attributes 
	void set_frame_id(int _frameid);
    int get_frame_id() {return m_frameid;}

	vector<cv::Mat> get_imgs_undist() { return m_imgsUndist; }
    vector<Camera> get_cameras(){return m_camsUndist; }

    virtual void configByJson(std::string jsonfile); 
    void         fetchData();
	cv::Mat visualizeSkels2D();
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
	std::vector<cv::VideoCapture>             m_caps;

    std::string m_boxDir; 
    std::string m_maskDir;  
    std::string m_keypointsDir; 
    std::string m_imgExtension; 
    std::string m_camDir; 
    std::string m_imgDir; 
	int         m_hourid;
	int         m_pignum; 
    void readImages(); 
    void readCameras(); 
    void readKeypoints(); 
    void readBoxes(); 
    void readMask(); 
    void processBoxes(); 
    void undistKeypoints(); 
    void undistImgs(); 
    void undistMask(); 
	void readUndistImages(); 
	void readImagesFromVideo();
	
protected:
	void assembleDets();
	cv::Mat m_map1; 
	cv::Mat m_map2;
	void initRectifyMap();
	bool m_is_video; 
	bool m_is_read_image;
	bool m_video_frameid;
};

