#pragma once 

#include <iostream> 
#include <fstream> 
#include <iomanip>
#include <Eigen/Eigen> 
#include <json/json.h> 
#include <vector> 
#include <opencv2/opencv.hpp> 
#include "math_utils.h"
#include "camera.h"
#include "image_utils.h"
#include "geometry.h" 
#include "clusterclique.h"
#include "Hungarian.h"
#include "skel.h" 
#include "matching.h"
#include "parsing.h" 

using std::vector; 


// data of a frame, together with matching process
class FrameData{
public: 
    FrameData(){
        m_camids = {0,1,2,5,6,7,8,9,10,11};
        m_camNum = 10; 
        m_frameid = -1; 
        m_imw = 1920; 
        m_imh = 1080; 
        getColorMap("anliang_rgb", m_CM); 
        startid = -1; 
        framenum = 0; 
        m_epi_thres = -1; 
        m_epi_type = "p2l";
    }
    ~FrameData(){} 
    int startid;
    int framenum; 

    void setFrameId(int _frameid){m_frameid=_frameid;}
    void configByJson(std::string jsonfile); 
    void fetchData(); 
    cv::Mat test(); 

    // top-down matching
    EpipolarMatching m_matcher; 
    void matching(); 
    vector<vector<int> > m_clusters; 
    vector<vector<Eigen::Vector3d> > m_skels3d; 
    void reproject_skels(); 

    // ransac based proposals (joint proposals)
    vector<vector<ConcensusData> > m_concensus; // [kptnum, candnum]
    // void ransacProposals(); 
    // void projectProposals(); 
    double m_ransac_nms_thres; 
    double m_sigma; // init reprojection threshold

    // visual function  
    cv::Mat visualizeSkels2D(); 
    cv::Mat visualizeIdentity2D();
    cv::Mat visualizeProj(); 
    // void writeSkeltoJson(std::string savepath); 
    // void readSkelfromJson(std::string jsonfile); 

protected:
    // io functions 
    void setCamIds(std::vector<int> _camids); 
    void readImages(); 
    void readCameras(); 
    void readKeypoints(); 
    void readBoxes(); 
    void undistKeypoints(); 
    void undistImgs(); 
    void processBoxes(); 

    void drawSkel(cv::Mat& img, const vector<Eigen::Vector3d>& _skel2d, int colorid);
    int m_imh; 
    int m_imw; 
    int m_frameid; 
    int m_camNum; 
    std::vector<Camera>                       m_cams; 
    std::vector<Camera>                       m_camsUndist; 
    std::vector<cv::Mat>                      m_imgs; 
    std::vector<cv::Mat>                      m_imgsUndist; 
    std::vector<int>                          m_camids;
    std::vector<Eigen::Vector3i>              m_CM;
    vector<vector<vector<Eigen::Vector3d> > > m_dets; // [viewid, candid, kptid]
    vector<vector<vector<Eigen::Vector3d> > > m_dets_undist; 
    vector<vector<Eigen::Vector4d> >          m_boxes_raw;
    vector<vector<Eigen::Vector4d> >          m_boxes_processed; 
    vector<vector<vector<Eigen::Vector3d> > > m_projs; // [viewid, candid, kptid]


    // config data, set by confByJson() 
    std::string m_sequence; 
    std::string m_keypointsDir; 
    std::string m_boxDir; 
    std::string m_camDir; 
    std::string m_imgDir; 
    std::string m_imgExtension; 
    double      m_epi_thres; 
    std::string m_epi_type;
    double      m_boxExpandRatio; 
    double      m_pruneThreshold; 
    int         m_cliqueSizeThreshold; 
    std::string m_skelType; 
    SkelTopology m_topo; 
};

