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

using std::vector; 

typedef vector<vector<Eigen::Vector3d> > PIGS3D;
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
    SkelTopology get_topo(){return m_topo;}
    PIGS3D get_skels3d(){return m_skels3d;}

    void configByJson(std::string jsonfile); 
    void fetchData(); 
    cv::Mat test(); 

    // top-down matching
    void matching(); 
    void tracking(); 
    void reproject_skels(); 

    // visual function  
    cv::Mat visualizeSkels2D(); 
    cv::Mat visualizeIdentity2D();
    cv::Mat visualizeProj(); 
    void writeSkel3DtoJson(std::string savepath); 
    void readSkel3DfromJson(std::string jsonfile); 

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
    void detNMS(); 
    int _compareSkel(const vector<Vec3>& skel1, const vector<Vec3>& skel2); 
    int _countValid(const vector<Vec3>& skel); 

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

    // matching & 3d data 
    vector<vector<int> > m_clusters; 
    vector<vector<Eigen::Vector3d> > m_skels3d; 
    vector<vector<Eigen::Vector3d> > m_skels3d_last;

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
    std::string m_skelType; 
    SkelTopology m_topo; 
    int m_startid;
    int m_framenum; 
};

