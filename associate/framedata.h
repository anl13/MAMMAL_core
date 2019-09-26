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

using std::vector; 

// data of a frame 
class FrameData{
public: 
    FrameData(){
        m_camids = {0,1,2,5,6,7,8,9,11};
        m_camNum = 9; 
        m_frameid = -1; 
        m_imw = 1920; 
        m_imh = 1080; 
        getColorMap("anliang", m_CM); 
        kpt_color_id = {
                0,0,0, // left face 
                1,1,1, // left front leg
                3,3,3, // left back leg
                2,0,0, // tail and right face
                4,4,4, // right front leg
                5,5,5, // right back leg 
                2,2    // center and ear middle
        };
        m_cliques.resize(20); 
        m_tables.resize(20); 
        m_invTables.resize(20); 
        m_G.resize(20); 
        startid = -1; 
        framenum = 0; 
        m_epi_thres = -1; 
        m_epi_type = "p2l";
        m_keypoint_conf_thres.resize(20, 0); 
    }
    ~FrameData(){} 

    void setFrameId(int _frameid){m_frameid=_frameid;}
    void configByJson(std::string jsonfile); 

    void fetchData(); 
    cv::Mat test(); 
    void checkEpipolar(int kpt_id); 
    void epipolarClustering(int kpt_id, vector<Vec3> &p3ds);
    void compute3d(); 
    void reproject(); 
    cv::Mat visualize(int type, int kpt_id=-1); 
    cv::Mat visualizeClique(int kpt_id); 

    vector<vector<vector<Eigen::Vector3d> > > dets; 
    vector<vector<vector<Eigen::Vector3d> > > dets_undist; 
    vector<vector<Eigen::Vector3d> > points3d; 
    vector<vector<vector<Eigen::Vector3d> > > dets_reproj; 
    vector<vector<vector<int> > > m_cliques; 
    vector<vector<std::pair<int,int> > > m_tables;
    vector<vector<vector<int > > > m_invTables;  
    vector<Eigen::MatrixXd> m_G; 

    int startid;
    int framenum; 
    std::vector<int> m_kpts_to_show;

private:
    void setCamIds(std::vector<int> _camids); 

    void readImages(); 
    void readCameras(); 
    void readKeypoints(); 
    void readKeypoint(std::string jsonfile);
    void undistKeypoints(const Camera& cam, const Camera& camnew, int imw, int imh); 
    void undistImgs(); 

    int m_imh, m_imw; 
    int m_frameid; 
    int m_camNum; 
    std::vector<Camera> m_cams; 
    std::vector<Camera> m_camsUndist; 
    std::vector<cv::Mat> m_imgs; 
    std::vector<cv::Mat> m_imgsUndist; 
    std::vector<int> m_camids;
    std::vector<Eigen::Vector3i> m_CM;
    std::vector<int> kpt_color_id;

    // config data
    std::string m_keypointsDir; 
    std::string m_camDir; 
    std::string m_imgDir; 
    std::string m_imgExtension; 
    double m_epi_thres; 
    std::string m_epi_type; 
    std::vector<double> m_keypoint_conf_thres; 
};

