#include <opencv2/opencv.hpp> 
#include <Eigen/Eigen> 
#include <fstream> 
#include <iostream> 
#include <sstream>
#include <string> 
#include <vector> 
#include <iomanip> 

#include "../utils/math_utils.h" 
#include "../utils/camera.h" 
#include "../utils/image_utils.h"
#include "../associate/framedata.h"
#include "../utils/geometry.h"
#include "../associate/skel.h" 

#include <gflags/gflags.h> 

DEFINE_string(debug_type, "assoc", "debug type"); 

int test_topdown(bool is_vis=false)
{
    FrameData frame; 
    std::string configFile = "/home/al17/animal/animal_calib/associate/config.json"; 
    frame.configByJson(configFile); 

    std::string videoname = "/home/al17/animal/animal_calib/result_data/reproj_topdown.avi"; 
    cv::VideoWriter writer(videoname, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 25.0, cv::Size(1920*4, 1080*3)); 
    if(!writer.isOpened())
    {
        std::cout << "can not open video file " << videoname << std::endl; 
        return -1; 
    }

    std::string videoname_det = "/home/al17/animal/animal_calib/result_data/det_topdown.avi"; 
    cv::VideoWriter writer_det(videoname_det, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 25.0, cv::Size(1920*4, 1080*3)); 
    if(!writer_det.isOpened())
    {
        std::cout << "can not open video file " << videoname << std::endl; 
        return -1; 
    }

    int start_id = frame.get_start_id(); 
    int frame_num = frame.get_frame_num(); 
    for(int frameid = start_id; frameid < start_id + frame_num; frameid++)
    {
        std::cout << "Run frame " << frameid << std::endl; 
        frame.set_frame_id(frameid); 
        frame.fetchData(); 
        frame.matching(); 
        frame.tracking(); 
    
        frame.reproject_skels();
        cv::Mat img1 = frame.visualizeIdentity2D(); 
        cv::Mat img = frame.visualizeProj(); 
        writer_det.write(img1); 
        writer.write(img); 
        std::stringstream ss; 
        ss << "/home/al17/animal/animal_calib/result_data/skels3d/skel_" 
           << std::setw(6) << std::setfill('0') << frameid << ".json";
        frame.writeSkel3DtoJson(ss.str()); 
        if(is_vis)
        {

            cv::namedWindow("assoc", cv::WINDOW_NORMAL); 
            cv::imshow("assoc", img1); 
            cv::namedWindow("projection", cv::WINDOW_NORMAL); 
            cv::imshow("projection", img); 
            int key = cv::waitKey(); 
            if(key == 27) break;  
        }
        
    }
}

int test_readingdata(bool is_vis=false)
{
    FrameData frame; 
    std::string configFile = "/home/al17/animal/animal_calib/associate/config.json"; 
    frame.configByJson(configFile); 

    int start_id = frame.get_start_id(); 
    int frame_num = frame.get_frame_num(); 
    for(int frameid = start_id; frameid < start_id + frame_num; frameid++)
    {
        std::cout << "Run frame " << frameid << std::endl; 
        frame.set_frame_id(frameid); 
        frame.fetchData(); 
        cv::Mat vis_det = frame.visualizeSkels2D(); 
        std::stringstream ss; 
        
        if(is_vis)
        {
            cv::namedWindow("detection", cv::WINDOW_NORMAL); 
            cv::imshow("detection", vis_det); 
            int key = cv::waitKey(); 
            if(key == 27) break;  
        }
    }
}


int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); 
    // test_readingdata(true); 
    test_topdown(false); 

    return 0; 
}