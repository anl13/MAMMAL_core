#include <opencv2/opencv.hpp> 
#include <Eigen/Eigen> 
#include <fstream> 
#include <iostream> 
#include <sstream>
#include <string> 
#include <vector> 

#include "../associate/math_utils.h" 
#include "../associate/camera.h" 
#include "../associate/image_utils.h"
#include "../associate/framedata.h"
#include "../associate/geometry.h"
#include "../associate/skel.h" 

#include <gflags/gflags.h> 

DEFINE_string(debug_type, "assoc", "debug type"); 

int test2d_write_video()
{
    FrameData frame; 
    std::string configFile = "/home/al17/animal/animal_calib/associate/config.json"; 
    frame.configByJson(configFile); 

    std::string videoname = "/home/al17/animal/animal_calib/data/keypoints_noon10000.avi"; 
    cv::VideoWriter writer(videoname, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 25.0, cv::Size(1920*4, 1080*3)); 
    if(!writer.isOpened())
    {
        std::cout << "can not open video file " << videoname << std::endl; 
        return -1; 
    }

    for(int frameid = 0; frameid < 1000; frameid++)
    {
        frame.setFrameId(frameid); 
        std::cout << "set frame id" << frameid << std::endl; 
        frame.fetchData(); 
        std::cout << "fetch data" << std::endl; 
        cv::Mat show = frame.test(); 
        cv::namedWindow("show", cv::WINDOW_NORMAL); 
        cv::imshow("show", show); 
        writer.write(show); 
        int key = cv::waitKey(1); 
        if(key == 27){
            break; 
        }
    }
    cv::destroyAllWindows(); 
    writer.release(); 
    return 0; 
}

int test_topdown()
{
    FrameData frame; 
    std::string configFile = "/home/al17/animal/animal_calib/associate/config.json"; 
    frame.configByJson(configFile); 
    for(int frameid = frame.startid; frameid < frame.startid+frame.framenum; frameid++)
    {
        frame.setFrameId(frameid); 
        std::cout << "set frame id" << frameid << std::endl; 
        frame.fetchData(); 
        std::cout << "fetch data" << std::endl; 
        frame.matching(); 
        std::cout << "match ok" << std::endl; 
        frame.reproject_skels();
        std::cout << "reproj ok" << std::endl; 
        cv::Mat img1 = frame.visualizeIdentity2D(); 
        cv::Mat img = frame.visualizeProj(); 
        cv::namedWindow("detection", cv::WINDOW_NORMAL); 
        cv::imshow("detection", img1); 
        cv::namedWindow("projection", cv::WINDOW_NORMAL); 
        cv::imshow("projection", img); 
        int key = cv::waitKey(); 
        if(key == 27) break;  
    }
}

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); 
    test_topdown(); 

    return 0; 
}