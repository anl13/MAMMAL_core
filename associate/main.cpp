#include <opencv2/opencv.hpp> 
#include <Eigen/Eigen> 
#include <fstream> 
#include <iostream> 
#include <sstream>
#include <string> 
#include <vector> 

#include "math_utils.h" 
#include "camera.h" 
#include "image_utils.h"
#include "framedata.h"
#include "geometry.h" 

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

int test3d()
{
    FrameData frame; 
    std::string configFile = "/home/al17/animal/animal_calib/associate/config.json"; 
    frame.configByJson(configFile); 

    std::vector< PIG_SKEL > last_skels; 
    for(int frameid = frame.startid; frameid < frame.startid + frame.framenum; frameid++)
    {
        frame.setFrameId(frameid); 
        std::cout << "set frame id" << frameid << std::endl; 
        frame.fetchData(); 
        std::cout << "fetch data" << std::endl; 

        frame.compute3d(); 
        frame.reproject(); 
        if(frameid == frame.startid)
        {
            frame.associateNearest(); 
            last_skels = frame.m_skels;         
        }
        else 
        {
            frame.track3DJoints(last_skels); 
            last_skels = frame.m_skels; 
        }
        frame.clean3DJointsByAFF(); 
        frame.reproject_skels(); 
        frame.computeBoneLen();

        
        for(int i = 0; i < 4; i++) std::cout << frame.m_skels[i] << std::endl;
       
        cv::Mat img_show_id = frame.visualizeIdentity2D(); 
        cv::namedWindow("identity", cv::WINDOW_NORMAL); 
        cv::imshow("identity", img_show_id); 
        std::stringstream ss; 
        ss << "results/with_id/" << std::setw(6) << std::setfill('0') << frameid << ".png";
        cv::imwrite(ss.str(), img_show_id); 

        std::stringstream ss1; 
        ss1 << "results/json/" << std::setw(6) << std::setfill('0') << frameid << ".json";
        frame.writeSkeltoJson(ss1.str()); 

        int key = cv::waitKey(1); 
        if (key == 27) break; 
        // for(int i = 0; i < frame.m_kpts_to_show.size(); i++)
        // {
        //     int kpt_id = frame.m_kpts_to_show[i]; 
        // // for(int kpt_id = 0; kpt_id < 20; kpt_id++)
        // // {
        //     cv::Mat raw_det = frame.visualize(1, kpt_id); 
        //     cv::Mat reproj = frame.visualize(2, kpt_id); 
        //     cv::Mat assoc = frame.visualizeClique(kpt_id); 

        //     // cv::namedWindow("raw", cv::WINDOW_NORMAL); 
        //     cv::namedWindow("proj", cv::WINDOW_NORMAL); 
        //     cv::namedWindow("assoc", cv::WINDOW_NORMAL); 
        //     // cv::imshow("raw", raw_det); 
        //     cv::imshow("proj", reproj); 
        //     cv::imshow("assoc", assoc); 
        //     // std::stringstream ss1; ss1 << "results/association_debug2/reproj_" << kpt_id << "_" << frameid << ".png";cv::imwrite(ss1.str(), reproj); 
        //     // std::stringstream ss2; ss2 << "results/association_debug2/assoc_" << kpt_id << "_" << frameid << ".png"; cv::imwrite(ss2.str(), assoc); 
        //     int key = cv::waitKey(); 
        //     if(key == 27) exit(-1); 
        // }


        // cv::Mat raw_det = frame.visualize(1, -1); 
        // cv::Mat reproj = frame.visualize(2, -1); 
        // cv::namedWindow("raw_detection", cv::WINDOW_NORMAL); 
        // cv::namedWindow("reprojection", cv::WINDOW_NORMAL); 
        // cv::imshow("raw_detection", raw_det); 
        // cv::imshow("reprojection", reproj); 
        // int key = cv::waitKey(); 
        // if(key == 27) break; 
        // else if(char(key) == 'p')
        // {
        //     cv::waitKey(); 
        // }

    }
    cv::destroyAllWindows(); 
    
    return 0; 
}

int main()
{
    // test2d_write_video(); 
    test3d(); 

    return 0; 
}