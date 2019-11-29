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
        frame.epipolarSimilarity(); 
        frame.compute3d(); 
        frame.reproject(); 


        if(FLAGS_debug_type == "tracking")
        {
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
            ss << "results/tracking_debug/" << std::setw(6) << std::setfill('0') << frameid << ".png";
            cv::imwrite(ss.str(), img_show_id); 

            std::stringstream ss1; 
            ss1 << "results/json/" << std::setw(6) << std::setfill('0') << frameid << ".json";
            frame.writeSkeltoJson(ss1.str()); 

            int key = cv::waitKey(1); 
            if (key == 27) break; 
        }

        if(FLAGS_debug_type == "assoc")
        {
            for(int i = 0; i < frame.m_kpts_to_show.size(); i++)
            {
                int kpt_id = frame.m_kpts_to_show[i]; 

                cv::Mat raw_det = frame.visualize(1, kpt_id); 
                cv::Mat reproj = frame.visualize(2, kpt_id); 
                cv::Mat assoc = frame.visualizeClique(kpt_id); 

                // cv::namedWindow("raw", cv::WINDOW_NORMAL); 
                cv::namedWindow("proj", cv::WINDOW_NORMAL); 
                cv::namedWindow("assoc", cv::WINDOW_NORMAL); 
                // cv::imshow("raw", raw_det); 
                cv::imshow("proj", reproj); 
                cv::imshow("assoc", assoc); 
                // std::stringstream ss1; ss1 << "results/assoc_debug_22/reproj_" << kpt_id << "_" << frameid << ".png";cv::imwrite(ss1.str(), reproj); 
                // std::stringstream ss2; ss2 << "results/assoc_debug_22/assoc_" << kpt_id << "_" << frameid << ".png"; cv::imwrite(ss2.str(), assoc); 
                int key = cv::waitKey(); 
                if(key == 27) exit(-1); 
            }
        }
    }
    cv::destroyAllWindows(); 
    
    return 0; 
}


int test_proposals()
{
    FrameData frame; 
    std::string configFile = "/home/al17/animal/animal_calib/associate/config.json"; 
    frame.configByJson(configFile); 

    for(int frameid = frame.startid; frameid < frame.startid + frame.framenum; frameid++)
    {
        frame.setFrameId(frameid); 
        std::cout << "set frame id" << frameid << std::endl; 
        frame.fetchData(); 
        std::cout << "fetch data" << std::endl; 

        frame.epipolarSimilarity(); 
        frame.ransacProposals(); 

        // frame.projectProposals(); 
        frame.parsing(); 
        frame.reproject_skels(); 
        cv::Mat img_show_id = frame.visualizeIdentity2D(); 
        cv::namedWindow("parsing", cv::WINDOW_NORMAL); 
        cv::imshow("parsing", img_show_id); 
        int key = cv::waitKey(); 
        if(key == 27) break; 
    }
    
    return 0; 
}

int test_box()
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
        cv::Mat img = frame.test();
        cv::namedWindow("box", cv::WINDOW_NORMAL); 
        cv::imshow("box", img); 
        int key = cv::waitKey(); 
        if(key == 27) break;  
    }
}

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); 
    // test2d_write_video(); 
    // test3d(); 
    test_proposals(); 
    // test_box(); 

    return 0; 
}