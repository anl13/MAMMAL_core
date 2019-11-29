#include "skel.h" 
#include <iostream> 

SkelTopology getSkelTopoByType(std::string type)
{
    SkelTopology A; 
    if(type=="PIG20")
    {
        A.joint_num = 20; 
        A.bone_num = 19; 
        A.bones = {
            {18, 19}, {19,0}, {18, 9},  // main body 
            {0,1}, {1,2}, {0,10}, {10,11}, // face
            {19,12}, {19,3}, 
            {12, 13}, {13,14}, {15,16}, {16,17}, // right legs
            {3,4}, {4,5}, {6,7}, {7,8}, // left legs 
            {9, 15}, {9,6}
        };
        A.label_names = {
            "nose",
            "left eye", 
            "left ear", 
            "left shoulder", 
            "left elbow", 
            "left hand",    // 5
            "left leg", 
            "left knee", 
            "left foot",    // 8
            "tail root",    // 9
            "right eye", 
            "right ear", 
            "right shoulder", 
            "right elbow", 
            "right hand",   // 14
            "right leg", 
            "right knee", 
            "right foot",   // 17
            "center", 
            "ear middle"    // this is virtual keypoint
        };
        A.kpt_color_ids = {
            0,0,0, // left face 
            1,1,1, // left front leg
            3,3,3, // left back leg
            2,0,0, // tail and right face
            4,4,4, // right front leg
            5,5,5, // right back leg 
            2,2    // center and ear middle
        };
        A.kpt_conf_thresh = {  // confidence for openpose 
            0.4, //  0, nose  
            0.4, //  1, left eye
            0.6, //  2, left ear 
            0.3, //  3, left shoulder
            0.6, //  4, left elbow
            0.6, //  5, left hand
            0.3, //  6, left leg
            0.6, //  7, left knee
            0.6, //  8, left foot
            0.4, //  9, tail root
            0.4, //  10, right eye 
            0.6, //  11, right ear
            0.3, //  12, right shoulder
            0.6, //  13, right elbow
            0.6, //  14, right hand
            0.3, //  15, right leg
            0.6, //  16, right knee
            0.6, //  17, right foot
            0.15, //  18, center
            0.2  //  19, ear middle (neck)
        }; 
    }
    else if (type=="PIG15")
    {
        A.joint_num = 15; 
        A.bone_num = 14; 
        A.label_names = {
            "left_ear", 
            "right_ear", 
            "nose", 
            "right_shoulder", 
            "right_front_paw", 
            "left_shoulder", 
            "left_front_paw", 
            "right_hip", 
            "right_knee", 
            "right_back_paw", 
            "left_hip", 
            "left_knee", 
            "left_back_paw", 
            "root_of_tail", 
            "center"
        };
        A.bones = {
            {0,2}, {1,2}, {2,14}, {5,6}, {5,14}, {3,4}, {3,14}, 
            {13,14}, {9,8}, {8,7}, {7,13}, {12,11}, {11,10}, {10,13}
        };
        A.kpt_color_ids = {
            0,0,0, // face 
            1,1, // right front leg 
            2,2, // left front leg 
            3,3,3, 4,4,4, // back leg
            2,2 // ceneter and tail 
        }; 
        A.kpt_conf_thresh = {
            0.3, //  0, left ear 
            0.3, //  1, right ear 
            0.3, //  2, nose
            0.5, //  3, right shoulder
            0.5, //  4, right front paw
            0.5, //  5, left shoulder
            0.5, //  6, left front paw
            0.5, //  7, right hip 
            0.5, //  8, right knee
            0.5, //  9, right back paw
            0.5, //  10, left hip 
            0.5, //  11, left knee 
            0.5, //  12, left back paw
            0.3, //  13, root of tail 
            0.1  //  14, center 
        }; 
    }
    else 
    {
        std::cout << "skel type " << type << " not implemented yet" << std::endl;
        exit(-1); 
    }
    
    return A; 
}
