#include "skel.h" 



 std::vector<std::string> LABEL_NAMES = {
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



std::vector<std::vector<int> > AFF = 
{
    {1, 10, 19, 2, 11}, 
    {0, 2, 10, 19, 11},
    {1, 19, 3, 11, 10, 0}, 
    {2, 19, 18, 4, 5, 6, 7, 8, 12, 13, 14}, 
    {3, 5, 2, 6, 18, 19, 12, 13, 14}, 
    {3, 4, 18, 2, 6, 7, 8, 12, 13, 14}, 
    {3, 7, 8, 9, 18, 15, 4, 5, 15, 16, 17}, 
    {6, 8, 3, 9, 18, 4, 5, 15, 16, 17}, 
    {6, 7, 3, 9, 18, 4, 5, 15, 16, 17}, 
    {6, 15, 18, 7, 16},
    {0, 11, 1, 19, 2}, 
    {10, 19, 12, 0, 1, 2}, 
    {11, 15, 13, 14, 18, 19, 16, 17, 3, 4, 5}, 
    {12, 14, 18, 15, 11, 16, 17, 3, 4, 5}, 
    {12, 13, 18, 11, 15, 16, 17, 3, 4, 5}, 
    {12, 9, 18, 16, 17, 13, 14, 6, 7, 8}, 
    {15, 17, 12, 9, 18, 13, 14, 6, 7, 8}, 
    {15, 16, 12, 9, 18, 13, 14, 6, 7, 8}, 
    {12, 15, 3, 6, 19, 9}, 
    {11, 2, 0, 1, 10}
};

// parents, define articulated skeleton
std::vector<int> PA = 
{
    19, 
    0, 1, 19, 3, 4, 9, 6, 7,
    18,
    0, 10, 19, 12, 13, 9, 15, 16, 
    -1, 18
};

// this bone lens data is computed 
// at 3732nd frame of morning sequence. 
std::vector<float> BONES_LEN =
{
    0.246, 0.146, 0.221, // main body 
    0.090, 0.060, 0.090, 0.060, //face
    0.151, 0.151, 
    0.084, 0.054, 0.081, 0.080, // right legs
    0.064, 0.077, 0.057, 0.064, //left legs 
    0.116, 0.148
};

std::vector<Eigen::Vector2i> BONES = 
{
    {18, 19}, {19,0}, {18, 9},  // main body 
    {0,1}, {1,2}, {0,10}, {10,11}, // face
    {19,12}, {19,3}, 
    {12, 13}, {13,14}, {15,16}, {16,17}, // right legs
    {3,4}, {4,5}, {6,7}, {7,8}, // left legs 
    {9, 15}, {9,6}
};