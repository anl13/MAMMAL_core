#include "skel.h" 
#include <iostream> 
#include <fstream>

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
    else if (type=="UNIV") // universal animal model 
    {
        A.joint_num = 23; 
        A.bone_num = 23; 
        A.label_names = {
         "nose", "eye_left", "eye_right", "ear_root_left", "ear_root_right", 
        "shoulder_left", "shoulder_right", "elbow_left", "elbow_right", "paw_left", "paw_right", 
        "hip_left", "hip_right", "knee_left", "knee_right", "foot_left", "foot_right", 
        "neck", "tail_root", "withers", "center", 
        "tail_middle", "tail_end"
        };
        A.bones = {
        {0,1}, {0,2}, {1,2}, {1,3}, {2,4},
        {0,17}, {17,5},{17,6}, {5,7}, {7,9}, {6,8}, {8,10},
        {17,19}, {19,20}, {20,18}, {18,21}, {21,22},
        {18,11}, {18,12}, {11,13}, {13,15}, {12,14}, {14,16}
        };
        A.kpt_color_ids = {
            0,0,0,0,0, // face 
            2,1,2,1,2,1, // front legs 
            4,3,4,3,4,3, // back legs 
            5,9,5,6,5,5 // ceneter and tail 
        }; 
        A.kpt_conf_thresh = {
            0.5, // nose 0
            0.5, // eye left  1
            0.5, // eye right   2 
            0.5, // ear root left 3
            0.5, // ear root right 4
            0.5, // left shoulder 5
            0.5, // right shoulder 6
            0.5, // left elbow 7
            0.5, // right elbow 8
            0.5, // left paw 9
            0.5, // right paw 10
            0.5, // hip left 11
            0.5, // hip right  12
            0.5, // knee left  13
            0.5, // knee right  14
            0.5, // foot left 15
            0.5, // foot right 16
            0.5, // neck 17
            0.5, // tail root 18
            0.5, // withers    19
            0.5, // center 20
            0.5, // tail middle  21
            0.5  // tail end 22
        }; 
    }
    else 
    {
        std::cout << "skel type " << type << " not implemented yet" << std::endl;
        exit(-1); 
    }
    
    return A; 
}

vector<std::pair<int, int> > getPigMapper()
{
	std::vector<std::pair<int, int> > Mapper = {
		{ 1, 239 }, // nose
	{ 1, 50 }, // left eye
	{ 1, 353 }, // right eye
	{ 1, 1551 }, // left ear 
	{ 1, 1571 }, // right ear 
	{ 0, 21 },
	{ 0, 6 },
	{ 0, 22 },
	{ 0, 7 },
	{ 0, 23 },
	{ 0,8 },
	{ 0, 39 },
	{ 0, 27 },
	{ 0,40 },
	{ 0,28 },
	{ 0,41 },
	{ 0,29 },
	{ -1, -1 },
	{ 0, 31 },
	{ -1, -1 },
	{ 0, 2 },
	{ -1, -1 },
	{ -1,-1 }
	};
	return Mapper; 
}

void BodyState::saveState(std::string filename)
{
	std::ofstream os(filename);
	if (!os.is_open())
	{
		std::cout << "Could not open " << filename << std::endl;
		exit(-1); 
	}
	// 1. trans, double * 3 
	os << trans << std::endl; 
	// 2. pose, double * 43 * 3
	os << pose << std::endl; 
	// 3. alpha, double * 1 
	os << scale << std::endl; 
	// 4. id, int * 1
	os << frameid << std::endl;
	os << id << std::endl; 
	// 5. points, double * 3 *3 
	os << points[0] << std::endl
		<< points[1] << std::endl
		<< points[2] << std::endl; 
	os.close(); 
}

void BodyState::loadState(std::string filename)
{
	std::ifstream is(filename); 
	if (!is.is_open())
	{
		std::cout << "could not open " << filename << std::endl; 
		exit(-1); 
	}
	for (int i = 0; i < 3; i++)is >> trans(i);
	pose.resize(43 * 3);
	for (int i = 0; i < 43 * 3; i++) is >> pose(i);
	is >> scale; 
	is >> frameid;
	double d_id; 
	is >> d_id; id = int(d_id); 
	points.resize(3); 
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			is >> points[i](j);
		}
	}
	center = points[1]; 
	is.close(); 
}


vector<Eigen::Vector3d> convertMatToVec(const Eigen::MatrixXd& skel)
{
	vector<Eigen::Vector3d> vec;
	vec.resize(skel.cols()); 
	for (int i = 0; i < skel.cols(); i++)
	{
		vec[i] = skel.col(i); 
	}
	return vec; 
}