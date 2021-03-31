#include "skel.h" 
#include <iostream> 
#include <fstream>
#include "geometry.h"

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
        A.bone_num = 19; 
        A.label_names = {
         "nose", "eye_left", "eye_right", "ear_root_left", "ear_root_right", 
        "shoulder_left", "shoulder_right", "elbow_left", "elbow_right", "paw_left", "paw_right", 
        "hip_left", "hip_right", "knee_left", "knee_right", "foot_left", "foot_right", 
        "neck", "tail_root", "withers", "center", 
        "tail_middle", "tail_end"
        };
        A.bones = {
        {0,1}, {0,2}, {1,2}, {1,3}, {2,4},
         {5,7}, {7,9}, {6,8}, {8,10},
        {20,18},
        {18,11}, {18,12}, {11,13}, {13,15}, {12,14}, {14,16},
		{0,20},{5,20},{6,20}
        };
        A.kpt_color_ids = {
            0,0,0,0,0, // face 
            2,1,2,1,2,1, // front legs 
            4,3,4,3,4,3, // back legs 
            5,9,5,6,5,5 // ceneter and tail 
        }; 
        A.kpt_conf_thresh = {
            0.8, // nose 0
            0.8, // eye left  1
            0.8, // eye right   2 
            0.8, // ear root left 3
            0.8, // ear root right 4
            0.8, // left shoulder 5
            0.8, // right shoulder 6
            0.8, // left elbow 7
            0.8, // right elbow 8
            0.9, // left paw 9
            0.9, // right paw 10
            0.8, // hip left 11
            0.8, // hip right  12
            0.8, // knee left  13
            0.8, // knee right  14
            0.85, // foot left 15
            0.85, // foot right 16
            0.5, // neck 17
            0.8, // tail root 18
            0.8, // withers    19
            0.8, // center 20
            0.5, // tail middle  21
            0.5  // tail end 22
        }; 
		//A.kpt_conf_thresh = {
		//	0.5, // nose 0
		//	0.5, // eye left  1
		//	0.5, // eye right   2 
		//	0.5, // ear root left 3
		//	0.5, // ear root right 4
		//	0.5, // left shoulder 5
		//	0.5, // right shoulder 6
		//	0.5, // left elbow 7
		//	0.5, // right elbow 8
		//	0.5, // left paw 9
		//	0.5, // right paw 10
		//	0.5, // hip left 11
		//	0.5, // hip right  12
		//	0.5, // knee left  13
		//	0.5, // knee right  14
		//	0.5, // foot left 15
		//	0.5, // foot right 16
		//	0.5, // neck 17
		//	0.5, // tail root 18
		//	0.5, // withers    19
		//	0.5, // center 20
		//	0.5, // tail middle  21
		//	0.5  // tail end 22
		//};
    }
    else 
    {
        std::cout << "skel type " << type << " not implemented yet" << std::endl;
        exit(-1); 
    }
    
    return A; 
}

//void BodyState::saveState(std::string filename)
//{
//	std::ofstream os(filename);
//	if (!os.is_open())
//	{
//		std::cout << "Could not open " << filename << std::endl;
//		exit(-1); 
//	}
//	// 1. trans, double * 3 
//	os << trans << std::endl; 
//	// 2. pose, double * 43 * 3
//	os << pose << std::endl; 
//	// 3. alpha, double * 1 
//	os << scale << std::endl; 
//	// 4. id, int * 1
//	os << frameid << std::endl;
//	os << id << std::endl; 
//	// 5. points, double * 3 *3 
//	os << points[0] << std::endl
//		<< points[1] << std::endl
//		<< points[2] << std::endl; 
//	os.close(); 
//}

//void BodyState::loadState(std::string filename)
//{
//	std::ifstream is(filename); 
//	if (!is.is_open())
//	{
//		std::cout << "could not open " << filename << std::endl; 
//		exit(-1); 
//	}
//	for (int i = 0; i < 3; i++)is >> trans(i);
//	pose.resize(43 * 3);
//	for (int i = 0; i < 43 * 3; i++) is >> pose(i);
//	is >> scale; 
//	is >> frameid;
//	double d_id; 
//	is >> d_id; id = int(d_id); 
//	points.resize(3); 
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			is >> points[i](j);
//		}
//	}
//	center = points[1]; 
//	is.close(); 
//}


vector<Eigen::Vector3f> convertMatToVec(const Eigen::MatrixXf& skel)
{
	vector<Eigen::Vector3f> vec;
	vec.resize(skel.cols()); 
	for (int i = 0; i < skel.cols(); i++)
	{
		vec[i] = skel.col(i); 
	}
	return vec; 
}

// *** visualization functions 
void drawSkelDebug(cv::Mat& img, const vector<Eigen::Vector3f>& _skel2d,
	SkelTopology m_topo
	)
{
	std::vector<Eigen::Vector3i> m_CM; 
	getColorMap("anliang_render", m_CM);
	for (int i = 0; i < _skel2d.size(); i++)
	{
		int colorid = m_topo.kpt_color_ids[i];
		Eigen::Vector3i color = m_CM[colorid];
		cv::Scalar cv_color(color(2), color(1), color(1));

		cv::Point2d p(_skel2d[i](0), _skel2d[i](1));
		double conf = _skel2d[i](2);
		if (conf < m_topo.kpt_conf_thresh[i]) continue;
		cv::circle(img, p, int(12 * conf), cv_color, -1);
	}
	for (int k = 0; k < m_topo.bone_num; k++)
	{
		int jid = m_topo.bones[k](0);
		int colorid = m_topo.kpt_color_ids[jid];
		Eigen::Vector3i color = m_CM[colorid];
		cv::Scalar cv_color(color(2), color(1), color(0));

		Eigen::Vector2i b = m_topo.bones[k];
		Eigen::Vector3f p1 = _skel2d[b(0)];
		Eigen::Vector3f p2 = _skel2d[b(1)];
		if (p1(2) < m_topo.kpt_conf_thresh[b(0)] || p2(2) < m_topo.kpt_conf_thresh[b(1)]) continue;
		cv::Point2d p1_cv(p1(0), p1(1));
		cv::Point2d p2_cv(p2(0), p2(1));
		cv::line(img, p1_cv, p2_cv, cv_color, 4);
	}
}

void drawSkelMonoColor(cv::Mat& img, const vector<Eigen::Vector3f>& _skel2d, int colorid, 
	SkelTopology m_topo)
{
	std::vector<Eigen::Vector3i> m_CM = getColorMapEigen("anliang_render"); 
	Eigen::Vector3i color = m_CM[colorid];
	cv::Scalar cv_color(color(2), color(1), color(0));
	for (int i = 0; i < _skel2d.size(); i++)
	{
		cv::Point2d p(_skel2d[i](0), _skel2d[i](1));
		double conf = _skel2d[i](2);
		if (conf < m_topo.kpt_conf_thresh[i]) continue;
		cv::circle(img, p, 8, cv_color, -1);
	}
	for (int k = 0; k < m_topo.bone_num; k++)
	{
		Eigen::Vector2i b = m_topo.bones[k];
		Eigen::Vector3f p1 = _skel2d[b(0)];
		Eigen::Vector3f p2 = _skel2d[b(1)];
		if (p1(2) < m_topo.kpt_conf_thresh[b(0)] || p2(2) < m_topo.kpt_conf_thresh[b(1)]) continue;
		cv::Point2d p1_cv(p1(0), p1(1));
		cv::Point2d p2_cv(p2(0), p2(1));
		cv::line(img, p1_cv, p2_cv, cv_color, 4);
	}
}

Eigen::VectorXf convertStdVecToEigenVec(const std::vector<Eigen::Vector3f>& joints)
{
	int jointnum = joints.size(); 
	Eigen::VectorXf data; 
	if (jointnum == 0) return data; 
	data.resize(jointnum * 3); 
	for (int i = 0; i < jointnum; i++) data.segment<3>(3 * i) = joints[i];
	return data; 
}

void printSkel(const std::vector<Eigen::Vector3f>& skel)
{
	for (int i = 0; i < skel.size(); i++)
	{
		std::cout << std::setw(2) << i << ": " << skel[i].transpose() << std::endl;
	}
}

float distSkel2DTo2D(const std::vector<Eigen::Vector3f>& skel1, const std::vector<Eigen::Vector3f>& skel2, const SkelTopology& topo, float& valid)
{
	if (skel1.size() < topo.joint_num || skel2.size() < topo.joint_num) return 1000000l;
	float dist = 0; 
	valid = 0;
	for (int i = 0; i < topo.joint_num; i++)
	{
		if (skel1[i](2) < topo.kpt_conf_thresh[i] || skel2[i](2) < topo.kpt_conf_thresh[i])continue; 
		dist += (skel1[i] - skel2[i]).norm(); 
		valid += 1; 
	}
	if (valid == 0) return 10000001;
	else return dist / valid;
}

float distBetween3DSkelAnd2DDet(const vector<Eigen::Vector3f>& skel3d,
	const MatchedInstance& det, const vector<Camera>& cams, const SkelTopology& topo
)
{
	float dist_all_views = 0;
	for (int view = 0; view < det.view_ids.size(); view++)
	{
		int camid = det.view_ids[view];
		Camera cam = cams[camid];
		std::vector<Eigen::Vector3f> points2d;
		project(cam, skel3d, points2d);
		int valid_num = 0;
		float total_dist = 0;
		for (int i = 0; i < points2d.size(); i++)
		{
			if (det.dets[view].keypoints[i](2) < topo.kpt_conf_thresh[i]) continue;
			Eigen::Vector3f diff = points2d[i] - det.dets[view].keypoints[i];
			total_dist += diff.norm();
			valid_num += 1;
		}
		dist_all_views += total_dist / valid_num;
	}
	dist_all_views /= det.view_ids.size();
	return dist_all_views;
}

float distBetweenSkel3D(const vector<Eigen::Vector3f>& S1, const vector<Eigen::Vector3f>& S2)
{
	if (S1.size() == 0 || S2.size() == 0) return 10000;
	int overlay = 0;
	float total_dist = 0;
	for (int i = 0; i < S1.size(); i++)
	{
		if (S1[i].norm() < 0.00001 || S2[i].norm() < 0.00001) continue;
		Eigen::Vector3f vec = S1[i] - S2[i];
		total_dist += vec.norm();
		overlay += 1;
	}
	if (overlay == 0) return 10000;
	// 2020 09 18: change 2.0 to 1.0
	return total_dist / pow(overlay, 1.0f);

}