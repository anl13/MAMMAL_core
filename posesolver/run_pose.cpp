#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <vector_functions.hpp>

#include <json/json.h> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "../articulation/pigmodel.h"
#include "../articulation/pigsolver.h"
#include "../utils/mesh.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"

#include "framesolver.h"
#include "main.h"

void save_joints(std::string folder, int pid, int fid, const std::vector<Eigen::Vector3f>& data)
{
	std::stringstream ss;
	ss << folder << "/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << fid << ".txt"; 
	std::ofstream outputfile(ss.str()); 
	for (int i = 0; i < data.size(); i++)
	{
		outputfile << data[i].transpose() << std::endl; 
	}
	outputfile.close(); 
}

int run_pose_smooth()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_render");

	FrameSolver frame;
	std::string configfile = get_config(); 
	frame.configByJson(conf_projectFolder + configfile);
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(0);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];

	// init renderer
	frame.is_smth = false;
	int start = frame.get_start_id();
	frame.init_parametric_solver(); 

	std::string joint62_folder = frame.m_result_folder+ "/joints_62/";
	std::string smth_folder = frame.m_result_folder + "/state_smth/"; 
	if (!boost::filesystem::is_directory(smth_folder))
		boost::filesystem::create_directory(smth_folder);

	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);

		frame.read_parametric_data();
		
		auto solvers = frame.mp_bodysolverdevice;

		for (int pid = 0; pid < frame.m_pignum; pid++)
		{
			std::stringstream ss; 
			ss << frame.m_result_folder << "/joints_smth/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
			std::vector<Eigen::Vector3f> points62 = read_points(ss.str()); 
			solvers[pid]->fitPoseToJointSameTopo(points62);

			std::stringstream ss_state; 
			ss_state << frame.m_result_folder << "/state_smth/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
			solvers[pid]->saveState(ss_state.str()); 
		}
	}

	return 0;
}



int run_pose_bone_length()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_render");

	FrameSolver frame;
	std::string configfile = get_config();
	frame.configByJson(conf_projectFolder + "configs/config_20191003_socialdemo.json");
	int startid = 9820;
	int framenum = 200;
	int targetid = 1;

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(startid);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];

	// init renderer
	frame.is_smth = false;
	frame.init_parametric_solver();

	std::vector<std::string> subfolders = {
	"triskel", "regskel"
	};
	for (int i = 0; i < subfolders.size(); i++)
	{
		if (!boost::filesystem::is_directory(frame.m_result_folder + subfolders[i]))
			boost::filesystem::create_directory(frame.m_result_folder + subfolders[i]);
	}


	std::string outputfolder = frame.m_result_folder + "/triskel/";
	for (int frameid = startid; frameid < startid + framenum; frameid++)
	{
		std::cout << "===========processing frame " << frameid << "===============" << std::endl; 
		frame.set_frame_id(frameid);
		frame.fetchData(); 
		frame.load_clusters(); 
		frame.DARKOV_Step1_setsource();
		//std::vector<Eigen::Vector3f> skels = frame.mp_bodysolverdevice[targetid]->directTriangulationHost(3); 
		frame.DirectTriangulation(); 
		std::vector<Eigen::Vector3f> skels = frame.m_skels3d[targetid];
		cv::Mat reproj = frame.visualizeProj();
		cv::Mat reproj_small = my_resize(reproj, 0.25);
		std::stringstream ss_proj;
		ss_proj << frame.m_result_folder << "/proj2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_proj.str(), reproj_small);
		save_joints(outputfolder, targetid, frameid, skels); 
	}

	return 0;
}
