#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <vector_functions.hpp>

#include <json/json.h> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
#include <filesystem> 

#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "../utils/mesh.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"
#include "../utils/image_utils.h"
#include "framesolver.h"
#include "main.h"

int run_pose_smooth()
{
	show_gpu_param();
	std::string conf_projectFolder = "H:/MAMMAL_core/";
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
	if (!std::filesystem::is_directory(smth_folder))
	{
		std::filesystem::create_directory(smth_folder);
	}
	std::string jsmth_folder = frame.m_result_folder + "/joints_62_smth/";
	if (!std::filesystem::is_directory(jsmth_folder))
	{
		std::filesystem::create_directory(jsmth_folder); 
	}
	std::string j23smth_folder = frame.m_result_folder + "/joints_23_smth/";
	if (!std::filesystem::is_directory(j23smth_folder))
	{
		std::filesystem::create_directory(j23smth_folder); 
	}

	// smoothing the whole sequence
	std::vector<std::vector<std::vector<Eigen::Vector3f> > > all_joints62; 
	all_joints62.resize(4); 
	
	for (int pid = 0; pid < 4; pid++)
	{
		all_joints62[pid].resize(frame.get_frame_num()); 
		for (int frameid = 0; frameid < frame.get_frame_num(); frameid++)
		{
			std::stringstream ss;
			ss << frame.m_result_folder << "/joints_62/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
			all_joints62[pid][frameid] = read_points(ss.str());
		}
		all_joints62[pid] = hanning_smooth(all_joints62[pid]); 
	}
	// re-fit smoothed joints; write states and renderings. 
	std::cout << "run joint smoothing ... " << std::endl; 
	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "===========write smoothed data frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);
		frame.read_parametric_data();
		auto solvers = frame.mp_bodysolverdevice;

		for (int pid = 0; pid < frame.m_pignum; pid++)
		{
			std::stringstream ss; 
			ss << frame.m_result_folder << "/joints_62_smth/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
			write_points(ss.str(), all_joints62[pid][frameid]); 
			solvers[pid]->fitPoseToJointSameTopo(all_joints62[pid][frameid]);

			std::stringstream ss_state; 
			ss_state << frame.m_result_folder << "/state_smth/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
			solvers[pid]->saveState(ss_state.str()); 

			std::stringstream ss_23; 
			ss_23 << frame.m_result_folder << "/joints_23_smth/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt"; 
			write_points(ss_23.str(), solvers[pid]->getRegressedSkel_host()); 
		}
	}

	return 0;
}


// this is for hind_leg angle profiling.
int run_pose_bone_length()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_render");

	FrameSolver frame;
	std::string configfile = get_config();
	frame.configByJson(conf_projectFolder + "configs/config_20190703_motion4.json");
	int startid = 79140;
	int framenum = 60;
	int targetid = 0;

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
		if (!std::filesystem::is_directory(frame.m_result_folder + subfolders[i]))
			std::filesystem::create_directory(frame.m_result_folder + subfolders[i]);
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
		cv::Mat reproj = frame.visualizeProj(targetid);
		std::stringstream ss_proj;
		ss_proj << frame.m_result_folder << "/proj2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_proj.str(), reproj);
		save_points(outputfolder, targetid, frameid, skels); 
	}

	return 0;
}
