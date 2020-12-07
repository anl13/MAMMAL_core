#include "main.h"
#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>

#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "../articulation/pigmodel.h"
#include "../articulation/pigsolver.h"
#include "framesolver.h"
#include "../utils/mesh.h"
#include <vector_functions.hpp>
#include "main.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"

void save_joints(int pid, int fid, const std::vector<Eigen::Vector3f>& data)
{
	std::stringstream ss;
	ss << "F:/pig_results_anchor_sil/joints/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << fid << ".txt"; 
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
	frame.configByJson(conf_projectFolder + "/posesolver/confignew.json");
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(0);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];

	// init renderer
	frame.result_folder = "F:/pig_results_anchor_sil/";
	frame.is_smth = false;
	int start = frame.get_start_id();
	frame.init_parametric_solver(); 

	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);

		frame.read_parametric_data();
		
		auto solvers = frame.mp_bodysolverdevice;

		//for (int pid = 0; pid < 4; pid++)
		//{
		//	save_joints(pid, frameid, solvers[pid]->GetJoints());
		//}

		//continue; 

		for (int pid = 0; pid < 4; pid++)
		{
			std::stringstream ss; 
			ss << "F:/pig_results_anchor_sil/joints_smth/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
			std::vector<Eigen::Vector3f> points62 = read_points(ss.str()); 
			solvers[pid]->fitPoseToJointSameTopo(points62);

			std::stringstream ss_state; 
			ss_state << "F:/pig_results_anchor_sil/state_smth/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
			solvers[pid]->saveState(ss_state.str()); 
		}
	}

	return 0;
}
