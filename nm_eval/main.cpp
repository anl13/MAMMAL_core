#include "main.h" 

#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
#include <filesystem>

#include <vector_functions.hpp>
#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "../posesolver/framesolver.h"
#include "../utils/mesh.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"
#include "../render/render_utils.h"

int main()
{
	//run_eval(); 
	//process_generate_label3d(); 
	//run_visualize_gt(); 
	//run_fitgt(); 
	run_eval_sil(); 
	//run_eval_reassoc(); 

	return 0; 
}

int run_visualize_gt()
{
	Part1Data loader; 
	loader.init();
	std::string gt_folder = "H:/examples/BamaPig3D/label_mix/"; 
	if (!std::filesystem::is_directory(gt_folder))
	{
		std::cout << "Please change 'gt_folder' to 'your_BamaPig3D_path/label_mix/' " << std::endl; 
		return 1; 
	}
	for (int i = 0; i < 25; i++)
	{
		int frameid = 25 * i; 
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		loader.set_frame_id(frameid);
		loader.read_imgs();
		loader.read_labeling();

		loader.m_gt_keypoints_3d = load_joint23(gt_folder, frameid); 

		//loader.compute_3d_gt();
		loader.reproject_skels(); 
		cv::Mat proj = loader.visualizeProj(); 
		cv::Mat detect = loader.visualize2D(); 
		cv::namedWindow("proj", cv::WINDOW_NORMAL);
		cv::namedWindow("detect", cv::WINDOW_NORMAL);
		cv::imshow("proj", proj);
		cv::imshow("detect", detect); 
		int key = cv::waitKey();
		if (key == 27)
			break; 
		cv::destroyAllWindows(); 

	}
	return 0; 
}