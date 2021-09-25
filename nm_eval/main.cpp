#include "main.h" 

#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
#include <vector_functions.hpp>
#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "../articulation/pigmodel.h"
#include "../articulation/pigsolver.h"
#include "../posesolver/framesolver.h"
#include "../utils/mesh.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"
#include "../render/render_utils.h"

int main()
{
	//run_eval(); 
	//process_generate_label3d(); 
	run_visualize_gt(); 

	return 0; 
}

int run_eval()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");

	FrameSolver frame;
	frame.configByJson(conf_projectFolder + "/configs/config_20190704_foreval.json");

	frame.set_frame_id(750);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];

	// init renderer
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(true);
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

	for (int i = 0; i < 40; i++)
	{
		int frameid = 750 + 25 * i;
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		frame.load_clusters();
		frame.read_parametric_data();


	}
	return 0; 
}

int run_visualize_gt()
{
	Part1Data loader; 
	loader.init();
	std::string gt_folder = "E:/evaluation_dataset/part1/dataset_process/label_mix/"; 
	for (int i = 0; i < 25; i++)
	{
		int frameid = 750 + 25 * i; 
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		loader.set_frame_id(frameid);
		loader.read_imgs();
		loader.read_labeling();

		loader.m_gt_keypoints_3d= load_joint23(gt_folder, frameid); 

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