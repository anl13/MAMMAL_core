#include "main.h" 

#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp> 

#include <vector_functions.hpp>
#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "../posesolver/framesolver.h"
#include "../utils/mesh.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"
#include "../render/render_utils.h"

int run_eval_sil()
{
	show_gpu_param();
	std::string conf_projectFolder = get_parent_folder();
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	std::string config_file = "configs/config_BamaPig3D_main.json";
	FrameSolver frame;
	frame.configByJson(conf_projectFolder + config_file);

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(frame.m_startid);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];
	int pignum = frame.m_pignum;

	// init renderer
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(true);
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

	frame.mp_renderEngine = &m_renderer;

	frame.is_smth = false;
	frame.init_parametric_solver();

	for (int i = 0; i < 70; i++)
	{
		int frameid = 25 * i;
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchGtData();
		frame.fetchData();
		frame.load_clusters(); 
		frame.read_parametric_data(); 
		frame.compute_silhouette_loss(); 
	}

	return 0;
}


int run_eval_reassoc()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	std::string config_file = "configs/config_20190704_foreval.json";
	FrameSolver frame;
	frame.configByJson(conf_projectFolder + config_file);

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(frame.m_startid);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];
	int pignum = frame.m_pignum;

	// init renderer
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(true);
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

	frame.mp_renderEngine = &m_renderer;

	frame.is_smth = false;
	frame.init_parametric_solver();

	for (int i = 0; i < 70; i++)
	{
		int frameid = 750 + 25 * i;
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		if (i > 0)
		{
			frame.set_frame_id(frameid - 1);

			frame.read_parametric_data();
			frame.DARKOV_Step5_postprocess(); 
		}
		frame.set_frame_id(frameid); 
		frame.fetchGtData();
		frame.fetchData();
		frame.load_clusters();
		frame.resetSolverStateMarker();
		frame.DARKOV_Step1_setsource();
		frame.DARKOV_Step2_loadanchor(); 
		if (i == 0)
		{
			for (int id = 0; id < 4; id++)
				frame.DARKOV_Step2_optimanchor(id); 
		}
		frame.DARKOV_Step4_fitrawsource(frame.m_solve_sil_iters); 
		frame.DARKOV_Step3_reassoc_type2(); 


		frame.compute_2dskel_loss_proj();

		continue; 
	}

	return 0;
}
