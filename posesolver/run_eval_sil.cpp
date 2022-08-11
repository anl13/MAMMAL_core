#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <filesystem> 
#include <vector_functions.hpp>

#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
#include <json/json.h> 

#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "../utils/mesh.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"
#include "../utils/definitions.h"
#include "../render/render_utils.h"

#include "framesolver.h"
#include "main.h" 

/* 
This function is used to perform silhouette evaluation on BamaPig3D evaluation. 
The results are written to "eval" directory under result folder. 
*/
int run_eval_sil()
{
	show_gpu_param();
	std::string conf_projectFolder = get_parent_folder();
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	//std::string config_file = "configs/config_BamaPig3D_nosil.json";
	std::string config_file = "configs/config_BamaPig3D_main.json"; 

	FrameSolver frame;
	frame.m_project_folder = conf_projectFolder; 
	frame.configByJson(conf_projectFolder + config_file);

	std::vector<std::string> subfolders = { "eval", "sil_vis" }; 
	for (int k = 0; k < subfolders.size(); k++)
	{
		std::string folder = frame.m_result_folder + "/" + subfolders[k];
		if (!std::filesystem::exists(folder))
			std::filesystem::create_directories(folder);
	}
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