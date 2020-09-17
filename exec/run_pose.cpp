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
#include "../associate/framedata.h"
#include "../utils/mesh.h"
#include "../nanorender/NanoRenderer.h"
#include <vector_functions.hpp>
#include "main.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"

int run_pose()
{

	show_gpu_param(); 
	//std::string pig_config = "D:/Projects/animal_calib/articulation/pigmodel_config.json";
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_render"); 

	FrameData frame;
	frame.configByJson(conf_projectFolder + "/associate/config.json");
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();
	//PigSolver shapesolver(pig_config);

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(0);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];

	// init renderer
	Eigen::Matrix3f K = cam.K; 
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(false); 
	Renderer m_renderer(conf_projectFolder + "/render/shader/"); 
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f)); 

	frame.mp_renderEngine = &m_renderer; 

	frame.result_folder = "G:/pig_results/";
	frame.is_smth = false; 
	int start = frame.get_start_id();

	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();

		if (frameid == start) frame.load_clusters();
		else frame.matching_by_tracking(); 
		TimerUtil::Timer<std::chrono::milliseconds> tt;
		tt.Start();
		if (frameid == start) frame.read_parametric_data();
		else frame.solve_parametric_model(); 
		std::cout << "solve model: " << tt.Elapsed() << " ms" << std::endl; 

		frame.save_clusters();
		frame.save_parametric_data();

		//cv::Mat proj_skel = frame.visualizeProj(); 
		//cv::imwrite("G:/pig_results/fitting/proj.png", proj_skel); 
		cv::Mat assoc = frame.visualizeIdentity2D(); 
		std::stringstream ss; 
		ss << "G:/pig_results/assoc/" << std::setw(6) << std::setfill('0') << frameid << ".png"; 
		cv::imwrite(ss.str(), assoc); 
		continue; 
		m_renderer.clearAllObjs();
		auto solvers = frame.mp_bodysolverdevice;
		
		for (int pid = 0; pid < 2; pid++)
		{
			solvers[pid]->debug_source_visualize(frameid);

			RenderObjectColor* p_model = new RenderObjectColor();
			solvers[pid]->UpdateNormalFinal();

			p_model->SetVertices(solvers[pid]->GetVertices());
			p_model->SetNormal(solvers[pid]->GetNormals());
			p_model->SetFaces(solvers[pid]->GetFacesVert());
			p_model->SetColor(m_CM[pid]);
			m_renderer.colorObjs.push_back(p_model);
		}

		std::vector<cv::Mat> rawImgs = frame.get_imgs_undist();
		cv::Mat pack_raw;
		packImgBlock(rawImgs, pack_raw);

		std::vector<cv::Mat> all_renders(cams.size());
		for (int camid = 0; camid < cams.size(); camid++)
		{
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			m_renderer.Draw(); 
			cv::Mat img = m_renderer.GetImage();
			
			all_renders[camid] = img;
		}
		m_renderer.createScene(conf_projectFolder);
		Eigen::Vector3f pos1(1.84296, -2.18987, 1.19391);
		Eigen::Vector3f up1(-0.265077, 0.293909, 0.918342);
		Eigen::Vector3f center1(0.0589942, -0.0909324, 0.00569892);
		m_renderer.s_camViewer.SetExtrinsic(pos1, up1, center1);
		m_renderer.Draw();
		cv::Mat img = m_renderer.GetImage();
		all_renders.push_back(img);

		Eigen::Vector3f pos2(0.0988611, -0.0113558, 3.00438);
		Eigen::Vector3f up2(0.00346774, 0.999541, -0.0301062);
		Eigen::Vector3f center2(0.0589942, -0.0909324, 0.00569892);
		m_renderer.s_camViewer.SetExtrinsic(pos2, up2, center2);
		m_renderer.Draw();
		img = m_renderer.GetImage();
		all_renders.push_back(img);

		cv::Mat packed_render; 
		packImgBlock(all_renders, packed_render);
		
		cv::Mat blend;
		overlay_render_on_raw_gpu(packed_render, pack_raw, blend); 

		std::stringstream all_render_file; 
		all_render_file << "G:/pig_results/render_all/overlay/" << std::setw(6) << std::setfill('0')
			<< frameid << "_overlay2.png";
		std::stringstream file2;
		file2 << "G:/pig_results/render_all/render/" << std::setw(6) << std::setfill('0')
			<< frameid << ".png";

		cv::imwrite(file2.str(), packed_render);

		cv::imwrite(all_render_file.str(), blend); 
	}

	
	return 0;
}
