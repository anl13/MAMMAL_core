#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

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

int run_inspect()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_render");

	FrameSolver frame;
	frame.configByJson(conf_projectFolder + "/posesolver/config_7.json");
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(0);
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
	int start = frame.get_start_id(); 

	std::string test_result_folder = frame.m_result_folder;
	if (!boost::filesystem::is_directory(test_result_folder))
	{
		boost::filesystem::create_directory(test_result_folder); 
	}
	std::vector<std::string> subfolders = {
		"assoc", "render_all", "clusters", "state", "reassoc2", "proj2", "before_swap2",
		"fitting"
	};
	for (int i = 0; i < subfolders.size(); i++)
	{
		if (!boost::filesystem::is_directory(test_result_folder + subfolders[i]))
			boost::filesystem::create_directory(test_result_folder + subfolders[i]);
	}

	frame.init_parametric_solver(); 


	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		m_renderer.SetBackgroundColor(Eigen::Vector4f(0, 0, 0, 0));

		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		TimerUtil::Timer<std::chrono::microseconds> tt;
		tt.Start();

		frame.set_frame_id(frameid);
		frame.fetchData();
		
		if (frameid == start)
			frame.matching_by_tracking();
			//frame.load_clusters();
		else
			frame.pureTracking(); 

		frame.save_clusters(); 

		frame.resetSolverStateMarker(); 

		if (false)
		{
			frame.read_parametric_data(); 
			frame.DARKOV_Step5_postprocess();
		}
		// pipeline 3 
		else
		{
			frame.try_load_anno(); 

			frame.DARKOV_Step1_setsource();
			//frame.DARKOV_Step2_loadanchor();
			//frame.DARKOV_Step2_searchanchor(2); 
			//frame.saveAnchors(frame.m_result_folder +	"/anchor_state/"); 
			//frame.DARKOV_Step2_optimanchor(2);

			frame.DARKOV_Step4_fitrawsource();
			frame.DARKOV_Step3_reassoc_type2();
			frame.DARKOV_Step4_fitreassoc();

			//cv::Mat sift = frame.visualizeSIFT();
			//std::stringstream ss_sift;
			//ss_sift << test_result_folder << "/sift/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			//cv::imwrite(ss_sift.str(), sift);

			frame.DARKOV_Step5_postprocess();
			frame.save_parametric_data();

			std::cout << "w/o rendering " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;
			cv::Mat assoc = frame.visualizeIdentity2D();
			std::stringstream ss;
			ss << test_result_folder << "/assoc/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::imwrite(ss.str(), assoc);
#if 1
			cv::Mat reassoc = frame.visualizeReassociation();
			std::stringstream ss_reassoc;
			ss_reassoc << test_result_folder << "/reassoc2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::imwrite(ss_reassoc.str(), reassoc);

			cv::Mat reproj = frame.visualizeVisibility();
			std::stringstream ss_proj;
			ss_proj << test_result_folder << "/proj2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::imwrite(ss_proj.str(), reproj);

			cv::Mat beforeimg = frame.visualizeSwap();
			std::stringstream ss_before;
			ss_before << test_result_folder << "/before_swap2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::imwrite(ss_before.str(), beforeimg);

			cv::Mat rawfit = frame.visualizeRawAssoc();
			std::stringstream ss_rawassoc; 
			ss_rawassoc << test_result_folder << "/fitting/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::imwrite(ss_rawassoc.str(), rawfit); 

			//frame.pipeline2_searchanchor();
			//frame.saveAnchors(test_result_folder + "/anchor_state_252");
#endif 
		}

#if 1 // render all view 
		m_renderer.clearAllObjs();

		auto solvers = frame.mp_bodysolverdevice;

		for (int pid = 0; pid < pignum; pid++)
		{
			RenderObjectColor* p_model = new RenderObjectColor();
			solvers[pid]->UpdateNormalFinal();

			p_model->SetVertices(solvers[pid]->GetVertices());
			p_model->SetNormal(solvers[pid]->GetNormals());
			p_model->SetFaces(solvers[pid]->GetFacesVert());
			p_model->SetColor(m_CM[pid]);
			m_renderer.colorObjs.push_back(p_model);

			std::vector<Eigen::Vector3f> joints = solvers[pid]->GetJoints();
		}

		std::vector<cv::Mat> rawImgs = frame.get_imgs_undist();
		cv::Mat a(cv::Size(1920, 1080), CV_8UC3);
		cv::Mat pack_raw;
		rawImgs.push_back(a);
		rawImgs.push_back(a);
		packImgBlock(rawImgs, pack_raw);

		std::vector<cv::Mat> all_renders(cams.size());
		for (int camid = 0; camid < cams.size(); camid++)
		{
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			m_renderer.Draw();
			cv::Mat img = m_renderer.GetImage();

			all_renders[camid] = img;
		}

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1)); 
		for (int pid = 0; pid < pignum; pid++)
			m_renderer.colorObjs[pid]->isMultiLight = true; 
		m_renderer.createSceneDetailed(conf_projectFolder);
		Eigen::Vector3f pos1(0.904806, -1.57754, 0.58256);
		Eigen::Vector3f up1(-0.157887, 0.333177, 0.929551);
		Eigen::Vector3f center1(0.0915295, -0.128604, -0.0713566);
		//Eigen::Vector3f pos1(2.05239, 0.0712245, 0.013074);
		//Eigen::Vector3f up1(-0.0138006, 0.160204, 0.986988);
		//Eigen::Vector3f center1(0.0589942, -0.0909324, 0.00569892);

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
		all_render_file << test_result_folder << "/render_all/" << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
		cv::imwrite(all_render_file.str(), blend);
		
		std::cout << "total:       " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;
		//if (frameid == start ) {
		//	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		//	while (!glfwWindowShouldClose(windowPtr))
		//	{
		//		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		//		m_renderer.Draw();

		//		glfwSwapBuffers(windowPtr);
		//		glfwPollEvents();
		//	};
		//}
#else
		m_renderer.clearAllObjs();
		auto solvers = frame.mp_bodysolverdevice;

		m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

		for (int pid = 0; pid < pignum; pid++)
		{
			RenderObjectColor* p_model = new RenderObjectColor();
			solvers[pid]->UpdateNormalFinal();

			p_model->SetVertices(solvers[pid]->GetVertices());
			p_model->SetNormal(solvers[pid]->GetNormals());
			p_model->SetFaces(solvers[pid]->GetFacesVert());
			p_model->SetColor(m_CM[pid]);
			m_renderer.colorObjs.push_back(p_model);
		}

		std::vector<int> render_views = { 0,7 };

		std::vector<cv::Mat> rawImgs = frame.get_imgs_undist();
		std::vector<cv::Mat> rawImgsSelect;
		for (int k = 0; k < render_views.size(); k++) rawImgsSelect.push_back(rawImgs[render_views[k]]);

		std::vector<cv::Mat> all_renders(render_views.size());
		for (int k = 0; k < render_views.size(); k++)
		{
			int camid = render_views[k];
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			//m_renderer.Draw();
			//cv::Mat img = m_renderer.GetImage();
			cv::Mat img = m_renderer.GetImageOffscreen();
			all_renders[k] = img;
		}
		m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
		m_renderer.createSceneDetailed(conf_projectFolder);
		Eigen::Vector3f up1; up1 << 0.260221, 0.36002, 0.895919;
		Eigen::Vector3f pos1; pos1 << -1.91923, -2.12171, 1.37056;
		Eigen::Vector3f center1 = Eigen::Vector3f::Zero();
		m_renderer.s_camViewer.SetExtrinsic(pos1, up1, center1);
		cv::Mat img = m_renderer.GetImageOffscreen();
		all_renders.push_back(img);
		rawImgsSelect.push_back(img);

		Eigen::Vector3f pos2(0.0988611, -0.0113558, 3.00438);
		Eigen::Vector3f up2(0.00346774, 0.999541, -0.0301062);
		Eigen::Vector3f center2(0.0589942, -0.0909324, 0.00569892);
		m_renderer.s_camViewer.SetExtrinsic(pos2, up2, center2);
		img = m_renderer.GetImageOffscreen();
		all_renders.push_back(img);
		rawImgsSelect.push_back(img);

		cv::Mat pack_raw;
		packImgBlock(rawImgsSelect, pack_raw);

		cv::Mat packed_render;
		packImgBlock(all_renders, packed_render);

		cv::Mat blend;
		overlay_render_on_raw_gpu(packed_render, pack_raw, blend);

		std::stringstream all_render_file;
		all_render_file << frame.m_result_folder << "/render_all/" << std::setw(6) << std::setfill('0')
			<< frameid << "_overlay.png";
		cv::imwrite(all_render_file.str(), blend);
		std::cout << "total:       " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;
#endif 
	}

 	return 0;
}
