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
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"

#include "main.h"

/* 
2021.10.11: 
This function is used to generate demo result used in paper fig.1
config_demo1.json: 
*/
void run_demo_result()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	std::string config_file = "/configs/config_demo11.json"; 
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
	Renderer::s_Init(false);
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
		"assoc", "render", "clusters", "state", "reassoc2", "proj2", "before_swap2",
		"fitting", "annotation", "skels", "anchor_state", "rawdet", "joints_62", "joints_23",
		"debug", "render_all_last", "render_all"
	};
	for (int i = 0; i < subfolders.size(); i++)
	{
		if (!boost::filesystem::is_directory(test_result_folder + subfolders[i]))
			boost::filesystem::create_directory(test_result_folder + subfolders[i]);
	}

	frame.init_parametric_solver();
	int framenum = frame.get_frame_num();
	int increment = 1;
	if (framenum < 0) increment = -1;

	frame.saveConfig();

	for (int frameid = start; frameid != start + framenum; frameid += increment)
	{
		m_renderer.SetBackgroundColor(Eigen::Vector4f(0, 0, 0, 0));

		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		TimerUtil::Timer<std::chrono::microseconds> tt;
		tt.Start();

		frame.set_frame_id(frameid);
		frame.fetchData();

		if (frame.m_use_init_cluster)
			frame.load_clusters();
		else
			frame.matching_by_tracking();

		frame.save_clusters();
		frame.resetSolverStateMarker();

		if (frame.m_use_triangulation_only)
		{
			frame.DARKOV_Step1_setsource();
			frame.DirectTriangulation();
			frame.save_skels();
		}
		std::cout << "w/o rendering " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;
		//std::vector<int> ids = { 0,3,7,8 };
		std::vector<int> ids = { 0,1,2,3,4,5,6,7,8,9 };
		auto rawimgs = frame.m_imgsUndist;
		for (int k = 0; k < ids.size(); k++)
		{
			int id = ids[k];
			std::stringstream ss;
			ss << test_result_folder << "/render_all/raw_" << id << ".png";
			cv::imwrite(ss.str(), rawimgs[id]);

			std::stringstream ss_rawdetpre1;
			ss_rawdetpre1 << test_result_folder << "/render_all/rawdetpure3_" << id << ".png";
			cv::Mat rawdetpure1 = frame.tmp_visualizeRawDetPure(id, false);
			cv::imwrite(ss_rawdetpre1.str(), rawdetpure1);


			std::stringstream ss_rawdetpre;
			ss_rawdetpre << test_result_folder << "/render_all/rawdetpure4_" << id << ".png";
			cv::Mat rawdetpure = frame.tmp_visualizeRawDetPure(id);
			cv::imwrite(ss_rawdetpre.str(), rawdetpure);

			/*std::stringstream ss_rawdet;
			ss_rawdet << "D:/results/paper_teaser/0704_demo/render_all/rawdet_" << id << ".png";
			cv::Mat rawdet = frame.tmp_visualizeRawDet(id);
			cv::imwrite(ss_rawdet.str(), rawdet);*/
			cv::Mat rawdet = blend_images(rawimgs[id], rawdetpure, 0.5); 
			std::stringstream ss_rawdet;
			ss_rawdet << "D:/results/paper_teaser/0704_demo/render_all/rawdet3_" << id << ".png";
			cv::imwrite(ss_rawdet.str(), rawdet);

			//cv::Mat assoc = frame.visualizeIdentity2D(id);
			//std::stringstream ss1;
			//ss1 << test_result_folder << "/assoc/assoc_" << k << "_" << std::setw(6) << std::setfill('0') << frameid << ".png";
			//cv::imwrite(ss1.str(), assoc);

			for (int pid = 0; pid < 4; pid++)
			{
				cv::Mat assoc2 = frame.visualizeIdentity2D(id, pid);
				std::stringstream ss2;
				ss2 << test_result_folder << "/render_all/assoc_" << k << "_" << pid << "_" << std::setw(6) << std::setfill('0') << frameid << ".png";
				cv::imwrite(ss2.str(), assoc2);

				cv::Mat assoc3 = frame.tmp_visualizeIdentity2D(id, pid);
				std::stringstream ss3;
				ss3 << test_result_folder << "/render_all/group_" << k << "_" << pid << ".png";
				cv::imwrite(ss3.str(), assoc3);
			}
		}
		cv::Mat assoc3 = frame.visualizeIdentity2D();
		std::stringstream ss3;
		ss3 << test_result_folder << "/assoc/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss3.str(), assoc3);

		cv::Mat rawfit = frame.visualizeRawAssoc();
		cv::Mat rawfit_small = my_resize(rawfit, 1);
		std::stringstream ss_rawassoc;
		ss_rawassoc << test_result_folder << "/fitting/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_rawassoc.str(), rawfit_small);

		if (frame.m_use_triangulation_only) {
			cv::Mat reproj = frame.visualizeProj();
			cv::Mat reproj_small = my_resize(reproj, 0.25);
			std::stringstream ss_proj;
			ss_proj << test_result_folder << "/proj2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::imwrite(ss_proj.str(), reproj_small);
		}

		frame.DARKOV_Step1_setsource();

#if 1// search anchor 
		if (frame.m_use_init_anchor)
		{
			for (int i = 0; i < 4; i++)
			{
				frame.DARKOV_Step2_searchanchor(i);
				frame.saveAnchors(frame.m_result_folder + "/anchor_state/");
				frame.DARKOV_Step2_optimanchor(i);
			}
		}
#else 
		frame.DARKOV_Step2_loadanchor();
		for (int i = 0; i < 4; i++)
			frame.DARKOV_Step2_optimanchor(i);
#endif 

#if 1 // comment to visualize anchor result
		frame.DARKOV_Step4_fitrawsource();
		frame.DARKOV_Step3_reassoc_type2();
		frame.DARKOV_Step4_fitreassoc();

		frame.DARKOV_Step5_postprocess();
		frame.save_parametric_data();

		std::cout << "w/o rendering " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;

		//cv::Mat reassoc = frame.visualizeReassociation();
		//std::stringstream ss_reassoc;
		//ss_reassoc << test_result_folder << "/reassoc2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		//cv::imwrite(ss_reassoc.str(), reassoc);

		cv::Mat beforeimg = frame.visualizeSwap();
		std::stringstream ss_before;
		ss_before << test_result_folder << "/before_swap2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_before.str(), beforeimg);

		// save joints 
		//for (int pid = 0; pid < pignum; pid++)
		//{
		//	auto joints_62 = solvers[pid]->GetJoints();
		//	save_points(frame.m_result_folder + "/joints_62/", pid, frameid, joints_62);
		//	auto joints_23 = solvers[pid]->getRegressedSkel_host();
		//	save_points(frame.m_result_folder + "/joints_23/", pid, frameid, joints_23);
		//}
#endif 
		//frame.read_parametric_data(); 

		std::string prefix = "final";
		m_renderer.clearAllObjs();
		std::vector<int> render_pig_ids = { 0,1,2,3 };
		auto solvers = frame.mp_bodysolverdevice;
		for (int pid = 0; pid < 4; pid++)
		{
			std::stringstream ss_obj;
			ss_obj << test_result_folder << "/debug/anchor_" << pid << ".txt";
			solvers[pid]->saveState(ss_obj.str());
		}
		for (int k = 0; k < render_pig_ids.size(); k++)
		{
			int pid = render_pig_ids[k];
			RenderObjectColor* p_model = new RenderObjectColor();
			solvers[pid]->UpdateNormalFinal();
			int colorid = frame.m_pig_names[pid];
			p_model->SetVertices(solvers[pid]->GetVertices());
			p_model->SetNormal(solvers[pid]->GetNormals());
			p_model->SetFaces(solvers[pid]->GetFacesVert());
			p_model->SetColor(m_CM[colorid]);
			m_renderer.colorObjs.push_back(p_model);

			std::vector<Eigen::Vector3f> joints = solvers[pid]->GetJoints();
		}

		std::vector<cv::Mat> rawImgs = frame.get_imgs_undist();
		cv::Mat a(cv::Size(1920, 1080), CV_8UC3);
		cv::Mat pack_raw;
		rawImgs.push_back(a);
		packImgBlock(rawImgs, pack_raw);

		std::vector<cv::Mat> all_renders(cams.size());
		for (int camid = 0; camid < cams.size(); camid++)
		{
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			m_renderer.Draw();
			cv::Mat img = m_renderer.GetImage();
			//std::stringstream ss_perimg;
			//ss_perimg << test_result_folder << "/render_all/" << prefix << "_" << camid << ".png";
			//cv::imwrite(ss_perimg.str(), img);
			all_renders[camid] = img;
		}

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));
		for (int k = 0; k < render_pig_ids.size(); k++)
		{
			int pid = render_pig_ids[k];
			m_renderer.colorObjs[k]->isMultiLight = true;
		}
		m_renderer.createPlane(conf_projectFolder); 

		for (int camid = 0; camid < cams.size(); camid++)
		{
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			cv::Mat img = m_renderer.GetImageOffscreen(); 
			std::stringstream ss_perimg;
			ss_perimg << test_result_folder << "/render_all_last/" << prefix << "_" << camid << "_" << frameid << ".png";
			cv::imwrite(ss_perimg.str(), img);
		}

		Eigen::Vector3f pos2(0, -0, 3);
		Eigen::Vector3f up2(0, 1, -0);
		Eigen::Vector3f center2(0, -0, 0);
		m_renderer.s_camViewer.SetExtrinsic(pos2, up2, center2);

		cv::Mat img = m_renderer.GetImageOffscreen(); 
		std::stringstream ss_rend; 
		ss_rend << test_result_folder << "/render_all_last/" << frameid << ".png";
		cv::imwrite(ss_rend.str(), img); 
		all_renders.push_back(img);

		cv::Mat packed_render;
		packImgBlock(all_renders, packed_render);

		cv::Mat blend;
		overlay_render_on_raw_gpu(packed_render, pack_raw, blend);

		std::stringstream all_render_file;
		all_render_file << test_result_folder << "/render_all_last/" << prefix << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
		cv::imwrite(all_render_file.str(), blend);

		std::cout << "total:       " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;

		Eigen::Vector3f pos4(-0.988364, -2.79656, 1.91186);
		Eigen::Vector3f up4(0.226098, 0.500471, 0.835709);
		Eigen::Vector3f center4(0.116397, -0.369781, 0.14214);

		m_renderer.s_camViewer.SetExtrinsic(pos4, up4, center4); 
		cv::Mat imgforpaper = m_renderer.GetImageOffscreen(); 
		cv::imwrite(test_result_folder + "/render_all_last/" + prefix + std::to_string(frameid) + ".png", imgforpaper);

		//if (frameid == start) {
		//	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		//	while (!glfwWindowShouldClose(windowPtr))
		//	{
		//		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		//		m_renderer.Draw();

		//		glfwSwapBuffers(windowPtr);
		//		glfwPollEvents();
		//	};
		//}
	}

}
