#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <filesystem>

#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "framesolver.h"
#include "../utils/mesh.h"
#include <vector_functions.hpp>
#include "main.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/image_utils.h" 
#include "../utils/show_gpu_param.h"

std::string get_config(std::string name)
{
	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(name);
	if (!instream.is_open())
	{
		std::cout << "can not open " << name  << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}
	std::string config_file = root["config_file"].asString(); 
	instream.close();
	return config_file; 
}

int run_MAMMAL_main()
{
	show_gpu_param();
	std::string conf_projectFolder = get_parent_folder();
	std::cout << "project folder: " << conf_projectFolder << std::endl;
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	std::string config_file = get_config(conf_projectFolder + "/configs/main_config.json"); 
	FrameSolver frame;
	frame.m_project_folder = conf_projectFolder; 
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
	int start = frame.get_start_id(); 
	int framenum = frame.get_frame_num();
	int increment = 1;
	if (framenum < 0) increment = -1;

	std::string result_folder = frame.m_result_folder;
	if (!std::filesystem::is_directory(result_folder))
	{
		std::filesystem::create_directory(result_folder); 
	}
	std::vector<std::string> subfolders = {
		"association", "render", "render_smth", "clusters", "state", "state_smth",
		"annotation", "triangulation", "anchor_state", "detection", 
		"joints_62", "joints_23", "joints_62_smth", "joints_23_smth",
		"projection"
	};
	for (int i = 0; i < subfolders.size(); i++)
	{
		if (!std::filesystem::is_directory(result_folder + subfolders[i]))
			std::filesystem::create_directory(result_folder + subfolders[i]);
	}

	frame.init_parametric_solver(); 


	frame.saveConfig();

	for (int frameid = start; frameid != start + framenum; frameid+=increment)
	{
		m_renderer.SetBackgroundColor(Eigen::Vector4f(0, 0, 0, 0));

		std::cout << "===========processing frame " << frameid << "===============" << std::endl;


		frame.set_frame_id(frameid);
		TimerUtil::Timer<std::chrono::microseconds> tt;
		tt.Start();
		frame.fetchData();
		float time1 = tt.Elapsed() / 1000.0; 
		//time_log_stream << time1 << " "; 
		std::cout << "fetch data   :  " << time1  << "  ms" << std::endl;
		tt.Start(); 
		if (frameid == start)
		{
			if (frame.m_use_init_cluster)
				frame.load_clusters(); 
			else
				frame.matching_by_tracking();
		}
		else
		{
			frame.pureTracking();
		}

#if 1
		//cv::Mat det_vis = frame.visualizeSkels2D(); 
		std::vector<cv::Mat> det_vis_group(frame.m_render_views.size());
		for (int k = 0; k < frame.m_render_views.size(); k++)
		{
			int camid = frame.m_render_views[k];
			cv::Mat det_vis = frame.tmp_visualizeRawDetPure(camid);
			det_vis_group[k] = det_vis;
		}
		cv::Mat det_vis; 
		packImgBlock(det_vis_group, det_vis); 
		cv::Mat det_vis_small = my_resize(det_vis, frame.m_render_resize_ratio); 
		std::stringstream ss_detvis;
		ss_detvis << result_folder << "/detection/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_detvis.str(),det_vis_small);
#endif 

		float time2 = tt.Elapsed() / 1000.0; 
		//time_log_stream << time2 << " "; 
		std::cout << "Assoc       : " << time2 << "  ms" << std::endl; 
		tt.Start(); 

		frame.save_clusters(); 
		frame.resetSolverStateMarker(); 

		if (frame.m_use_triangulation_only)
		{
			frame.DARKOV_Step1_setsource();
			frame.DirectTriangulation();
			frame.save_skels(); 

			cv::Mat assoc = frame.visualizeIdentity2D();
			cv::Mat assoc_small = my_resize(assoc, frame.m_render_resize_ratio); 
			std::stringstream ss;
			ss << result_folder << "/association/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::imwrite(ss.str(),assoc_small);

			cv::Mat reproj = frame.visualizeProj();
			cv::Mat reproj_small = my_resize(reproj, frame.m_render_resize_ratio); 
			std::stringstream ss_proj;
			ss_proj << result_folder << "/projection/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::imwrite(ss_proj.str(), reproj_small);

			continue; 
		}

		if (frame.m_try_load_anno && frame.try_load_anno())
		{
			frame.save_parametric_data(); 
			frame.DARKOV_Step5_postprocess();
		}
		else if (frameid == start && frame.m_use_init_pose)
		{
			frame.read_parametric_data(); 
			frame.DARKOV_Step5_postprocess();
		}
		// pipeline 3 
		else
		{
			frame.DARKOV_Step1_setsource();
			if (frameid == start && frame.m_use_init_anchor)
			{
				for (int i = 0; i < 4; i++)
				{
					frame.DARKOV_Step2_searchanchor(i);
					frame.saveAnchors(frame.m_result_folder + "/anchor_state/");
					frame.DARKOV_Step2_optimanchor(i);

					//frame.DARKOV_Step2_loadanchor();
				}
			}
			else
			{
				frame.m_params.m_w_anchor_term = 0; 
				for (int i = 0; i < 4; i++)
				{
					frame.mp_bodysolverdevice[i]->m_params.m_w_anchor_term = 0;
				}
			}

			if (frameid == start)
			{
				frame.DARKOV_Step4_fitrawsource(frame.m_initialization_iters);
			}
			else 
			{
				frame.DARKOV_Step4_fitrawsource(frame.m_solve_sil_iters);
				frame.DARKOV_Step3_reassoc_type2();
				frame.DARKOV_Step4_fitreassoc(frame.m_solve_sil_iters_2nd_phase);
			}


			frame.DARKOV_Step5_postprocess();
			frame.save_parametric_data();
		}
		float time3 = tt.Elapsed() / 1000.0; 
		//time_log_stream << time3 << std::endl;
		std::cout << "w/o rendering " << time3 << "  ms" << std::endl;
		
		{
#if 1
			cv::Mat assoc = frame.visualizeIdentity2D();
			std::stringstream ss;
			ss << result_folder << "/association/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::Mat assoc_small = my_resize(assoc, frame.m_render_resize_ratio); 
			cv::imwrite(ss.str(), assoc_small);

			if (!(frame.m_use_init_pose && frameid == start))
			{
				//cv::Mat reassoc = frame.visualizeReassociation();
				//std::stringstream ss_reassoc;
				//ss_reassoc << result_folder << "/reassoc2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
				//cv::imwrite(ss_reassoc.str(), reassoc);

				//cv::Mat reproj = frame.visualizeVisibility();
				//std::stringstream ss_proj;
				//ss_proj << result_folder << "/proj2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
				//cv::imwrite(ss_proj.str(), reproj);

				//cv::Mat beforeimg = frame.visualizeSwap();
				//std::stringstream ss_before;
				//ss_before << result_folder << "/before_swap2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
				//cv::imwrite(ss_before.str(), beforeimg);

			}
			//cv::Mat rawfit = frame.visualizeRawAssoc();
			//std::stringstream ss_rawassoc; 
			//ss_rawassoc << result_folder << "/fitting/" << std::setw(6) << std::setfill('0') << frameid << ".png";
			//cv::imwrite(ss_rawassoc.str(), rawfit); 

			//frame.pipeline2_searchanchor();
			//frame.saveAnchors(result_folder + "/anchor_state_252");
#endif 
		}

		m_renderer.clearAllObjs();

		auto solvers = frame.mp_bodysolverdevice;

		// save joints 
		{
			for (int pid = 0; pid < pignum; pid++)
			{
				auto joints_62 = solvers[pid]->GetJoints(); 
				save_points(frame.m_result_folder + "/joints_62/", pid, frameid, joints_62); 
				auto joints_23 = solvers[pid]->getRegressedSkel_host(); 
				save_points(frame.m_result_folder + "/joints_23/", pid, frameid, joints_23);

			}
		}

		for (int pid = 0; pid < pignum; pid++)
		{
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

		std::vector<int> render_views = frame.m_render_views; 

		std::vector<cv::Mat> rawImgs; 
		for (int k = 0; k < render_views.size(); k++)
			rawImgs.push_back(frame.get_imgs_undist()[k]); 
		cv::Mat a(cv::Size(1920, 1080), CV_8UC3);
		rawImgs.push_back(a);
		cv::Mat pack_raw;
		packImgBlock(rawImgs, pack_raw);

		std::vector<cv::Mat> all_renders(render_views.size());
		for (int k = 0; k < render_views.size(); k++)
		{
			int camid = render_views[k]; 
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			m_renderer.Draw();
			cv::Mat img = m_renderer.GetImageOffscreen(); 
			all_renders[camid] = img;
		}

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1)); 
		for (int pid = 0; pid < pignum; pid++)
			m_renderer.colorObjs[pid]->isMultiLight = true; 
		m_renderer.createSceneDetailed(conf_projectFolder, 1.1);

		Eigen::Vector3f pos2(0, -0, 4.1);
		Eigen::Vector3f up2(0, 1, -0);
		Eigen::Vector3f center2(0, -0, 0);
		m_renderer.s_camViewer.SetExtrinsic(pos2, up2, center2);
		m_renderer.Draw();
		cv::Mat img = m_renderer.GetImageOffscreen();
		all_renders.push_back(img);

		cv::Mat packed_render;
		packImgBlock(all_renders, packed_render);

		cv::Mat blend;
		overlay_render_on_raw_gpu(packed_render, pack_raw, blend);

		cv::Mat blend_small = my_resize(blend, frame.m_render_resize_ratio);
		std::stringstream all_render_file;
		all_render_file << result_folder << "/render/" << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
		cv::imwrite(all_render_file.str(), blend_small);
		
		std::cout << "render:       " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;

	}

	/// smooth. TODO: add processs control in the config file .
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
	/// write smoothed render images. 
	frame.is_smth = true;
	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "===========render smoothed results " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		frame.load_clusters();
		frame.read_parametric_data();
		m_renderer.clearAllObjs();
		auto solvers = frame.mp_bodysolverdevice;
		m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

		for (int pid = 0; pid < pignum; pid++)
		{
			int colorid = frame.m_pig_names[pid];
			RenderObjectColor* p_model = new RenderObjectColor();
			solvers[pid]->UpdateNormalFinal();
			p_model->SetVertices(solvers[pid]->GetVertices());
			p_model->SetNormal(solvers[pid]->GetNormals());
			p_model->SetFaces(solvers[pid]->GetFacesVert());
			p_model->SetColor(m_CM[colorid]);
			m_renderer.colorObjs.push_back(p_model);
		}

		std::vector<int> render_views = frame.m_render_views; 

		std::vector<cv::Mat> rawImgs; 
		for (int k = 0; k < render_views.size(); k++)
		{
			int camid = render_views[k]; 
			rawImgs.push_back(frame.get_imgs_undist()[camid]);
		}
		cv::Mat a(cv::Size(1920, 1080), CV_8UC3);
		rawImgs.push_back(a);
		cv::Mat pack_raw;
		packImgBlock(rawImgs, pack_raw);

		std::vector<cv::Mat> all_renders(render_views.size());
		for (int k = 0; k < render_views.size(); k++)
		{
			int camid = render_views[k];
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			cv::Mat img = m_renderer.GetImageOffscreen();
			all_renders[k] = img;
		}
		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));
		for (int pid = 0; pid < pignum; pid++)
			m_renderer.colorObjs[pid]->isMultiLight = true;
		m_renderer.createSceneDetailed(conf_projectFolder, 1.1);

		Eigen::Vector3f pos2(0, -0, 4.1);
		Eigen::Vector3f up2(0, 1, -0);
		Eigen::Vector3f center2(0, -0, 0);
		m_renderer.s_camViewer.SetExtrinsic(pos2, up2, center2);
		m_renderer.Draw();
		cv::Mat img = m_renderer.GetImageOffscreen();
		all_renders.push_back(img);

		cv::Mat packed_render;
		packImgBlock(all_renders, packed_render);

		cv::Mat blend;
		overlay_render_on_raw_gpu(packed_render, pack_raw, blend);

		cv::Mat small_img = my_resize(blend, frame.m_render_resize_ratio);
		std::stringstream all_render_file;
		all_render_file << frame.m_result_folder << "/render_smth/" << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
		cv::imwrite(all_render_file.str(), small_img);

		//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		//while (!glfwWindowShouldClose(windowPtr))
		//{
		//	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		//	m_renderer.Draw();

		//	glfwSwapBuffers(windowPtr);
		//	glfwPollEvents();
		//};
		//return 0;
	}

 	return 0;
}
