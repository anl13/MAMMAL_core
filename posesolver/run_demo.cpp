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
#include "framesolver.h"
#include "../utils/mesh.h"
#include <vector_functions.hpp>
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"

#include "main.h"

void run_demo_20211008()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	//std::string config_file = get_config();
	std::string config_file = "configs/config_20190704_fordemo.json"; 
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
		"debug", "render_all", "render_last"
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

	std::string output_dir = "render_7888"; 

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

		std::cout << "w/o rendering " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;
		std::vector<int> ids = { 0,3,8};
		auto rawimgs = frame.m_imgsUndist;
		for (int k = 0; k < ids.size(); k++)
		{
			int id = ids[k];
			std::stringstream ss;
			ss << "D:/results/paper_teaser/0704_demo/" << output_dir << "/raw_" << id << ".png";
			cv::imwrite(ss.str(), rawimgs[id]);

			//std::stringstream ss_rawdetpre1;
			//ss_rawdetpre1 << "D:/results/paper_teaser/0704_demo/" << output_dir << "/rawdetpure1_" << id << ".png";
			//cv::Mat rawdetpure1 = frame.tmp_visualizeRawDetPure(id, false);
			//cv::imwrite(ss_rawdetpre1.str(), rawdetpure1);


			std::stringstream ss_rawdetpre;
			ss_rawdetpre << "D:/results/paper_teaser/0704_demo/" << output_dir << "/rawdetpure4_" << id << ".png";
			cv::Mat rawdetpure = frame.tmp_visualizeRawDetPure(id);
			cv::imwrite(ss_rawdetpre.str(), rawdetpure);

			for (int pid = 0; pid < 4; pid++)
			{
				std::stringstream ss_rawdet;
				ss_rawdet << "D:/results/paper_teaser/0704_demo/" << output_dir << "/rawdet_" << id << "_" << pid << ".png";
				cv::Mat rawdet = frame.tmp_visualizeRawDet(id, pid);
				cv::imwrite(ss_rawdet.str(), rawdet);
			}
			continue; 

			//cv::Mat rawdet3 = blend_images(rawimgs[id], rawdetpure, 0.5); 
			//std::stringstream ss_rawdet3;
			//ss_rawdet3 << "D:/results/paper_teaser/0704_demo/" << output_dir << "/rawdet3_" << id << ".png";
			//cv::imwrite(ss_rawdet3.str(), rawdet3);

			cv::Mat assoc = frame.visualizeIdentity2D(id);
			std::stringstream ss1;
			ss1 << test_result_folder << "/" << output_dir << "/assoc_" << k << "_" << std::setw(6) << std::setfill('0') << frameid << ".png";
			cv::imwrite(ss1.str(), assoc);

			for (int pid = 0; pid < 4; pid++)
			{
				cv::Mat assoc2 = frame.visualizeIdentity2D(id, pid);
				std::stringstream ss2;
				ss2 << test_result_folder << "/" << output_dir << "/assoc_" << k << "_" << pid << "_" << std::setw(6) << std::setfill('0') << frameid << ".png";
				cv::imwrite(ss2.str(), assoc2);

				cv::Mat assoc3 = frame.tmp_visualizeIdentity2D(id, pid);
				std::stringstream ss3;
				ss3 << test_result_folder << "/" << output_dir << "/group_" << k << "_" << pid << ".png";
				cv::imwrite(ss3.str(), assoc3);
			}
		}
		return; 

		cv::Mat rawfit = frame.visualizeRawAssoc();
		cv::Mat rawfit_small = my_resize(rawfit, 1);
		std::stringstream ss_rawassoc;
		ss_rawassoc << test_result_folder << "/" << output_dir << "/fitting" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_rawassoc.str(), rawfit_small);


		frame.DARKOV_Step1_setsource();
		frame.DirectTriangulation();
		cv::Mat reproj = frame.visualizeProj(1);
		std::stringstream ss_proj;
		ss_proj << test_result_folder << "/" << output_dir << "/proj" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_proj.str(), reproj);
		
		frame.read_parametric_data(); 
		cv::Mat reproj_fit = frame.visualizeProj(1);
		std::stringstream ss_proj_fit;
		ss_proj_fit << test_result_folder << "/" << output_dir << "/proj_fit" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_proj_fit.str(), reproj_fit);

		continue; 
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
		frame.DARKOV_Step4_fitrawsource(frame.m_solve_sil_iters);
		frame.DARKOV_Step3_reassoc_type2();
		frame.DARKOV_Step4_fitreassoc(frame.m_solve_sil_iters_2nd_phase);

		frame.DARKOV_Step5_postprocess();
		frame.save_parametric_data();

		std::cout << "w/o rendering " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;

		std::vector<int> reassoc_ids = { 0,1,2,3 };
		cv::Mat reassoc = frame.visualizeReassociation(reassoc_ids,0);
		std::stringstream ss_reassoc;
		ss_reassoc << test_result_folder << "/reassoc2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_reassoc.str(), reassoc);

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


		std::string prefix = "anchor"; 
		m_renderer.clearAllObjs();
		std::vector<int> render_pig_ids = { 1,3 }; 
		auto solvers = frame.mp_bodysolverdevice;
		for (int pid = 0; pid < 4; pid++)
		{
			std::stringstream ss_obj;
			ss_obj << "D:/results/paper_teaser/0704_demo/debug/anchor_" << pid << ".txt";
			solvers[pid]->saveState(ss_obj.str()); 
		}
		for (int k = 0; k  <render_pig_ids.size(); k++)
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
		//rawImgs.push_back(a);
		packImgBlock(rawImgs, pack_raw);

		std::vector<cv::Mat> all_renders(cams.size());
		for (int camid = 0; camid < cams.size(); camid++)
		{
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			m_renderer.Draw();
			cv::Mat img = m_renderer.GetImage();
			std::stringstream ss_perimg; 
			ss_perimg << test_result_folder << "/render_last/" << prefix << "_" << camid << ".png"; 
			cv::imwrite(ss_perimg.str(), img); 
			all_renders[camid] = img;
		}

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));
		for (int k = 0; k < render_pig_ids.size(); k++)
		{
			int pid = render_pig_ids[k]; 
			m_renderer.colorObjs[k]->isMultiLight = true;
		}

		Eigen::Vector3f pos2(0.0988611, -0.0113558, 3.00438);
		Eigen::Vector3f up2(0.00346774, 0.999541, -0.0301062);
		Eigen::Vector3f center2(0.0589942, -0.0909324, 0.00569892);
		m_renderer.s_camViewer.SetExtrinsic(pos2, up2, center2);

		m_renderer.Draw();
		cv::Mat img = m_renderer.GetImage();
		all_renders.push_back(img);

		cv::Mat packed_render;
		packImgBlock(all_renders, packed_render);

		cv::Mat blend;
		overlay_render_on_raw_gpu(packed_render, pack_raw, blend);

		cv::Mat blend_small = my_resize(blend, 0.25);
		std::stringstream all_render_file;
		all_render_file << test_result_folder << "/render_last/" << prefix << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
		cv::imwrite(all_render_file.str(), blend_small);

		std::cout << "total:       " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;

		//Eigen::Vector3f pos4(1.11903, 1.58545, 1.70294);
		//Eigen::Vector3f up4(-0.243542, -0.541976, 0.804331);
		//Eigen::Vector3f center4(0.544902, -0.361053, 0.206058);
		//m_renderer.s_camViewer.SetExtrinsic(pos4, up4, center4); 
		m_renderer.s_camViewer.SetExtrinsic(cams[8].R, cams[8].T + Eigen::Vector3f(0,0,0.125)); 
		m_renderer.Draw(); 
		cv::Mat imgforpaper = m_renderer.GetImage(); 
		cv::imwrite(test_result_folder + "/render_last/" + prefix + ".png", imgforpaper); 
		if (frameid == start ) {
			GLFWwindow* windowPtr = m_renderer.s_windowPtr;
			while (!glfwWindowShouldClose(windowPtr))
			{
				//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

				m_renderer.Draw();

				glfwSwapBuffers(windowPtr);
				glfwPollEvents();
			};
		}
	}

}


void run_demo_visualize_depth()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	//std::string config_file = get_config();
	std::string config_file = "configs/config_20190704_fordemo.json";
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
		"debug", "render_all", "render_last"
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

	for (int frameid = 7888;;)
	{
		m_renderer.SetBackgroundColor(Eigen::Vector4f(0, 0, 0, 0));

		frame.set_frame_id(frameid);
		frame.fetchData();

		frame.load_clusters();
		frame.resetSolverStateMarker();
		frame.DARKOV_Step1_setsource();

		for (int pid = 0; pid < 4; pid++)
		{
			if (pid != 1 && pid != 3) continue; 
			cv::Mat assoc3 = frame.tmp_visualizeIdentity2D(8, pid);
			std::stringstream ss3;
			ss3 << test_result_folder << "/debug/group_" << "_" << pid << ".png";
			cv::imwrite(ss3.str(), assoc3);
		}

		//if (frame.m_use_init_anchor)
		//{
		//	for (int i = 0; i < 4; i++)
		//	{
		//		frame.DARKOV_Step2_searchanchor(i);
		//		frame.saveAnchors(frame.m_result_folder + "/anchor_state/");
		//		frame.DARKOV_Step2_optimanchor(i);
		//	}
		//}

		frame.DARKOV_Step2_loadanchor();
		//for (int i = 0; i < 4; i++)
		//	frame.DARKOV_Step2_optimanchor(i);
		for (int k = 0; k < 4; k++)
		{
			std::stringstream ss; 
			ss << test_result_folder << "/debug/anchor_" << k << ".txt"; 
			frame.mp_bodysolverdevice[k]->readState(ss.str());
			frame.mp_bodysolverdevice[k]->UpdateVertices(); 
			frame.mp_bodysolverdevice[k]->UpdateNormalFinal(); 
			frame.mp_bodysolverdevice[k]->map_reduced_vertices(); 
		}

#if 1 // comment to visualize anchor result
		frame.DARKOV_Step4_fitrawsource(frame.m_solve_sil_iters);
		frame.DARKOV_Step3_reassoc_type2();
		frame.DARKOV_Step4_fitreassoc(frame.m_solve_sil_iters_2nd_phase);

		frame.DARKOV_Step5_postprocess();
		frame.save_parametric_data();

		cv::Mat reassoc = frame.visualizeReassociation({ 1,3 }, 8, false);
		std::stringstream ss_reassoc;
		ss_reassoc << test_result_folder << "/reassoc2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_reassoc.str(), reassoc);
#endif 
		//frame.read_parametric_data(); 
		std::string prefix = "anchor";
		m_renderer.clearAllObjs();
		std::vector<Eigen::Vector3f> id_colors = {
			{1.0f, 0.0f,0.0f},
			{0.0f, 1.0f, 0.0f},
			{0.0f, 0.0f, 1.0f},
			{1.0f, 1.0f, 0.0f}
				};
		std::vector<int> render_pig_ids = { 1,3 };
		auto solvers = frame.mp_bodysolverdevice;
		for (int k = 0; k < render_pig_ids.size(); k++)
		{
			int pid = render_pig_ids[k];
			RenderObjectColor* p_model = new RenderObjectColor();
			solvers[pid]->UpdateNormalFinal();
			int colorid = frame.m_pig_names[pid];
			p_model->SetVertices(solvers[pid]->GetVertices());
			p_model->SetNormal(solvers[pid]->GetNormals());
			p_model->SetFaces(solvers[pid]->GetFacesVert());
			//p_model->SetColor(m_CM[colorid]);
			p_model->SetColor(id_colors[pid]);
			m_renderer.colorObjs.push_back(p_model);

			std::vector<Eigen::Vector3f> joints = solvers[pid]->GetJoints();
		}

		m_renderer.s_camViewer.SetExtrinsic(cams[8].R, cams[8].T);
		float * depth_device = m_renderer.renderDepthDevice();
		// render depth 
		cv::Mat depth;
		depth.create(cv::Size(1920, 1080), CV_32FC1);
		cudaMemcpy(depth.data, depth_device, 1920 * 1080 * sizeof(float), cudaMemcpyDeviceToHost);

		cv::Mat depth_pseudo = pseudoColor(depth);

		cv::imshow("depth", depth_pseudo);
		cv::imwrite(test_result_folder + "/debug/depth_afteroptim.png", depth_pseudo); 
		cv::waitKey();
		cv::destroyAllWindows();

		// visibility check 
		std::vector<uchar> visibility(solvers[3]->GetVertexNum(), 0);
		pcl::gpu::DeviceArray<Eigen::Vector3f> points_device;
		points_device.upload(solvers[3]->GetVertices());

		TimerUtil::Timer<std::chrono::microseconds> tt1;
		tt1.Start();
		check_visibility(depth_device, WINDOW_WIDTH, WINDOW_HEIGHT, points_device,
			cams[8].K, cams[8].R, cams[8].T, visibility);
		std::cout << tt1.Elapsed() << std::endl;

		int vertexNum = solvers[3]->GetVertexNum();
		std::vector<Eigen::Vector3f> colors(vertexNum, Eigen::Vector3f(1.0f, 1.0f, 1.0f));

		TimerUtil::Timer<std::chrono::microseconds> tt;
		tt.Start();
		std::vector<Eigen::Vector3f> vertices = solvers[3]->GetVertices(); 
		for (int i = 0; i < vertexNum; i++)
		{
			Eigen::Vector3f v = vertices[i];
			Eigen::Vector3f uv = project(cams[8], v);
			float d = queryDepth(depth, uv(0), uv(1));
			v = cams[8].R * v + cams[8].T;
			//std::cout << "d: " << d << "  gt: " << v(2) << std::endl;
			if (d > 0 && abs(d - v(2)) < 0.02f)
			{
				colors[i] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
			}
			else
			{
				colors[i] = Eigen::Vector3f(0.f, 0.f, 1.0f);
			}
			if (visibility[i] > 0) colors[i] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
			else colors[i] = Eigen::Vector3f(0.f, 0.f, 1.0f);
		}

		m_renderer.clearAllObjs(); 
		RenderObjectMesh* meshcolor = new RenderObjectMesh();
		meshcolor->SetVertices(vertices);
		meshcolor->SetFaces(solvers[3]->GetFacesVert());
		meshcolor->SetColors(colors);
		meshcolor->SetNormal(solvers[3]->GetNormals());
		m_renderer.meshObjs.push_back(meshcolor); 
		// end visibility check

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));
		for (int k = 0; k < render_pig_ids.size(); k++)
		{
			int pid = render_pig_ids[k];
			m_renderer.colorObjs[k]->isMultiLight = true;
		}

		//Eigen::Vector3f pos2(0.0988611, -0.0113558, 3.00438);
		//Eigen::Vector3f up2(0.00346774, 0.999541, -0.0301062);
		//Eigen::Vector3f center2(0.0589942, -0.0909324, 0.00569892);
		//m_renderer.s_camViewer.SetExtrinsic(pos2, up2, center2);

		GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		while (!glfwWindowShouldClose(windowPtr))
		{
			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			m_renderer.Draw();
			glfwSwapBuffers(windowPtr);
			glfwPollEvents();
		};

		break;
	}

}
