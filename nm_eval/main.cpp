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
	//run_visualize_gt(); 
	//run_fitgt(); 
	//run_eval_sil(); 
	run_eval_reassoc(); 

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


int run_fitgt()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	std::string config_file = "configs/config_20190704_gt.json";
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

	std::string test_result_folder = frame.m_result_folder;
	if (!boost::filesystem::is_directory(test_result_folder))
	{
		boost::filesystem::create_directory(test_result_folder);
	}
	std::vector<std::string> subfolders = {
		"assoc", "render_all", "clusters", "state", "reassoc2", "proj2", "before_swap2",
		"fitting", "annotation", "skels", "anchor_state", "rawdet", "joints_62", "joints_23"
	};
	for (int i = 0; i < subfolders.size(); i++)
	{
		if (!boost::filesystem::is_directory(test_result_folder + subfolders[i]))
			boost::filesystem::create_directory(test_result_folder + subfolders[i]);
	}

	frame.init_parametric_solver();

	frame.saveConfig();

	for (int i = 0; i < 70; i++)
	{
		int frameid = 750 + 25 * i;
		m_renderer.SetBackgroundColor(Eigen::Vector4f(0, 0, 0, 0));

		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		TimerUtil::Timer<std::chrono::microseconds> tt;
		tt.Start();

		frame.set_frame_id(frameid);
		frame.fetchGtData();
		frame.fetchData();


		//frame.matching_by_tracking(); 

		frame.resetSolverStateMarker();

		frame.DARKOV_Step1_setsource();
		frame.DirectTriangulation();
		frame.save_skels();
		std::cout << "w/o rendering " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;

		cv::Mat assoc = frame.visualizeIdentity2D();
		cv::Mat assoc_small = my_resize(assoc, 0.25);
		std::stringstream ss;
		ss << test_result_folder << "/assoc/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss.str(), assoc_small);

		cv::Mat reproj = frame.visualizeProj();
		cv::Mat reproj_small = my_resize(reproj, 0.25);
		std::stringstream ss_proj;
		ss_proj << test_result_folder << "/proj2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_proj.str(), reproj_small);

		//frame.DARKOV_Step2_searchanchor(i);
		//frame.saveAnchors(frame.m_result_folder + "/anchor_state/");
		//frame.DARKOV_Step2_optimanchor(i);

		frame.DARKOV_Step4_fitrawsource();
		frame.DARKOV_Step5_postprocess();
		frame.save_parametric_data();

		std::cout << "w/o rendering " << tt.Elapsed() / 1000.0 << "  ms" << std::endl;

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

			all_renders[camid] = img;
		}

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));
		for (int pid = 0; pid < pignum; pid++)
			m_renderer.colorObjs[pid]->isMultiLight = true;
		m_renderer.createSceneDetailed(conf_projectFolder, 1);

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
		all_render_file << test_result_folder << "/render_all/" << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
		cv::imwrite(all_render_file.str(), blend_small);

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
	}

	return 0;
}
