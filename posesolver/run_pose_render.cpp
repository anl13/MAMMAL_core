#include "main.h"
#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
//#include <boost/filesystem.hpp>
#include <filesystem>
#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
//#include "../articulation/pigmodel.h"
//#include "../articulation/pigsolver.h"
#include "framesolver.h"
#include "../utils/mesh.h"
#include <vector_functions.hpp>
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"
#include "../render/render_utils.h"

int run_pose_render()
{
	show_gpu_param();
	std::string conf_projectFolder = "H:/MAMMAL_core/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");

	FrameSolver frame;
	frame.configByJson(conf_projectFolder + get_config(conf_projectFolder + "/configs/main_config.json"));

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(0);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];

	std::string smth_folder = frame.m_result_folder + "/render_smth/";
	if (!std::filesystem::is_directory(smth_folder))
		std::filesystem::create_directory(smth_folder);

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
	frame.is_smth = true;
	int start = frame.get_start_id();
	int pignum = frame.m_pignum;

	Camera another_cam = Camera::getDefaultCameraUndist();
	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		frame.load_clusters();
		frame.read_parametric_data();

		/// 2022.04.03 draw assoc
		//cv::Mat assoc = frame.visualizeIdentity2D(2);
		//cv::Mat assoc_small = my_resize(assoc, 0.5); 
		//std::stringstream ss;
		//ss << frame.m_result_folder << "/assoc3/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		//cv::imwrite(ss.str(), assoc_small);
		//continue; 

		m_renderer.clearAllObjs();
		auto solvers = frame.mp_bodysolverdevice;

		m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

		/// 2022.03.30 used for nm_video1
		//std::vector<cv::Mat> rawdetlist; 
		//for (int i = 0; i < 10; i++)
		//{
		//	cv::Mat assoc = frame.tmp_visualizeRawDetPure(i);
		//	cv::Mat assoc_small = my_resize(assoc, 0.25);
		//	rawdetlist.push_back(assoc_small); 
		//}
		//cv::Mat fullrawdet; 
		//packImgBlock(rawdetlist, fullrawdet);
		//std::stringstream ss;
		//ss << smth_folder << std::setw(6) << std::setfill('0') << frameid << ".png";
		//cv::imwrite(ss.str(), fullrawdet);

		///// 2022.03.30 proj for nm_video2
		//frame.DARKOV_Step5_postprocess(); 
		//cv::Mat reproj = frame.visualizeProj();
		//std::stringstream ss_proj;
		//ss_proj << frame.m_result_folder << "/proj2/" << std::setw(6) << std::setfill('0') << frameid << ".png";
		//cv::imwrite(ss_proj.str(), reproj);
		//continue; 

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

		std::vector<int> render_views = {0,1,2,3,4,5,6,7,8,9};

		std::vector<cv::Mat> rawImgs = frame.get_imgs_undist();
		
		std::vector<cv::Mat> rawImgsSelect;
		for (int k = 0; k < render_views.size(); k++) rawImgsSelect.push_back(rawImgs[render_views[k]]);

		std::vector<cv::Mat> all_renders(render_views.size());
		for (int k = 0; k < render_views.size(); k++)
		{
			int camid = render_views[k];
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			cv::Mat img = m_renderer.GetImageOffscreen();
			all_renders[k] = img;
		}
		m_renderer.createSceneDetailed(conf_projectFolder, 1.1);
		//m_renderer.createSceneHalf(conf_projectFolder, 1.1);

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1)); 
		Eigen::Vector3f pos3(0.0, -0.0, 4.1);
		Eigen::Vector3f up3(0.0, 0.1, -0.0);
		Eigen::Vector3f center3(0.0, -0.0, 0.0); 
		m_renderer.s_camViewer.SetExtrinsic(pos3, up3, center3);
		m_renderer.s_camViewer.SetIntrinsic(another_cam.K, 1920, 1080); 
		cv::Mat img = m_renderer.GetImageOffscreen(); 
		all_renders.push_back(img);
		rawImgsSelect.push_back(img); 
		m_renderer.s_camViewer.SetIntrinsic(cams[0].K, 1920, 1080);

		//Eigen::Vector3f pos4(-2.2866, -1.61577, 2.71787);
		//Eigen::Vector3f up4(0.52083, 0.378857, 0.764986);
		//Eigen::Vector3f center4(0.241644, -0.127209, 0.250072);
		//m_renderer.s_camViewer.SetExtrinsic(pos4, up4, center4); 
		//img = m_renderer.GetImageOffscreen();
		//std::stringstream ss_out; 
		//ss_out << frame.m_result_folder << "/render_all/render" << std::setw(6) << std::setfill('0')
		//	<< frameid << ".png";
		//cv::imwrite(ss_out.str(), img); 

		cv::Mat packed_render;
		packImgBlock(all_renders, packed_render);
		cv::Mat pack_raw; 
		packImgBlock(rawImgsSelect, pack_raw); 

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


void run_trajectory()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_blend");
	std::vector<Eigen::Vector3f> CM2 = getColorMapEigenF("anliang_render");
	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	// pose1: face towards feeding area
	//Eigen::Vector3f up1; up1 << 0.267364, 0.545056, 0.794626;
	//Eigen::Vector3f pos1; pos1 << -0.994971, -1.44537, 1.43656;
	//Eigen::Vector3f center1; center1 << 0.0266508, -0.121682, 0.192684;

	// pose2: face towards drinking area
	Eigen::Vector3f pos1; pos1 << 1.05564, -1.11705, 1.11743;
	Eigen::Vector3f up1; up1 << -0.256466, 0.308172, 0.916109;
	Eigen::Vector3f center1; center1 << 0.113146, -0.0304874, 0.488063;

	// init renderer 
	Renderer::s_Init(true);

	Renderer m_renderer(conf_projectFolder + "/render/shader/");

	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos1, up1, center1);

	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
	SkelTopology topo = getSkelTopoByType("UNIV");

	int window = 50;

	std::vector<Eigen::Vector2i> bones = {
		{0,1}, {0,2}, {1,2}, {1,3}, {2,4},
		 {5,7}, {7,9}, {6,8}, {8,10},
		{20,18},
		{18,11}, {18,12}, {11,13}, {13,15}, {12,14}, {14,16},
		{0,20},{5,20},{6,20}
	};
	std::vector<int> kpt_color_ids = {
		0,0,0,0,0,
		3,4,3,4,3,4,
		5,6,5,6,5,6,
		2,2,2,2,2,2
	};
	std::vector<int> bone_color_ids = {
		0,0,0,0,0,3,3,4,4,
		2,5,6,5,5,6,6,
		2,3,4
	};

	cv::VideoWriter writer("D:/results/teaser/1003stand.avi", cv::VideoWriter::fourcc('m', 'p', 'e', 'g'), 25.0, cv::Size(1920, 1080));
	if (!writer.isOpened())
	{
		std::cout << "not open" << std::endl;
		return;
	}

	std::vector<std::deque<std::vector<Eigen::Vector3f> > > joints_queues;
	joints_queues.resize(4);

	FrameSolver frame;
	std::string configfile = "configs/config_20191003_stand2.json"; 
	frame.configByJson(conf_projectFolder + configfile);
	frame.init_parametric_solver(); 
	auto solvers = frame.get_solvers(); 
	int start = frame.get_start_id(); 
	int num = frame.get_frame_num(); 
	frame.is_smth = true; 
	std::vector<int> ids_draw = { 1,3 };
	for (int frameid = start; frameid < start + num; frameid++)
	{
		std::cout << frameid << std::endl;

		frame.m_is_read_image = false;
		frame.set_frame_id(frameid);
		frame.read_parametric_data();
		// push to queue 
		for (int pid = 0; pid < frame.m_pignum; pid++)
		{
			std::stringstream ss;

			std::vector<Eigen::Vector3f> points = solvers[pid]->getRegressedSkel_host(); 

			joints_queues[pid].push_back(points);
			if (joints_queues[pid].size() > window) joints_queues[pid].pop_front();
		}

		m_renderer.clearAllObjs();


#if 0 // trajectory type1 
		for (int index = 0; index < joints_queues[0].size(); index++)
		{
			for (int pid = 2; pid < 3; pid++)
			{
				int ratio_index = window - joints_queues[0].size() + index;
				float ratio = (2 - (ratio_index / float(window)));

				std::vector<Eigen::Vector3f> skels = joints_queues[pid][index];
				std::vector<Eigen::Vector3f> balls;
				std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
				GetBallsAndSticks(skels, bones, balls, sticks);
				int jointnum = skels.size();
				std::vector<float> ball_sizes;
				ball_sizes.resize(jointnum, 0.005 / ratio);
				std::vector<float> stick_sizes;
				stick_sizes.resize(sticks.size(), 0.002 / ratio);
				std::vector<Eigen::Vector3f> ball_colors(jointnum);
				std::vector<Eigen::Vector3f> stick_colors(sticks.size());
				for (int i = 0; i < jointnum; i++)
				{
					ball_colors[i] = CM[kpt_color_ids[i]] * ratio;
				}
				for (int i = 0; i < sticks.size(); i++)
				{
					//stick_colors[i] = CM[bone_color_ids[i]] * ratio;
					stick_colors[i] = CM2[pid] * ratio;
				}

				BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
					balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
				p_skel->isMultiLight = false;
				m_renderer.skels.push_back(p_skel);

				RenderObjectColor* p_model = new RenderObjectColor();
				solvers[pid]->UpdateNormalFinal();
				p_model->SetVertices(solvers[pid]->GetVertices());
				p_model->SetNormal(solvers[pid]->GetNormals());
				p_model->SetFaces(solvers[pid]->GetFacesVert());
				p_model->SetColor(CM2[pid]);
				p_model->isFill = false;
				m_renderer.colorObjs.push_back(p_model);
			}
		}
#else 
		for (int index = 0; index < joints_queues[0].size(); index++)
		{
			if (index == joints_queues[0].size() - 1)
			{
				for (int pid = 0; pid < frame.m_pignum; pid++)
				{
					int pigname = frame.m_pig_names[pid];
					if (!in_list(pigname, ids_draw)) continue;
					int ratio_index = window - joints_queues[0].size() + index;
					float ratio = (2 - (ratio_index / float(window)));

					std::vector<Eigen::Vector3f> skels = joints_queues[pid][index];
					std::vector<Eigen::Vector3f> balls;
					std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
					GetBallsAndSticks(skels, bones, balls, sticks);
					int jointnum = skels.size();
					std::vector<float> ball_sizes;
					ball_sizes.resize(jointnum, 0.015);
					std::vector<float> stick_sizes;
					stick_sizes.resize(sticks.size(), 0.009);
					std::vector<Eigen::Vector3f> ball_colors(jointnum);
					std::vector<Eigen::Vector3f> stick_colors(sticks.size());
					for (int i = 0; i < jointnum; i++)
					{
						ball_colors[i] = CM[kpt_color_ids[i]] * ratio;
					}
					for (int i = 0; i < sticks.size(); i++)
					{
						stick_colors[i] = CM[bone_color_ids[i]] * ratio;
					}

					BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
						balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
					p_skel->isMultiLight = false;
					m_renderer.skels.push_back(p_skel);

					RenderObjectColor* p_model = new RenderObjectColor();
					solvers[pid]->UpdateNormalFinal();
					p_model->SetVertices(solvers[pid]->GetVertices());
					p_model->SetNormal(solvers[pid]->GetNormals());
					p_model->SetFaces(solvers[pid]->GetFacesVert());
					p_model->SetColor(Eigen::Vector3f(0.8,0.8,0.8));
					p_model->isFill = false;
					m_renderer.colorObjs.push_back(p_model);
				}
			}
			else
			{
				for (int pid = 0; pid < frame.m_pignum; pid++)
				{
					int pigname = frame.m_pig_names[pid];
					if (!in_list(pigname, ids_draw)) continue;
					float ratio = (2 - (index / float(window)));

					std::vector<Eigen::Vector3f> skels = joints_queues[pid][index];
					std::vector<Eigen::Vector3f> balls;
					std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
					GetBallsAndSticks(skels, bones, balls, sticks);
					int jointnum = skels.size();
					std::vector<float> ball_sizes;
					ball_sizes.resize(jointnum, 0.002);

					// sticks: connect last to current
					sticks.clear();
					sticks.resize(jointnum);
					std::vector<float> stick_sizes;
					for (int k = 0; k < jointnum; k++)
					{
						sticks[k].first = joints_queues[pid][index][k];
						sticks[k].second = joints_queues[pid][index + 1][k];
					}
					stick_sizes.resize(sticks.size(), 0.001);
					std::vector<Eigen::Vector3f> ball_colors(jointnum);
					std::vector<Eigen::Vector3f> stick_colors(sticks.size());
					for (int i = 0; i < jointnum; i++)
					{
						ball_colors[i] = CM[kpt_color_ids[i]] * ratio;
					}
					for (int i = 0; i < sticks.size(); i++)
					{
						stick_colors[i] = CM[kpt_color_ids[i]] * ratio;
					}

					BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
						balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
					m_renderer.skels.push_back(p_skel);
				}
			}
		}
#endif 
		m_renderer.createSceneHalf2(conf_projectFolder, 1.08);

		cv::Mat img = m_renderer.GetImageOffscreen();
		writer.write(img); 

		//std::stringstream output_ss; 
		//output_ss << "G:/pig_middle_data/teaser/video/trajectory_" << std::setw(6) << std::setfill('0') << frameid << ".png"; 
		//cv::imwrite(output_ss.str(), img);
		//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		//while (!glfwWindowShouldClose(windowPtr))
		//{
		//	m_renderer.Draw();
		//	glfwSwapBuffers(windowPtr);
		//	glfwPollEvents();
		//};
		//exit(-1);
	}
}

int run_pose_render_demo()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");

	FrameSolver frame;
	frame.configByJson(conf_projectFolder + "/configs/config_20190704_fordemo.json");

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
	frame.is_smth = false;
	int start = frame.get_start_id();
	int pignum = frame.m_pignum;
	for (int frameid = 7880; frameid < 7881; frameid++)
	{
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();

		frame.load_clusters();
		frame.read_parametric_data();

		// render 2d 
		

		// render 3d 
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
			p_model->isFill = false; 
			m_renderer.colorObjs.push_back(p_model);
		}

		/// add skels 
#if 1
		std::vector<Eigen::Vector3f> CM3 = getColorMapEigenF("anliang_blend");
		Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
		Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
		MeshEigen ballMeshEigen(ballMesh);
		MeshEigen stickMeshEigen(stickMesh);
		std::vector<Eigen::Vector2i> bones = {
		{0,1}, {0,2}, {1,2}, {1,3}, {2,4},
		 {5,7}, {7,9}, {6,8}, {8,10},
		{20,18},
		{18,11}, {18,12}, {11,13}, {13,15}, {12,14}, {14,16},
		{0,20},{5,20},{6,20}
		};
		std::vector<int> kpt_color_ids = {
			0,0,0,0,0,
			3,4,3,4,3,4,
			5,6,5,6,5,6,
			2,2,2,2,2,2
		};
		std::vector<int> bone_color_ids = {
			0,0,0,0,0,3,3,4,4,
			2,5,6,5,5,6,6,
			2,3,4
		};
		// skels 
		for (int i = 0; i < 4; i++)
		{
			std::vector<Eigen::Vector3f> skels = solvers[i]->getRegressedSkel_host();
			std::vector<Eigen::Vector3f> balls;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
			GetBallsAndSticks(skels, bones, balls, sticks);
			int jointnum = skels.size();
			std::vector<float> ball_sizes;
			ball_sizes.resize(jointnum, 0.015);
			std::vector<float> stick_sizes;
			stick_sizes.resize(sticks.size(), 0.008);
			std::vector<Eigen::Vector3f> ball_colors(jointnum);
			std::vector<Eigen::Vector3f> stick_colors(sticks.size());
			for (int i = 0; i < jointnum; i++)
			{
				ball_colors[i] = CM3[kpt_color_ids[i]];
			}
			for (int i = 0; i < sticks.size(); i++)
			{
				stick_colors[i] = CM3[bone_color_ids[i]];
			}

			BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
				balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
			m_renderer.skels.push_back(p_skel);
		}
#endif 

		std::vector<int> render_views = { 0,1,2,3,4,5,6,7,8,9};

		std::vector<cv::Mat> rawImgs = frame.get_imgs_undist();

		std::vector<cv::Mat> rawImgsSelect;
		for (int k = 0; k < render_views.size(); k++) rawImgsSelect.push_back(rawImgs[render_views[k]]);

		std::vector<cv::Mat> all_renders(render_views.size());
		for (int k = 0; k < render_views.size(); k++)
		{
			int camid = render_views[k];
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			cv::Mat img = m_renderer.GetImageOffscreen();
			all_renders[k] = img;
		}
		//m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
		//m_renderer.createSceneDetailed(conf_projectFolder, 1);
		//m_renderer.createSceneHalf(conf_projectFolder, 1.08);

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));
		Eigen::Vector3f pos3(0.0, -0.0, 2.5);
		Eigen::Vector3f up3(0.0, 0.1, -0.0);
		Eigen::Vector3f center3(0.0, -0.0, 0.0);
		m_renderer.s_camViewer.SetExtrinsic(pos3, up3, center3);
		cv::Mat img = m_renderer.GetImageOffscreen();
		all_renders.push_back(img);
		rawImgsSelect.push_back(img);

		//Eigen::Vector3f pos4(-2.2866, -1.61577, 2.71787);
		//Eigen::Vector3f up4(0.52083, 0.378857, 0.764986);
		//Eigen::Vector3f center4(0.241644, -0.127209, 0.250072);
		//m_renderer.s_camViewer.SetExtrinsic(pos4, up4, center4);
		////img = m_renderer.GetImageOffscreen();
		//m_renderer.Draw(); 
		//img = m_renderer.GetImage(); 
		std::stringstream ss_out;
		ss_out << frame.m_result_folder << "/render_last/render" << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
		cv::imwrite(ss_out.str(), img);
		img = my_resize(img, 0.5); 

		//cv::Mat packed_render;
		//packImgBlock(all_renders, packed_render);
		//cv::Mat pack_raw;
		//packImgBlock(rawImgsSelect, pack_raw);

		//cv::Mat blend;
		//overlay_render_on_raw_gpu(packed_render, pack_raw, blend);

		//cv::Mat small_img = my_resize(blend, 1);
		//std::stringstream all_render_file;
		//all_render_file << frame.m_result_folder << "/render_all/" << std::setw(6) << std::setfill('0')
		//	<< frameid << ".png";
		//cv::imwrite(all_render_file.str(), small_img);
		
		GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		while (!glfwWindowShouldClose(windowPtr))
		{
			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			m_renderer.Draw();

			glfwSwapBuffers(windowPtr);
			glfwPollEvents();
		};
		return 0;
	}

	return 0;
}
