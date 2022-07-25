#include "main.h"
#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>
#include <filesystem> 

#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "framesolver.h"
#include "../utils/mesh.h"
#include <vector_functions.hpp>
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"
#include "../render/render_utils.h"

int nm_monocolor_44_clips()
{
	show_gpu_param();
	std::string conf_projectFolder = "H:/MAMMAL_core/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_blend");

	FrameSolver frame;
	//frame.configByJson(conf_projectFolder + "/configs/config_20190704_foreval-seq.json");
	frame.configByJson(conf_projectFolder + "/configs/config_seq2.json");


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
	frame.is_smth = true;
	int start = frame.get_start_id();
	int pignum = frame.m_pignum;

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

	std::vector<std::vector<int> > configs = {
		// start, end, pig name, view id
		{0,	875,	0,	9},
	{1190,	1400,	0,	2},
	{2620,	2800,	0,	2},
	{3800,	4329,	0,	8},
	{5170,	5712,	0,	7},
	{5960,	6600,	0,	1} ,
	{6910,	7128,	0,	9},
	{9330,	10000,	0,	1},
	{0,	80,	1,	8},
	{350,	841,	1,	2},
	{1035,	1580,	1,	4},
	{2540,	2900,	1,	7},
	{3648,	4679,	1,	0},
	{6315,	6600,	1,	6},
	{7800,	8000,	1,	1},
	{8270,	8490,	1,	9},
	{9000,	9139,	1,	6},
	{9240,	9340,	1,	6},
	{0	,120,	2,	1},
	{290,	1500,	2,	2},
	{2330,	3200,	2,	3},
	{4925,	5117,	2,	9},
	{5137,	5920,	2,	1},
	{6600,	6700,	2,	7},
	{7021,	7121,	2,	2},
	{9000,	9139,	2,	0},
	{9210,	9270,	2,	5},
	{0,	120,	3,	2},
	{795,	1060,	3,	4},
	{1130,	1400,	3,	6},
	{1857,	2100,	3,	0},
	{2330,	2480,	3,	8},
	{2730,	3047,	3,	1},
	{3554,	4329,	3,	0},
	{5960,	6400,	3,	1},
	{6600,	6820,	3,	9},
	{9150,	9500,	3,	6}
	};
	for (int clip_id = 0; clip_id < configs.size(); clip_id++)
	{
		int start = configs[clip_id][0]; 
		int end = configs[clip_id][1]; 
		int pig_name = configs[clip_id][2]; 
		int view_id = configs[clip_id][3]; 
		int true_clip_id = clip_id + 7; 
		std::cout << "clip : " << true_clip_id << std::endl; 
		std::string output_folder = "G:/results/seq_noon/render_mono_20220724/clip_" + std::to_string(true_clip_id); 
		if (!std::filesystem::exists(output_folder))
			std::filesystem::create_directories(output_folder); 
		for (int frameid = start; frameid < end; frameid++)
		{
			std::cout << "===========processing frame " << frameid << "===============" << std::endl;
			frame.set_frame_id(frameid);
			frame.fetchData();
			frame.load_clusters();
			frame.read_parametric_data();
			auto solvers = frame.mp_bodysolverdevice;
			m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));
			for (int pid = 0; pid < pignum; pid++)
			{
				int current_pig_name = frame.m_pig_names[pid];
				if (current_pig_name != pig_name) continue;
				m_renderer.clearAllObjs();
				RenderObjectColor* p_model = new RenderObjectColor();
				solvers[pid]->UpdateNormalFinal();
				p_model->SetVertices(solvers[pid]->GetVertices());
				p_model->SetNormal(solvers[pid]->GetNormals());
				p_model->SetFaces(solvers[pid]->GetFacesVert());
				p_model->SetColor(m_CM[1]);
				p_model->isFill = true;
				m_renderer.colorObjs.push_back(p_model);

				std::vector<int> render_views = {};
				render_views.push_back(view_id);

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
				cv::Mat packed_render;
				packImgBlock(all_renders, packed_render);

				cv::Mat pack_raw;
				packImgBlock(rawImgsSelect, pack_raw);
				cv::Mat blend;
				overlay_render_on_raw_gpu(packed_render, pack_raw, blend);
				cv::Mat small_img = my_resize(blend, 1);
				std::stringstream all_render_file;
				all_render_file << output_folder << "/" << std::setfill('0') << std::setw(6) << frameid - start << ".png";
				cv::imwrite(all_render_file.str(), small_img);
			}
		}
	}

	return 0;
}


// 2022.03.30: for nm_video5 (behaviors) 
int nm_monocolor_singlebody()
{
	show_gpu_param();
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_blend"); 

	FrameSolver frame;
	frame.configByJson(conf_projectFolder + "/configs/config_20190704_foreval-seq.json");
	//frame.configByJson(conf_projectFolder + "/configs/config_seq2.json");


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
	frame.is_smth = true;
	int start = frame.get_start_id();
	int pignum = frame.m_pignum;

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
	//for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	/// for rendering seq_noon demo picture. 
	//std::vector<int> frame_to_rend = { 5312, 5103, 3017, 4006, 1202, 9102, 1456};
	//std::vector<int> id_to_rend = { 0,0,0,1,1,2,2};
	//std::vector<int> view_to_rend = { 6,7,7,7,8,6,7}; 
	//std::vector<int> names = { 4,5,9,10,13,35,39};
	/// for rendering seq_morning demo picture. 
	std::vector<int> frame_to_rend = { 1783 };
	std::vector<int> id_to_rend = { 0 };
	std::vector<int> view_to_rend = { 7 };
	std::vector<int> names = { 18 };
	for(int t = 0; t < frame_to_rend.size(); t++)
	{
		int frameid = frame_to_rend[t]; 
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		frame.load_clusters();
		frame.read_parametric_data();
		auto solvers = frame.mp_bodysolverdevice;
		m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));
		for (int pid = 0; pid < pignum; pid++)
		{
			if (pid != id_to_rend[t]) continue; 
			m_renderer.clearAllObjs(); 
			RenderObjectColor* p_model = new RenderObjectColor();
			solvers[pid]->UpdateNormalFinal();
			p_model->SetVertices(solvers[pid]->GetVertices());
			p_model->SetNormal(solvers[pid]->GetNormals());
			p_model->SetFaces(solvers[pid]->GetFacesVert());
			p_model->SetColor(m_CM[1]);
			p_model->isFill = false; 
			m_renderer.colorObjs.push_back(p_model);

			std::vector<int> render_views = {};
			render_views.push_back(view_to_rend[t]); 

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
			cv::Mat packed_render;
			packImgBlock(all_renders, packed_render);

			cv::Mat pack_raw;
			packImgBlock(rawImgsSelect, pack_raw);
			cv::Mat blend;
			overlay_render_on_raw_gpu(packed_render, pack_raw, blend);
			cv::Mat small_img = my_resize(blend, 1);
			std::stringstream all_render_file;
			all_render_file << "G:/results/seq_noon/render_mono_sup/" << names[t] << ".png"; 
			cv::imwrite(all_render_file.str(), small_img);

			//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
			//while (!glfwWindowShouldClose(windowPtr))
			//{
			//	m_renderer.Draw();
			//	glfwSwapBuffers(windowPtr);
			//	glfwPollEvents();
			//};
			//return 0; 
		}
	}

	return 0;
}

// 2022.04.05: for nm_video5 (behaviors) 
int nm_video5_freeview()
{
	show_gpu_param();
	std::string conf_projectFolder = "H:/MAMMAL_core/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_blend");

	Camera cam = Camera::getDefaultCameraUndist(); 

	// init renderer
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(true);
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

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

	std::vector<int> to_be_rend = { 4,5,9,10,13,18,35,39 }; 
	for (int k = 0; k < to_be_rend.size(); k++)
	{
		int index = to_be_rend[k]; 
		std::string obj_path = "D:/Projects/animal_social/tmp6-3/objs/" + std::to_string(index-1) + ".obj";
		PigSolverDevice solver(conf_projectFolder+ "/articulation/artist_config_sym.json");

		Mesh pigmesh(obj_path);
		solver.fitPoseToVSameTopo(pigmesh.vertices_vec);

		m_renderer.clearAllObjs();
		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));

		std::vector<Eigen::Vector3f> skels = solver.getRegressedSkel_host();
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
			ball_colors[i] = CM[kpt_color_ids[i]];
		}
		for (int i = 0; i < sticks.size(); i++)
		{
			stick_colors[i] = CM[bone_color_ids[i]];
		}

		BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
			balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
		p_skel->isMultiLight = true;
		m_renderer.skels.push_back(p_skel);

		RenderObjectColor* p_model = new RenderObjectColor();
		solver.UpdateNormalFinal();
		p_model->SetVertices(solver.GetVertices());
		p_model->SetNormal(solver.GetNormals());
		p_model->SetFaces(solver.GetFacesVert());
		p_model->SetColor(Eigen::Vector3f(0.4, 0.4, 0.4));
		p_model->isFill = false;
		p_model->isMultiLight = true;
		m_renderer.colorObjs.push_back(p_model);

		m_renderer.createPlane(conf_projectFolder);

		Eigen::Vector3f pos4(-0.126062, 1.22237, 0.572144);
		Eigen::Vector3f up4(0.0136052, -0.318185, 0.947931);
		Eigen::Vector3f center4(-0.0621624, -0.180543, 0.100319);
		m_renderer.s_camViewer.SetExtrinsic(pos4, up4, center4);
		cv::Mat output = m_renderer.GetImageOffscreen();

		for (int timeindex = 0; timeindex < 76; timeindex++)
		{
			for (int k = 0; k < m_renderer.meshObjs.size(); k++)
			{
				m_renderer.meshObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, timeindex / 75.f * M_PI * 2), 1);
			}
			for (int k = 0; k < m_renderer.skels.size(); k++)
			{
				m_renderer.skels[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, timeindex / 75.f * M_PI * 2), 1);
			}
			for (int k = 0; k < m_renderer.colorObjs.size(); k++)
			{
				m_renderer.colorObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, timeindex / 75.f * M_PI * 2), 1);
			}
			m_renderer.Draw();
			glfwPollEvents();
			cv::Mat img = m_renderer.GetImageOffscreen();
			std::stringstream ss_out;
			//ss_out.str("");
			ss_out << "H:/results/seq_noon/render_free2/" << index << "/" << timeindex << ".png";
			cv::imwrite(ss_out.str(), img);
		}
	}

	//TimerUtil::Timer<std::chrono::milliseconds> timer; 
	//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	//while (!glfwWindowShouldClose(windowPtr))
	//{
	//	float t = timer.Elapsed() / 1000.f; 
	//	for (int k = 0; k < m_renderer.meshObjs.size(); k++)
	//	{
	//		m_renderer.meshObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, t * M_PI / 10), 1); 
	//	}
	//	for (int k = 0; k < m_renderer.colorObjs.size(); k++)
	//	{
	//		m_renderer.colorObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, t * M_PI / 10), 1);
	//	}
	//	for (int k = 0; k < m_renderer.skels.size(); k++)
	//	{
	//		m_renderer.skels[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, t * M_PI / 10), 1);
	//	}
	//	
	//	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//	m_renderer.Draw();
	//	glfwSwapBuffers(windowPtr);
	//	glfwPollEvents();
	//};
	return 0;	

}

// 2022.04.05: for nm_video5 (behaviors) 
int nm_fig_skel_rend_demo()
{
	show_gpu_param();
	std::string conf_projectFolder = "H:/MAMMAL_core/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_blend");

	Camera cam = Camera::getDefaultCameraUndist();

	// init renderer
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(false);
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

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

	FrameSolver frame;
	frame.configByJson(conf_projectFolder + "/configs/config_20190704_fordemo.json");
	
	frame.set_frame_id(7888); 
	frame.init_parametric_solver(); 
	frame.fetchData(); 

	
	frame.load_clusters(); 
	frame.read_parametric_data(); 
	
	frame.resetSolverStateMarker();
	frame.DARKOV_Step1_setsource();
	frame.DirectTriangulation(); 
	std::vector<Eigen::Vector3f> skels = frame.get_skels3d()[1];

	//auto solvers = frame.mp_bodysolverdevice;
	//auto solver = solvers[1];
	//std::vector<Eigen::Vector3f> skels = solver->getRegressedSkel_host();

	
	m_renderer.clearAllObjs();

	Eigen::Vector3f root_pos = skels[20];
	root_pos[2] = 0; 
	for (int k = 0; k < skels.size(); k++) {
		if (skels[k].norm() == 0) continue; 
		skels[k] -= root_pos;
	}

	std::vector<Eigen::Vector3f> balls;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	GetBallsAndSticks(skels, bones, balls, sticks);
	int jointnum = skels.size();
	std::vector<float> ball_sizes;
	ball_sizes.resize(jointnum, 0.015);
	std::vector<float> stick_sizes;
	stick_sizes.resize(sticks.size(), 0.009);
	//std::vector<Eigen::Vector3f> ball_colors(jointnum);
	//std::vector<Eigen::Vector3f> stick_colors(sticks.size());
	std::vector<Eigen::Vector3f> ball_colors; 
	std::vector<Eigen::Vector3f> stick_colors; 
	for (int i = 0; i < jointnum; i++)
	{
		if (skels[i].norm() == 0) continue; 
		ball_colors.push_back(CM[kpt_color_ids[i]]);
	}
	for (int i = 0; i < bones.size(); i++)
	{
		if (skels[bones[i][0]].norm() == 0) continue; 
		if (skels[bones[i][1]].norm() == 0) continue; 
		stick_colors.push_back(CM[bone_color_ids[i]]);
	}

	BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
		balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
	p_skel->isMultiLight = true;
	m_renderer.skels.push_back(p_skel);

	//RenderObjectColor* p_model = new RenderObjectColor();
	//std::vector<Eigen::Vector3f> vertices = solver->GetVertices(); 
	//for (int k = 0; k < vertices.size(); k++) vertices[k] -= root_pos; 
	//p_model->SetVertices(vertices);
	//p_model->SetNormal(solver->GetNormals());
	//p_model->SetFaces(solver->GetFacesVert());
	//p_model->SetColor(Eigen::Vector3f(0.8, 0.8, 0.8));
	//p_model->isFill = false;
	//p_model->isMultiLight = true;
	//m_renderer.colorObjs.push_back(p_model);

	m_renderer.createPlane(conf_projectFolder);

	Eigen::Vector3f pos4(-0.11472 , 1.82237, 0.655653);
	Eigen::Vector3f up4(0.00476978, - 0.244485 ,  0.969641);
	Eigen::Vector3f center4(-0.0621624, - 0.180543,   0.150319);
	m_renderer.s_camViewer.SetExtrinsic(pos4, up4, center4);
	cv::Mat output = m_renderer.GetImageOffscreen();
	m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1)); 
	for (int timeindex = 0; timeindex < 75; timeindex++)
	{
		for (int k = 0; k < m_renderer.meshObjs.size(); k++)
		{
			m_renderer.meshObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, timeindex / 75.f * M_PI * 2), 1);
		}
		for (int k = 0; k < m_renderer.skels.size(); k++)
		{
			m_renderer.skels[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, timeindex / 75.f * M_PI * 2), 1);
		}
		for (int k = 0; k < m_renderer.colorObjs.size(); k++)
		{
			m_renderer.colorObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, timeindex / 75.f * M_PI * 2), 1);
		}
		m_renderer.Draw();
		glfwPollEvents();
		cv::Mat img = m_renderer.GetImageOffscreen();
		std::stringstream ss_out;
		//ss_out.str("");
		ss_out << "H:/results/paper_teaser/demo_tri/" << timeindex << ".png";
		cv::imwrite(ss_out.str(), img);
	}	
	

	//TimerUtil::Timer<std::chrono::milliseconds> timer; 
	//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	//while (!glfwWindowShouldClose(windowPtr))
	//{
	//	float t = timer.Elapsed() / 1000.f; 
	//	for (int k = 0; k < m_renderer.meshObjs.size(); k++)
	//	{
	//		m_renderer.meshObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, t * M_PI / 10), 1); 
	//	}
	//	for (int k = 0; k < m_renderer.colorObjs.size(); k++)
	//	{
	//		m_renderer.colorObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, t * M_PI / 10), 1);
	//	}
	//	for (int k = 0; k < m_renderer.skels.size(); k++)
	//	{
	//		m_renderer.skels[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, t * M_PI / 10), 1);
	//	}
	//	
	//	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//	m_renderer.Draw();
	//	glfwSwapBuffers(windowPtr);
	//	glfwPollEvents();
	//};
	return 0;

}

// 2022.03.30: for nm_video2 (comparison) 
int nm_skelrender_for_comparison()
{
	//show_gpu_param();
	std::string conf_projectFolder = "H:/MAMMAL_core/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");

	FrameSolver frame;
	frame.configByJson(conf_projectFolder + "/configs/config_BamaPigEval3D_main.json");

	frame.set_frame_id(0);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];

	std::string proj_folder = "H:/results/paper_teaser/skel_projs/fit10view/"; 
	std::string rend_folder = "H:/results/paper_teaser/skel_renders/fit10view/";

	//std::string skel_folder = "E:/results/paper_teaser/0704_eval2-5views/joints_23/"; 
	//std::string skel_folder = "E:/results/paper_teaser/0704_eval2-(057)/joints_23/"; 
	std::string skel_folder = "H:/results/BamaPigEval3D_main/joints_23/"; 
	//std::string skel_folder = "D:/results/paper_teaser/0704_eval_tri/skels/";
	//std::string skel_folder = "E:/results/paper_teaser/0704_eval2-5views/skels/"; 
	//std::string skel_folder = "E:/results/paper_teaser/0704_eval2-(057)/skels/"; 
	//std::vector<int> nameids = { 2,0,3,1 };
	std::vector<int> nameids = { 0,2,3,1 };
	frame.m_pig_names = nameids; 

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
	int start = frame.get_start_id();
	int pignum = frame.m_pignum;

	std::vector<Eigen::Vector2i> bones = {
	{0,1}, {0,2}, {1,2}, {1,3}, {2,4},
	 {5,7}, {7,9}, {6,8}, {8,10},
	{20,18},
	{18,11}, {18,12}, {11,13}, {13,15}, {12,14}, {14,16},
	{0,20},{5,20},{6,20}
	};
	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	
	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "===========processing frame " << frameid << "===============" << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();

		m_renderer.clearAllObjs();
		m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

		frame.read_skels(skel_folder, frameid); 

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

		/// 2022.03.30 proj for nm_video2
		cv::Mat reproj = frame.visualizeProj(0);
		std::stringstream ss_proj;
		ss_proj << proj_folder << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::Mat small_reproj = my_resize(reproj, 0.5); 
		cv::imwrite(ss_proj.str(), small_reproj);

		auto skels_data = frame.get_skels3d(); 
		for (int pid = 0; pid < 4; pid++)
		{
			std::vector<Eigen::Vector3f> skels = skels_data[pid];
			std::vector<Eigen::Vector3f> balls;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;

			GetBallsAndSticks(skels, bones, balls, sticks);
			int jointnum = skels.size();
			std::vector<float> ball_sizes;
			ball_sizes.resize(jointnum, 0.025);
			std::vector<float> stick_sizes;
			stick_sizes.resize(sticks.size(), 0.012);
			std::vector<Eigen::Vector3f> ball_colors(jointnum);
			std::vector<Eigen::Vector3f> stick_colors(sticks.size());
			for (int i = 0; i < jointnum; i++)
			{
				ball_colors[i] = m_CM[nameids[pid]] ;
			}
			for (int i = 0; i < sticks.size(); i++)
			{
				stick_colors[i] = m_CM[nameids[pid]];
			}

			BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
				balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
			p_skel->isMultiLight = true;
			m_renderer.skels.push_back(p_skel);
		}
		//m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
		//m_renderer.createSceneDetailed(conf_projectFolder, 1);
		//m_renderer.createSceneHalf(conf_projectFolder, 1.08);
		m_renderer.createPlane(conf_projectFolder, 1.08);

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));
		Eigen::Vector3f pos4(-2.91765 , 1.31419 , 1.87457);
		Eigen::Vector3f up4(0.384713, - 0.170107 , 0.907226);
		Eigen::Vector3f center4(0.241644, - 0.127209,  0.250072);
		m_renderer.s_camViewer.SetExtrinsic(pos4, up4, center4); 
		cv::Mat img = m_renderer.GetImageOffscreen();
		std::stringstream ss_out; 
		ss_out << rend_folder << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_out.str(), img); 

		if (frameid == 1510 || frameid == 1274)
		{
			for (int timeindex = 0; timeindex < 151; timeindex++)
			{
				for (int k = 0; k < m_renderer.meshObjs.size(); k++)
				{
					m_renderer.meshObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, timeindex / 150.f * M_PI * 2), 1);
				}
				for (int k = 0; k < m_renderer.skels.size(); k++)
				{
					m_renderer.skels[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, timeindex / 150.f * M_PI * 2), 1);
				}
				m_renderer.Draw();
				glfwPollEvents();
				cv::Mat img = m_renderer.GetImageOffscreen();
				std::stringstream ss_out;
				//ss_out.str("");
				ss_out << rend_folder << "/freeview/" << std::setw(6) << std::setfill('0') << frameid << "_" << timeindex << ".png";
				cv::imwrite(ss_out.str(), img);
			}
		}

		//return 0; 

		//TimerUtil::Timer<std::chrono::milliseconds> timer; 
		//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		//while (!glfwWindowShouldClose(windowPtr))
		//{
		//	float t = timer.Elapsed() / 1000.f; 
		//	for (int k = 0; k < m_renderer.meshObjs.size(); k++)
		//	{
		//		m_renderer.meshObjs[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, t * M_PI / 10), 1); 
		//	}
		//	for (int k = 0; k < m_renderer.skels.size(); k++)
		//	{
		//		m_renderer.skels[k]->SetTransform(Eigen::Vector3f::Zero(), Eigen::Vector3f(0, 0, t * M_PI / 10), 1);
		//	}
		//	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		//	m_renderer.Draw();
		//	glfwSwapBuffers(windowPtr);
		//	glfwPollEvents();
		//};
		//return 0;
	}

	return 0;
}


// 2021.10.6: 
// This function is used to generate 3D trajectory rendering for NM paper. 
// 2022.07.24
// Run trajectory to create trajectory images as Fig.1e
void run_trajectory2()
{
	std::string conf_projectFolder = "H:/MAMMAL_core/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_blend");
	std::vector<Eigen::Vector3f> CM2 = getColorMapEigenF("anliang_render");
	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	// pose2: face towards drinking area
	//Eigen::Vector3f pos1; pos1 << 1.05564, -1.11705, 1.11743;
	//Eigen::Vector3f up1; up1 << -0.256466, 0.308172, 0.916109;
	//Eigen::Vector3f center1; center1 << 0.113146, -0.0304874, 0.488063;
	Eigen::Vector3f pos1(2.26127, -0.940305, 1.67403);
	Eigen::Vector3f up1(-0.427835, 0.158784, 0.8898);
	Eigen::Vector3f center1(0.256568, -0.164695, 0.561785);
	Eigen::Vector3f pos4(-0.988364, -2.79656, 1.91186);
	Eigen::Vector3f up4(0.226098, 0.500471, 0.835709);
	Eigen::Vector3f center4(0.116397, -0.369781, 0.14214);
	// init renderer 
	Renderer::s_Init(false);

	Renderer m_renderer(conf_projectFolder + "/render/shader/");

	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos4, up4, center4);

	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
	SkelTopology topo = getSkelTopoByType("UNIV");

	int window = 150;

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

	cv::VideoWriter writer("G:/pig_result_nm/trajectory0.avi", cv::VideoWriter::fourcc('m', 'p', 'e', 'g'), 25.0, cv::Size(1920, 1080));
	if (!writer.isOpened())
	{
		std::cout << "not open" << std::endl;
		return;
	}

	std::vector<std::deque<std::vector<Eigen::Vector3f> > > joints_queues;
	joints_queues.resize(4);

	FrameSolver frame;
	std::string configfile = "configs/config_BamaPig3D_main.json";
	frame.configByJson(conf_projectFolder + configfile);
	frame.init_parametric_solver();
	auto solvers = frame.get_solvers();
	int start = 0;
	int num = 300;
	frame.is_smth = true;
	std::vector<int> ids_draw = { 0,1,2,3 };
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
					p_model->SetColor(Eigen::Vector3f(0.8, 0.8, 0.8));
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
		//m_renderer.createSceneHalf2(conf_projectFolder, 1.1);
		m_renderer.createPlane(conf_projectFolder); 

		cv::Mat img = m_renderer.GetImageOffscreen();
		writer.write(img);

		//std::stringstream output_ss; 
		//output_ss << "G:/pig_middle_data/teaser/video/trajectory_" << std::setw(6) << std::setfill('0') << frameid << ".png"; 
		//cv::imwrite(output_ss.str(), img);
	/*	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		while (!glfwWindowShouldClose(windowPtr))
		{
			m_renderer.Draw();
			glfwSwapBuffers(windowPtr);
			glfwPollEvents();
		};
		exit(-1);*/
	}
}


// 2022.07.24
// Run trajectory to create trajectory images as Fig.1e
void nm_trajectory()
{
	std::string conf_projectFolder = "H:/MAMMAL_core/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_blend");
	std::vector<Eigen::Vector3f> CM2 = getColorMapEigenF("anliang_paper");
	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	Eigen::Vector3f pos4(-0.988364, -2.79656, 1.91186);
	Eigen::Vector3f up4(0.226098, 0.500471, 0.835709);
	Eigen::Vector3f center4(0.116397, -0.369781, 0.14214);
	// init renderer 
	Renderer::s_Init(true);

	Renderer m_renderer(conf_projectFolder + "/render/shader/");

	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos4, up4, center4);

	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
	SkelTopology topo = getSkelTopoByType("UNIV");

	int window = 200;

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

	std::vector<std::deque<std::vector<Eigen::Vector3f> > > joints_queues;
	joints_queues.resize(4);

	FrameSolver frame;
	std::string configfile = "configs/config_BamaPig3D_main.json";
	frame.configByJson(conf_projectFolder + configfile);
	frame.init_parametric_solver();
	auto solvers = frame.get_solvers();
	int start = 0;
	int num = window;
	frame.is_smth = true;

	for (int frameid = start; frameid < start + num; frameid++)
	{
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
	}

	for (int pig_to_draw = 0; pig_to_draw < 4; pig_to_draw++)
	{
		m_renderer.clearAllObjs();

		for (int index = 0; index < num; index++)
		{
			if (index == 0 || index == num - 1)
			{
				std::cout << "index: " << index << std::endl;
				for (int pid = 0; pid < frame.m_pignum; pid++)
				{
					int pigname = frame.m_pig_names[pid];
					if (pigname != pig_to_draw) continue;
					int ratio_index = window - num + index;
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

					frame.set_frame_id(index + start);
					frame.read_parametric_data();
					solvers = frame.get_solvers();
					RenderObjectColor* p_model = new RenderObjectColor();
					solvers[pid]->UpdateNormalFinal();
					p_model->SetVertices(solvers[pid]->GetVertices());
					p_model->SetNormal(solvers[pid]->GetNormals());
					p_model->SetFaces(solvers[pid]->GetFacesVert());
					p_model->SetColor(CM2[pigname]);
					if (index == 0)
						p_model->isFill = false;
					else p_model->isFill = true;
					p_model->isMultiLight = true; 
					m_renderer.colorObjs.push_back(p_model);
				}
			}
			else
			{
				for (int pid = 0; pid < frame.m_pignum; pid++)
				{
					int pigname = frame.m_pig_names[pid];
					if (pigname != pig_to_draw) continue;
					float ratio = (2 - 1 * (index / float(window)));

					std::vector<int> joints_to_rend = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20};

					std::vector<Eigen::Vector3f> skels;
					for(int k = 0; k < joints_to_rend.size(); k++)
						skels.push_back(joints_queues[pid][index][joints_to_rend[k]]);
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
						int joint_id = joints_to_rend[k]; 
						sticks[k].first = joints_queues[pid][index][joint_id];
						sticks[k].second = joints_queues[pid][index + 1][joint_id];
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

		m_renderer.createSceneHalf(conf_projectFolder, 1.08);
		//m_renderer.createPlane(conf_projectFolder);

		cv::Mat img = m_renderer.GetImageOffscreen();

		std::stringstream ss;
		ss << "G:/pig_result_nm/pig_" << pig_to_draw << ".png";
		cv::imwrite(ss.str(), img);
	}

	
	std::vector<int> render_indices;
	render_indices.push_back(start); 
	render_indices.push_back(start + num); 
	for (int k = 0; k < render_indices.size(); k++)
	{
		m_renderer.clearAllObjs();
		int index = render_indices[k];
		for (int pid = 0; pid < 4; pid++)
		{
			int pigname = frame.m_pig_names[pid]; 
			frame.set_frame_id(index);
			frame.read_parametric_data();
			solvers = frame.get_solvers();
			RenderObjectColor* p_model = new RenderObjectColor();
			solvers[pid]->UpdateNormalFinal();
			p_model->SetVertices(solvers[pid]->GetVertices());
			p_model->SetNormal(solvers[pid]->GetNormals());
			p_model->SetFaces(solvers[pid]->GetFacesVert());
			p_model->SetColor(CM2[pigname]);
			p_model->isMultiLight = true;
			m_renderer.colorObjs.push_back(p_model);
		}

		m_renderer.createPlane(conf_projectFolder);
		cv::Mat img = m_renderer.GetImageOffscreen();
		std::stringstream ss;
		ss << "G:/pig_result_nm/pig_all_" << index << ".png";
		cv::imwrite(ss.str(), img);
	}
}