#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>

#include "../utils/colorterminal.h" 
#include "../utils/obj_reader.h"
#include "../utils/timer_util.h"
#include "../articulation/pigmodel.h"
#include "../articulation/pigsolver.h"
#include "../associate/framedata.h"
#include "../utils/model.h"
#include "../utils/timer.hpp" 
#include "../render/renderer.h"
#include "../render/render_object.h" 
#include "../render/render_utils.h"
#include "../utils/camera.h"
#include "../utils/math_utils.h" 
#include "../utils/image_utils.h" 

#include "main.h"

using std::vector;

void run_pose_render()
{
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");

	FrameData frame;
	frame.configByJson(conf_projectFolder + "/associate/config.json");
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(0);
	frame.fetchData();
	auto cams = frame.get_cameras();
	
	// render config 
	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
	std::cout << K << std::endl;

	Eigen::Vector3f up; up << 0, 0, -1;
	Eigen::Vector3f pos; pos << -1, 1.5, -0.8;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();

	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.SetBackgroundColor(Eigen::Vector4f::Zero()); 
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	//m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
	m_renderer.s_camViewer.SetExtrinsic(cams[0].R.cast<float>(), cams[1].T.cast<float>());

	// init element obj
	const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj");
	const ObjData cameraObj(conf_projectFolder + "/render/data/obj_model/camera.obj");

	//RenderObjectTexture* chess_floor = new RenderObjectTexture();
	//chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	//chess_floor->SetFaces(squareObj.faces, false);
	//chess_floor->SetVertices(squareObj.vertices);
	//chess_floor->SetTexcoords(squareObj.texcoords);
	//chess_floor->SetTransform({ 0, 0, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	//m_renderer.texObjs.push_back(chess_floor);

	frame.result_folder = "G:/pig_results/";
	frame.is_smth = false; 
	int start = frame.get_start_id(); 
	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "processsing frame " << frameid << std::endl; 
		frame.set_frame_id(frameid); 
		frame.fetchData();

		frame.load_clusters(); 
		frame.solve_parametric_model();
		frame.save_parametric_data(); 

		auto p_solvers = frame.mp_bodysolver;
		for (int k = 0; k < m_renderer.colorObjs.size(); k++)
		{
			delete m_renderer.colorObjs[k];
		}
		m_renderer.colorObjs.clear();
		for (int k = 0; k < 4; k++)
		{
			Eigen::MatrixXf v_f = p_solvers[k]->GetVertices().cast<float>();
			Eigen::MatrixXu f_u = p_solvers[k]->GetFacesVert();

			RenderObjectColor* animal_model = new RenderObjectColor();
			animal_model->SetVertices(v_f);
			animal_model->SetFaces(f_u);
			animal_model->SetColor(CM[k]);

			m_renderer.colorObjs.push_back(animal_model);
		}

		std::vector<cv::Mat> all_renders; 
		for (int view = 0; view < cams.size(); view++)
		{
			m_renderer.s_camViewer.SetExtrinsic(cams[view].R.cast<float>(),
				cams[view].T.cast<float>()); 
			m_renderer.Draw();
			cv::Mat img = m_renderer.GetImage();
			all_renders.push_back(img); 
		}
		cv::Mat output; 
		packImgBlock(all_renders, output); 
		std::stringstream ss_out;
		ss_out << frame.result_folder + "/render_all/" <<
			std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss_out.str(), output);

		//GLFWwindow* windowPtr = m_renderer.s_windowPtr;

		//while (!glfwWindowShouldClose(windowPtr))
		//{
		//	m_renderer.Draw();
		//	glfwSwapBuffers(windowPtr);
		//	glfwPollEvents();
		//};

		//break;
	}

	return;
}