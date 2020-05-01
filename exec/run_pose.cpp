#include "main.h"
#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 

#include "../utils/colorterminal.h" 
#include "../utils/obj_reader.h"
#include "../utils/timer_util.h"
#include "../render/render_object.h"
#include "../render/render_utils.h"
#include "../render/renderer.h"

#include "../smal/pigmodel.h"
#include "../smal/pigsolver.h"

#include "../associate/framedata.h"
#include "../utils/volume.h"

#define RUN_SEQ
#define VIS 
#define DEBUG_VIS
//#define LOAD_STATE
#define SHAPE_SOLVER
//#define VOLUME

using std::vector;

int run_pose()
{
	std::string pig_config = "D:/Projects/animal_calib/smal/smal2_config.json";
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	FrameData frame;
	frame.configByJson(conf_projectFolder + "/associate/config.json");
	int startid = frame.get_start_id();

#ifdef VIS
	//// rendering pipeline. 
	auto CM = getColorMapEigen("anliang_rgb");
	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
	Eigen::Vector3f up; up << 0, 0, -1;
	Eigen::Vector3f pos; pos << -1, 1.5, -0.8;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	//m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData cubeObj(conf_projectFolder + "/render/data/obj_model/cube.obj");
	const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj");
	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, true);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ 0.28f, 0.2f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

#endif 
	int framenum = frame.get_frame_num();
	PigSolver shapesolver(pig_config);
	int m_pid = 0; // pig identity to solve now. 

	for (int frameid = 0; frameid < 3; frameid++)
	{
		std::cout << "processing frame " << frameid << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		//frame.view_dependent_clean();
		frame.matching_by_tracking();
		//frame.load_labeled_data();
		frame.solve_parametric_model();
		auto m_matched = frame.get_matched();
		cv::Mat det_img = frame.visualizeIdentity2D();
		std::stringstream ss1;
		ss1 << "E:/debug_pig3/assoc/" << std::setw(6) << std::setfill('0')
			<< frameid << ".jpg";
		cv::imwrite(ss1.str(), det_img);

		auto m_rois = frame.getROI(m_pid);
		shapesolver.setCameras(frame.get_cameras());
		shapesolver.normalizeCamera();
		shapesolver.setId(m_pid);
		shapesolver.setSource(m_matched[m_pid]);
		shapesolver.normalizeSource();
		shapesolver.InitNodeAndWarpField();
		shapesolver.LoadWarpField();
		shapesolver.UpdateVertices();
		shapesolver.globalAlign();
		shapesolver.optimizePose(20, 0.001);

		shapesolver.mp_renderer = &m_renderer; 
		shapesolver.m_rois = m_rois;

		shapesolver.optimizePoseSilhouette(3);

		m_renderer.colorObjs.clear(); 
		RenderObjectColor* pig_render = new RenderObjectColor();
		Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces
			= shapesolver.GetFacesVert();
		Eigen::MatrixXf vs = shapesolver.GetVertices().cast<float>();
		pig_render->SetFaces(faces);
		pig_render->SetVertices(vs);
		pig_render->SetColor(CM[0]);
		m_renderer.colorObjs.push_back(pig_render);

		//while (!glfwWindowShouldClose(windowPtr))
		//{
			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			auto cameras = frame.get_cameras();
			auto rawimgs = frame.get_imgs_undist();
			cv::Mat rawpack;
			packImgBlock(rawimgs, rawpack);

			std::vector<cv::Mat> renders;
			for (int camid = 0; camid < cameras.size(); camid++)
			{
				Eigen::Matrix3f R = cameras[camid].R.cast<float>();
				Eigen::Vector3f T = cameras[camid].T.cast<float>();
				m_renderer.s_camViewer.SetExtrinsic(R, T);
				m_renderer.Draw();
				cv::Mat capture = m_renderer.GetImage();
				renders.push_back(capture);
			}
			cv::Mat pack_render;
			packImgBlock(renders, pack_render);
			std::stringstream ss;
			ss << "E:/debug_pig3/render/" << std::setw(6) << std::setfill('0')
				<< frameid << ".jpg";
			cv::Mat blended;
			blended = blend_images(pack_render, rawpack, 0.5);
			cv::imwrite(ss.str(), blended);

			glfwSwapBuffers(windowPtr);
			glfwPollEvents();
		//}
	}
	return 0;
}