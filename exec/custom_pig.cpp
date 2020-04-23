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

#include "main.h" 

#define RUN_SEQ
#define VIS 
//#define DEBUG_VIS
//#define LOAD_STATE

using std::vector; 


int run_on_sequence()
{
	const float kFloorDx = 0.28;
	const float kFloorDy = 0.2;
	std::vector<Eigen::Vector2i> bones = {
		{ 1 , 0 },
	{ 2 , 1 },
	{ 3 , 2 },
	{ 4 , 3 },
	{ 5 , 4 },
	{ 6 , 5 },
	{ 7 , 6 },
	{ 8 , 7 },
	{ 9 , 8 },
	{ 10 , 9 },
	{ 11 , 4 },
	{ 12 , 11 },
	{ 13 , 12 },
	{ 14 , 13 },
	{ 15 , 14 },
	{ 16 , 14 },
	{ 17 , 16 },
	{ 18 , 14 },
	{ 19 , 18 },
	{ 20 , 4 },
	{ 21 , 20 },
	{ 22 , 21 },
	{ 23 , 22 },
	{ 24 , 23 },
	{ 25 , 24 },
	{ 26 , 0 },
	{ 27 , 26 },
	{ 28 , 27 },
	{ 29 , 28 },
	{ 30 , 29 },
	{ 31 , 0 },
	{ 32 , 31 },
	{ 33 , 32 },
	{ 34 , 33 },
	{ 35 , 34 },
	{ 36 , 35 },
	{ 37 , 36 },
	{ 38 , 0 },
	{ 39 , 38 },
	{ 40 , 39 },
	{ 41 , 40 },
	{ 42 , 41 }
	};

#ifdef _WIN32
    std::string folder = "D:/Projects/animal_calib/data/pig_model/"; 
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
#else 
	std::string folder = "/home/al17/animal/animal_calib/data/pig_model/"; 
	std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
#endif 
	SkelTopology topo = getSkelTopoByType("UNIV"); 
	FrameData frame; 
	frame.configByJson(conf_projectFolder + "/associate/config.json");
	int startid = frame.get_start_id(); 

#ifdef VIS
    //// rendering pipeline. 
    auto CM = getColorMapEigen("anliang_rgb"); 

    // init a camera 
    Eigen::Matrix3f K; 
    //K << 0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0, 1;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
    // std::cout << K << std::endl; 

    Eigen::Vector3f up; up << 0,0, -1; 
    Eigen::Vector3f pos; pos << -1, 1.5, -0.8; 
    Eigen::Vector3f center = Eigen::Vector3f::Zero(); 
    // init renderer 
    Renderer::s_Init(); 
    Renderer m_renderer(conf_projectFolder + "/render/shader/"); 
    m_renderer.s_camViewer.SetIntrinsic(K, 1, 1); 
	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));

    m_renderer.s_camViewer.SetExtrinsic(pos, up, center); 

    // init element obj
    const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData cubeObj(conf_projectFolder + "/render/data/obj_model/cube.obj");
    const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj"); 

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, true);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ kFloorDx, kFloorDy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor); 

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 
#endif 
	int framenum = frame.get_frame_num();

//#ifdef VIS
//	std::string videoname_render = "E:/debug_pig2/render.avi";
//	cv::VideoWriter writer_render(videoname_render, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 25.0, cv::Size(1920, 1080));
//	if (!writer_render.isOpened())
//	{
//		std::cout << "can not open video file " << videoname_render << std::endl;
//		return -1;
//	}
//#endif 
	TimerUtil::Timer<std::chrono::seconds> total;
	total.Start(); 

	for (int frameid = startid; frameid < startid + framenum; frameid++)
	{
		std::cout << "processing frame " << frameid << std::endl; 
		frame.set_frame_id(frameid); 
#ifndef LOAD_STATE
		frame.fetchData();
		//frame.view_dependent_clean(); 
		//frame.matching_by_tracking(); 
		frame.load_labeled_data();
		frame.solve_parametric_model(); 
		//for (int i = 0; i < 4; i++) frame.debug_fitting(i); 
		//frame.debug_chamfer(0); 

#else
		frame.read_parametric_data(); 
#endif 
		auto models = frame.get_models(); 
		
#ifdef VIS
		m_renderer.colorObjs.clear(); 
		m_renderer.skels.clear(); 
		cv::Mat det_img = frame.visualizeIdentity2D();
		std::stringstream ss1; 
		ss1 << "E:/debug_pig2/assoc/" << std::setw(6) << std::setfill('0')
			<< frameid << ".jpg"; 
		cv::imwrite(ss1.str(), det_img); 
		for (int pid = 0; pid < 4; pid++)
		{
			Eigen::Vector3f color = rgb2bgr(CM[pid]);

			RenderObjectColor* pig_render = new RenderObjectColor();
			Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces = models[pid]->GetFacesVert();
			Eigen::MatrixXf vs = models[pid]->GetVertices().cast<float>();
			pig_render->SetFaces(faces);
			pig_render->SetVertices(vs);
			pig_render->SetColor(color);
			m_renderer.colorObjs.push_back(pig_render);

			///// skels, require Z
			//std::vector<Eigen::Vector3f> balls;
			//std::vector< std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
			//Eigen::MatrixXd joints = models[pid]->getZ();
			//Eigen::MatrixXd joint_reg = models[pid]->getRegressedSkel();
			//GetBallsAndSticks(joints, topo.bones, balls, sticks);
			//BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks,
			//	0.015f, 0.01f, 0.5f * color);
			//m_renderer.skels.push_back(skelObject);

			//std::vector<Eigen::Vector2i> ori_bones = {
			//	{0,1}, {1,2} };
			//std::vector<Eigen::Vector3f> colors;
			//colors.push_back(CM[pid] * 1.1f);
			//colors.push_back(CM[pid] * 0.6f);
			//colors.push_back(CM[pid] * 0.2f);
			//Eigen::MatrixXd joints;
			//joints.resize(3, 3); 
			//vector<Eigen::Vector3d> pivots = models[pid]->getPivot(); 
			//joints.col(0) = pivots[0];
			//joints.col(1) = pivots[1]; 
			//joints.col(2) = pivots[2]; 
			//std::vector<Eigen::Vector3f> balls;
			//std::vector< std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
			//GetBallsAndSticks(joints, ori_bones, balls, sticks);
			//BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks,
			//	0.015f, 0.01f, colors);
			//m_renderer.skels.push_back(skelObject);
		}

#ifndef RUN_SEQ
		while (!glfwWindowShouldClose(windowPtr))
		{
#endif 
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
			ss << "E:/debug_pig2/render/" << std::setw(6) << std::setfill('0')
				<< frameid << ".jpg";
			cv::Mat blended; 
			blended = blend_images(pack_render, rawpack, 0.5); 
			cv::imwrite(ss.str(), blended);
			
			//m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
			//m_renderer.Draw();
			//cv::Mat cap = m_renderer.GetImage();
			//writer_render.write(cap);

			glfwSwapBuffers(windowPtr);
			glfwPollEvents();

#endif 

	}
	double timeavg = total.Elapsed(); 
	timeavg /= framenum; 
	std::cout << "Avg time: " << timeavg << " s per frame. " << std::endl; 
	std::ofstream os_time("E:/debug_pig2/avgtime.txt");
	os_time << timeavg << std::endl; 
	os_time.close();

	//system("pause"); 
    return 0; 
}