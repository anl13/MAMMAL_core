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

int run_shape()
{
	std::string folder = "D:/Projects/animal_calib/data/pig_model_noeye/";
	std::string pig_config = "D:/Projects/animal_calib/smal/pigmodel_config.json";
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
	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
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

	for (int frameid = startid; frameid < startid + framenum; frameid++)
	{
		std::cout << "processing frame " << frameid << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		frame.view_dependent_clean();
		frame.matching_by_tracking();
		frame.solve_parametric_model();
		//for (int i = 0; i < 4; i++) frame.debug_fitting(i); 
		auto models = frame.get_models();
		auto m_matched = frame.get_matched();
		
		
#ifdef VOLUME
		m_renderer.colorObjs.clear();
		m_renderer.skels.clear();

		for (int pid = 0; pid < 4; pid++)
		{
			auto m_rois = frame.getROI(pid);
			std::cout << "pid: " << pid << std::endl; 
			Volume V;
			Eigen::MatrixXd joints = models[pid]->getZ();
			std::cout << joints.transpose() << std::endl;
			V.center = joints.col(20).cast<float>();
			V.computeVolumeFromRoi(m_rois);
			std::cout << "compute volume now. " << std::endl;
			V.getSurface();
			std::cout << "find surface now." << std::endl;

			std::stringstream ss;
			ss << "E:/debug_pig2/visualhull/" << pid << "/" << std::setw(6) << std::setfill('0')
				<< frameid << ".xyz";
			V.saveXYZFileWithNormal(ss.str());
			std::stringstream cmd;
			cmd << "D:/Projects/animal_calib/PoissonRecon.x64.exe --in " << ss.str() << " --out " << ss.str() << ".ply";
			const std::string cmd_str = cmd.str();
			const char* cmd_cstr = cmd_str.c_str();
			system(cmd_cstr);

			//int p_num = V.point_cloud.size();
			//std::vector<Eigen::Vector3f> colors(p_num, CM[pid]*0.5);
			//std::vector<float> sizes(p_num, 0.003);
			//std::vector<Eigen::Vector3f> balls;
			//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
			//Eigen::VectorXi parents;
			//GetBallsAndSticks(V.point_cloud_eigen, parents, balls, sticks);
			//BallStickObject* pointcloud = new BallStickObject(ballObj, balls, sizes, colors);
			//m_renderer.skels.push_back(pointcloud);

			//std::vector<Eigen::Vector3d> points;
			//std::vector<Eigen::Vector2i> edges;
			//V.get3DBox(points, edges);
			//std::vector<Eigen::Vector3f> balls_box;
			//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks_box;
			//GetBallsAndSticks(points, edges, balls_box, sticks_box);
			//BallStickObject* box_3d =
			//	new BallStickObject(ballObj, stickObj, balls_box,
			//		sticks_box, 0.01, 0.005, CM[pid]);
			//m_renderer.skels.push_back(box_3d);
			//std::cout << "render box!" << std::endl;

			//RenderObjectColor* pig_render = new RenderObjectColor();
			//Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces = models[pid]->GetFacesVert();
			//Eigen::MatrixXf vs = models[pid]->GetVertices().cast<float>();
			//pig_render->SetFaces(faces);
			//pig_render->SetVertices(vs);
			//pig_render->SetColor(CM[pid]);
			//m_renderer.colorObjs.push_back(pig_render);
			//std::cout << "render pigmodel" << std::endl;
		}

		//while (!glfwWindowShouldClose(windowPtr))
		//{
		//	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		//	glPolygonMode(GL_FRONT_AND_BACK, GL_CULL_FACE);

		//	m_renderer.Draw();
		//	glfwSwapBuffers(windowPtr);
		//	glfwPollEvents();
		//};
#endif 

#ifdef SHAPE_SOLVER
		auto m_rois = frame.getROI(0);
		Eigen::VectorXd pose = models[m_pid]->GetPose(); 
		Eigen::Vector3d trans = models[m_pid]->GetTranslation(); 
		double scale = models[m_pid]->GetScale(); 
		shapesolver.SetPose(pose); 
		shapesolver.SetTranslation(trans); 
		shapesolver.SetScale(scale);
		shapesolver.RescaleOriginVertices(); 
		shapesolver.UpdateNormalOrigin(); 
		shapesolver.UpdateNormalShaped();
		//cv::Mat packMask;
		//vector<cv::Mat> masks;
		//for (int i = 0; i < m_rois.size(); i++)masks.push_back(m_rois[i].mask);
		//packImgBlock(masks, packMask);
		//cv::namedWindow("mask", cv::WINDOW_NORMAL);
		//cv::imshow("mask", packMask);
		//cv::waitKey();

		shapesolver.InitNodeAndWarpField(); 

		int iter = 0; 
		for (; iter < 10; iter++)
		{
			shapesolver.UpdateVertices();
			shapesolver.UpdateVerticesTex();
			m_renderer.colorObjs.clear(); 
			m_renderer.texObjs.clear(); 

			Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces = shapesolver.GetFacesTex();
			Eigen::MatrixXf vs = shapesolver.GetVerticesTex().cast<float>();
			Eigen::MatrixXf texcoords = shapesolver.GetTexcoords().cast<float>(); 
			
			RenderObjectColor* pig_render = new RenderObjectColor();
			pig_render->SetFaces(faces);
			pig_render->SetVertices(vs);
			//Eigen::Vector3f color = CM[0];
			Eigen::Vector3f color(1.0, 1.0, 1.0); 
			pig_render->SetColor(color);
			m_renderer.colorObjs.push_back(pig_render);

			//RenderObjectTexture* pig_tex_render = new RenderObjectTexture(); 
			//pig_tex_render->SetFaces(faces); 
			//pig_tex_render->SetVertices(vs); 
			//pig_tex_render->SetTexture(folder + "/piguv1.png");
			//pig_tex_render->SetTexcoords(texcoords); 
			//m_renderer.texObjs.push_back(pig_tex_render); 
			
			auto cameras = frame.get_cameras();
			std::vector<cv::Mat> renders;
			for (int view = 0; view < m_matched[m_pid].view_ids.size(); view++)
			{
				int camid = m_matched[m_pid].view_ids[view];
				Eigen::Matrix3f R = cameras[camid].R.cast<float>();
				Eigen::Vector3f T = cameras[camid].T.cast<float>();
				m_renderer.s_camViewer.SetExtrinsic(R, T);
				m_renderer.Draw();
				cv::Mat capture = m_renderer.GetImage();
				shapesolver.feedData(m_rois[view], models[m_pid]->getBodyState());
				shapesolver.feedRender(capture);
				renders.push_back(capture);
			}

			// debug
			cv::Mat pack_render;
			packImgBlock(renders, pack_render);
			//cv::namedWindow("test", cv::WINDOW_NORMAL);
			//cv::imshow("test", pack_render);
			//cv::waitKey();
			std::stringstream ss; 
			ss << "E:/debug_pig2/shapeiter/" << std::setw(6)
				<< iter << ".jpg"; 
			cv::imwrite(ss.str(), pack_render); 

			std::cout << RED_TEXT("iter:") << iter << std::endl;

			//shapesolver.iterateStep(iter);
			shapesolver.NaiveNodeDeformStep(iter); 
			shapesolver.clearData(); 

			glfwSwapBuffers(windowPtr);
			glfwPollEvents();
		}
		shapesolver.saveState("shapestate.txt"); 
#endif // SHAPE_SOLVER


	}
	return 0;
}