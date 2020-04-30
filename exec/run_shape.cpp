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
	std::string pig_config = "D:/Projects/animal_calib/smal/smal2_config.json";
	//std::string pig_config = "D:/Projects/animal_calib/smal/pigmodel_config.json";

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

	for (int frameid = startid; frameid < startid + 1; frameid++)
	{
		std::cout << "processing frame " << frameid << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		//frame.view_dependent_clean();
		//frame.matching_by_tracking();
		frame.load_labeled_data();
		frame.solve_parametric_model();
		auto m_matched = frame.get_matched();
		
#ifdef VOLUME
		m_renderer.colorObjs.clear();
		m_renderer.skels.clear();

		for (int pid = 0; pid < 4; pid++)
		{
			//auto m_rois = frame.getROI(pid);
			//std::cout << "pid: " << pid << std::endl; 
			Volume V;

			std::stringstream ss;
			ss << "E:/debug_pig2/visualhull/" << pid << "/" << std::setw(6) << std::setfill('0')
				<< frameid << ".xyz";

			V.readXYZFileWithNormal(ss.str());

			int p_num = V.point_cloud.size();
			std::vector<Eigen::Vector3f> colors(p_num, CM[pid]*0.5);
			std::vector<float> sizes(p_num, 0.006);
			std::vector<Eigen::Vector3f> balls;
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
			Eigen::VectorXi parents;
			GetBallsAndSticks(V.point_cloud_eigen, parents, balls, sticks);
			BallStickObject* pointcloud = new BallStickObject(ballObj, balls, sizes, colors);
			m_renderer.skels.push_back(pointcloud);
		}

#endif 

#ifdef SHAPE_SOLVER
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
		//shapesolver.UpdateNormalOrigin(); 
		//shapesolver.UpdateNormalShaped();

		//shapesolver.mp_renderer = &m_renderer; 
		//shapesolver.m_rois = m_rois;

		
		std::shared_ptr<Model> targetModel = std::make_shared<Model>();
		targetModel->Load("E:/debug_pig2/visualhull/0/000000.obj");
		//shapesolver.setTargetModel(targetModel);
		//shapesolver.setSourceModel();
		//shapesolver.totalSolveProcedure();
		//shapesolver.SaveWarpField();
		//shapesolver.SaveObj("E:/debug_pig2/warped.obj");

		
		RenderObjectColor* pig_render = new RenderObjectColor();
		Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces
			= shapesolver.GetFacesVert();
		Eigen::MatrixXf vs = shapesolver.GetVertices().cast<float>();
		pig_render->SetFaces(faces);
		pig_render->SetVertices(vs);
		pig_render->SetColor(Eigen::Vector3f(0.8, 0.8, 0.8));
		m_renderer.colorObjs.push_back(pig_render);

		Eigen::VectorXd pose = shapesolver.GetPose();
		Eigen::Vector3d globalR = pose.segment<3>(0);
		pose.setZero(); 
		shapesolver.SetPose(pose);
		Eigen::Vector3d trans = shapesolver.GetTranslation(); 
		trans.setZero();
		shapesolver.SetTranslation(trans);
		shapesolver.UpdateVertices();
		RenderObjectColor* pig_render_target = new RenderObjectColor();
		Eigen::MatrixXf vs_t = shapesolver.GetVertices().cast<float>();
		pig_render_target->SetFaces(faces);
		pig_render_target->SetVertices(vs_t);
		pig_render_target->SetColor(Eigen::Vector3f(0.0, 0.8, 0.1));
		m_renderer.colorObjs.push_back(pig_render_target);
		shapesolver.SaveObj("D:/Projects/animal_calib/data/smal_deform_0.obj");

		/// render target volume 
		RenderObjectColor* pig_render2 = new RenderObjectColor();
		Eigen::MatrixXf vs2 = targetModel->vertices.cast<float>();
		Eigen::MatrixXu faces2 = targetModel->faces;
		pig_render2->SetFaces(faces2);
		pig_render2->SetVertices(vs2);
		pig_render2->SetColor(Eigen::Vector3f(0.0, 0.1, 0.8));
		m_renderer.colorObjs.push_back(pig_render2);
	

		while (!glfwWindowShouldClose(windowPtr))
		{
			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

			m_renderer.Draw();

			glfwSwapBuffers(windowPtr);
			glfwPollEvents();
		}
#endif // SHAPE_SOLVER
	}
	return 0;
}