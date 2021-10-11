#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <io.h> 
#include <process.h> 

#include "../render/renderer.h"
#include "../render/render_object.h" 
#include "../render/render_utils.h"
#include "../utils/camera.h"
#include "../utils/math_utils.h" 
#include "../utils/image_utils.h" 

#include "pigmodel.h"
#include "pigsolver.h"
#include "pigmodeldevice.h" 
#include "pigsolverdevice.h" 
#include "../utils/mesh.h"

#include "test_main.h"
#include "../utils/timer_util.h"

// 2021.10.6: 
// This code is used to render behavior. 

int test_render_behavior()
{
	// render config 
	std::cout << "In render scene now!" << std::endl;

	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");
	std::vector<Eigen::Vector3f> CM2 = getColorMapEigenF("anliang_rgb");
	std::vector<Eigen::Vector3f> CM3 = getColorMapEigenF("anliang_blend");
	std::vector<Camera> cams = readCameras();

	// init a camera 
	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	//// this parameter is used for NM paper. 
	Eigen::Vector3f up(-0.671268, -0.0130926, 0.741099);
	Eigen::Vector3f pos(2.39672, -0.0313694, 1.54245);
	Eigen::Vector3f center(0.697682, 0.0413541, 0.0165031);
	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	// init element obj
	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);


	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config_sym.json";
	PigSolverDevice smal(smal_config);
	smal.UpdateVertices();
	smal.UpdateNormalFinal();

	std::string result_folder = "H:/results/0703_motion2/state/";
	int A = 2; 
	int B = 1; 
	std::vector<int> frameids = { 13507, 13510, 13514, 13518, 13521 }; 
	
	for (int index = 0; index < 5; index++)
	{
		m_renderer.clearAllObjs(); 

		std::stringstream ss_A;
		ss_A << result_folder << "pig_" << A << "_frame_" << std::setw(6) << std::setfill('0') << frameids[index] << ".txt";
		smal.readState(ss_A.str());
		smal.UpdateVertices();
		smal.UpdateNormalFinal();
#if 1
		//// smal random pose 
		RenderObjectColor * animal_modelA = new RenderObjectColor();
		auto verticesA = smal.GetVertices(); 
		//for (int i = 0; i < verticesA.size(); i++)
		//{
		//	verticesA[i](1) -= 0.6; 
		//	verticesA[i](1) += index * 0.3;
		//}
		animal_modelA->SetFaces(smal.GetFacesVert());
		animal_modelA->SetVertices(verticesA);
		animal_modelA->SetNormal(smal.GetNormals());
		//animal_modelA->SetColor(CM[0]*(0.6 + 0.1 * index));
		animal_modelA->SetColor(CM[0]); 
		animal_modelA->isMultiLight = true;
		if(index < 4) 
		animal_modelA->isFill = false;
		else animal_modelA->isFill = false; 
		m_renderer.colorObjs.push_back(animal_modelA);

		std::stringstream ss_B;
		ss_B << result_folder << "pig_" << B << "_frame_" << std::setw(6) << std::setfill('0') << frameids[index] << ".txt";
		smal.readState(ss_B.str());
		smal.UpdateVertices();
		smal.UpdateNormalFinal();

		RenderObjectColor * animal_modelB = new RenderObjectColor();
		auto verticesB = smal.GetVertices();
		//for (int i = 0; i < verticesB.size(); i++)
		//{
		//	verticesB[i](1) -= 0.6;
		//	verticesB[i](1) += index * 0.3;
		//}
		animal_modelB->SetFaces(smal.GetFacesVert());
		animal_modelB->SetVertices(verticesB);
		animal_modelB->SetNormal(smal.GetNormals());
		//animal_modelB->SetColor(CM[1] * (0.6 + 0.1 * index));
		animal_modelB->SetColor(CM[1]); 
		animal_modelB->isMultiLight = true;
		if(index < 4)
		animal_modelB->isFill = false;
		else animal_modelB->isFill = false; 
		m_renderer.colorObjs.push_back(animal_modelB);
#endif 


#if 0 // render joints 
		std::vector<Eigen::Vector3f> joints = smal.GetJoints();
		std::vector<int> parents = smal.GetParents();
		std::vector<Eigen::Vector3f> balls;
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
		GetBallsAndSticks(joints, parents, balls, sticks);
		int jointnum = smal.GetJointNum();
		std::vector<float> sizes;
		sizes.resize(jointnum, 0.01);
		std::vector<int> ids = { 2, 4, 5, 6, 7, 8, 13, 14, 15, 16, 38, 39, 40, 41, 54, 55, 56, 57, 21, 22, 23 };
		std::vector<Eigen::Vector3f> colors;
		colors.resize(jointnum, Eigen::Vector3f(1.0, 0.95, 0.85));
		for (int k = 0; k < ids.size(); k++)
		{
			colors[ids[k]] = CM[2];
			sizes[ids[k]] = 0.012;
		}
		std::vector<float> bone_sizes(sticks.size(), 0.005);
		std::vector<Eigen::Vector3f> bone_colors(sticks.size());
		for (int k = 0; k < sticks.size(); k++)
		{
			bone_colors[k] = Eigen::Vector3f(1.0, 0.95, 0.85);
		}
		BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen, balls, sticks, sizes, bone_sizes, colors, bone_colors);
		p_skel->isMultiLight = false;
		m_renderer.skels.push_back(p_skel);
#endif 

#if 0 // render skel 
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
		std::vector<Eigen::Vector3f> skels = smal.getRegressedSkel_host();
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

#endif 
		//m_renderer.createPlane(conf_projectFolder, 1.5);
		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));

		GLFWwindow* windowPtr = m_renderer.s_windowPtr;

		//m_renderer.Draw();
		cv::Mat img = m_renderer.GetImageOffscreen();
		cv::imwrite("D:/paper_writing_figs/behavior/head_attack_" + std::to_string(index)+".png", img);
	}
	//m_renderer.createPlane(conf_projectFolder, 2);
	//m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));

	//GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	////m_renderer.Draw();
	//cv::Mat img = m_renderer.GetImageOffscreen();
	//cv::imwrite("D:/paper_writing_figs/behavior/head_attack_++.png", img);
	//while (!glfwWindowShouldClose(windowPtr))
	//{
	//	m_renderer.Draw();
	//	glfwSwapBuffers(windowPtr);
	//	glfwPollEvents();
	//};

	return 0;
}
