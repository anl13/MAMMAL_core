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

int test_mean_pose()
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

	//Eigen::Vector3f up; up << 0.f, 0.f, 1.f;
	//Eigen::Vector3f pos; pos << -1.f, 1.5f, 0.8f;
	//Eigen::Vector3f center = Eigen::Vector3f::Zero();
	//Eigen::Vector3f up; up << 0.00644722, 0.00535911, 0.999965;
	//Eigen::Vector3f pos; pos << 0.92963, 0.678552, -0.0248966;
	//Eigen::Vector3f center; center << 0.131863 , 0.0613784, - 0.0129145;

	Eigen::Vector3f up; up << -0.289519, -0.293115, 0.911188;
	Eigen::Vector3f pos; pos << 0.78681, 0.706331, 0.402439;
	Eigen::Vector3f center; center << 0.131863, 0.0613784, -0.0129145;

	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
	//m_renderer.s_camViewer.SetExtrinsic(cams[0].R.cast<float>(), cams[1].T.cast<float>());

	// init element obj
	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);


	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config_sym.json";
	PigModelDevice smal(smal_config);
	smal.UpdateVertices();
	smal.UpdateNormalFinal(); 

	//// smal random pose 
	RenderObjectColor * animal_model = new RenderObjectColor(); 
	animal_model->SetFaces(smal.GetFacesVert());
	animal_model->SetVertices(smal.GetVertices());
	animal_model->SetNormal(smal.GetNormals()); 
	animal_model->SetColor(Eigen::Vector3f(1.0,0.95,0.85));
	animal_model->isMultiLight = true; 
	animal_model->isFill = false; 
	m_renderer.colorObjs.push_back(animal_model);

#if 0
	std::vector<Eigen::Vector3f> joints = smal.GetJoints(); 
	std::vector<int> parents = smal.GetParents();
	std::vector<Eigen::Vector3f> balls;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	GetBallsAndSticks(joints, parents, balls, sticks);
	int jointnum = smal.GetJointNum();
	std::vector<float> sizes;
	sizes.resize(jointnum, 0.01); 
	std::vector<int> ids = {2, 4, 5, 6, 7, 8, 13, 14, 15, 16, 38, 39, 40, 41, 54, 55, 56, 57, 21, 22, 23 };
	std::vector<Eigen::Vector3f> colors;
	colors.resize(jointnum, Eigen::Vector3f(1.0,0.95,0.85));
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

#if 1
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

	m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1)); 

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	m_renderer.Draw(); 
	cv::Mat img = m_renderer.GetImage(); 
	cv::imwrite("E:/pig_middle_data/picture_model/skel.png", img);
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

	return 0;
}



int test_body_part()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");
	std::vector<Camera> cams = readCameras();

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
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	//m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
	m_renderer.s_camViewer.SetExtrinsic(cams[0].R.cast<float>(), cams[1].T.cast<float>());

	// init element obj
	// init element obj
	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareMesh.faces_v_vec);
	chess_floor->SetVertices(squareMesh.vertices_vec);
	chess_floor->SetNormal(squareMesh.normals_vec, 2);
	chess_floor->SetTexcoords(squareMesh.textures_vec, 1);
	chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config_sym.json";
	PigModel smal(smal_config);
	smal.determineBodyPartsByWeight2(); 

	Eigen::MatrixXf vertices = smal.GetVertices().cast<float>(); 

	std::vector<Eigen::Vector3f> balls; 
	std::vector < std::pair < Eigen::Vector3f, Eigen::Vector3f> > sticks; 
	std::vector<Eigen::Vector3f> colors; 
	std::vector<float> sizes; 
	for (int i = 0; i < smal.GetVertexNum(); i++)
	{
		balls.push_back(vertices.col(i));
	}
	colors.resize(vertices.cols());
	sizes.resize(vertices.cols(), 0.003); 
	std::vector<BODY_PART> parts = smal.GetBodyPart(); 
	for (int i = 0; i < vertices.cols(); i++)
	{
		//int part = parts[i];
		//colors[i] = CM[part]; 
		if (parts[i] == HEAD) colors[i] = CM[0]; 
		else colors[i] = CM[1]; 
	}
	BallStickObject* pointcloud = new  BallStickObject(ballMeshEigen, balls, sizes, colors);
	m_renderer.skels.push_back(pointcloud); 

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

	return 0;
}



void test_texture()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	//Eigen::Vector3f up; up << 0.f, 0.f, 1.f;
	//Eigen::Vector3f pos; pos << -1.f, 1.5f, 0.8f;
	//Eigen::Vector3f center = Eigen::Vector3f::Zero();

	Eigen::Vector3f up; up << -0.289519, -0.293115, 0.911188;
	Eigen::Vector3f pos; pos << 0.78681, 0.706331, 0.402439;
	Eigen::Vector3f center; center << 0.131863, 0.0613784, -0.0129145;
	// init renderer 
	Renderer::s_Init();

	Renderer m_renderer(conf_projectFolder + "/render/shader/");

	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	//std::string point_file = conf_projectFolder + "/data/calibdata/adjust_new/points3d.txt";
	//std::vector<Eigen::Vector3f> points = read_points(point_file);
	//std::cout << "pointsize:  " << points.size() << std::endl;
	//std::vector<float> sizes(points.size(), 0.05f);
	//std::vector<Eigen::Vector3f> balls, colors;
	//balls = points; 
	//colors.resize(points.size());
	//for (int i = 0; i < points.size(); i++)
	//{
	//	colors[i] = CM[0];
	//}
	//BallStickObject* skelObject = new BallStickObject(ballMeshEigen, balls, sizes, colors);
	//m_renderer.skels.push_back(skelObject);

	Mesh obj;
	obj.Load("D:/Projects/animal_calib/data/artist_model_sym3/manual_artist_sym.obj");
	for (int i = 0; i < obj.vertices_vec.size(); i++)
	{
		obj.vertices_vec[i] += Eigen::Vector3f(0, 0, 0);
	}

	obj.ReMapTexture();

	//RenderObjectColor * p_model = new RenderObjectColor(); 
	//p_model->SetVertices(obj.vertices_vec); 
	//p_model->SetFaces(obj.faces_v_vec); 
	//p_model->SetNormal(obj.normals_vec); 
	//p_model->SetColor(CM[0]); 
	//m_renderer.colorObjs.push_back(p_model); 

	RenderObjectTexture* p_model = new RenderObjectTexture();
	p_model->SetTexture(conf_projectFolder + "/articulation/visibility/colored_parts.png");
	p_model->SetFaces(obj.faces_t_vec);
	p_model->SetVertices(obj.vertices_vec_t);
	p_model->SetNormal(obj.normals_vec_t, 2);
	p_model->SetTexcoords(obj.textures_vec, 1);
	p_model->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(p_model);

	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));

	//m_renderer.createScene(conf_projectFolder);


	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}



void test_bone_var()
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

	Eigen::Vector3f up; up << -0.289519, -0.293115, 0.911188;
	Eigen::Vector3f pos; pos << 0.78681, 0.706331, 0.402439;
	Eigen::Vector3f center; center << 0.131863, 0.0613784, -0.0129145;

	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
	//m_renderer.s_camViewer.SetExtrinsic(cams[0].R.cast<float>(), cams[1].T.cast<float>());

	// init element obj
	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);


	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config_sym.json";
	PigModelDevice smal(smal_config);
	std::vector<int> jointids = {
		39,40,41,42, 55,56,57,58
	};
	for (int i = 0; i < jointids.size(); i++)
	{	
		int jid = jointids[i];
		smal.m_host_boneScales[jid] = 1;
	}
	smal.UpdateVertices();
	smal.UpdateNormalFinal();

	//// smal random pose 
	RenderObjectColor * animal_model = new RenderObjectColor();
	animal_model->SetFaces(smal.GetFacesVert());
	animal_model->SetVertices(smal.GetVertices());
	animal_model->SetNormal(smal.GetNormals());
	animal_model->SetColor(Eigen::Vector3f(1.0, 0.95, 0.85));
	animal_model->isMultiLight = true;
	animal_model->isFill = false;
	m_renderer.colorObjs.push_back(animal_model);

#if 1
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

	m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	m_renderer.Draw();
	cv::Mat img = m_renderer.GetImageOffscreen();
	//cv::imwrite("E:/pig_middle_data/picture_model/skel.png", img);
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

}

