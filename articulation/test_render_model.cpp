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
#include "../utils/obj_reader.h"

#include "test_main.h"

int test_mean_pose()
{
	// render config 
	const float kFloorDx = 0;
	const float kFloorDy = 0;

	std::cout << "In render scene now!" << std::endl;

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
	const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj");
	const ObjData cameraObj(conf_projectFolder + "/render/data/obj_model/camera.obj");

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, false);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ kFloorDx, kFloorDy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	PigModel smal(smal_config);

	//// smal random pose 
	//Eigen::VectorXd pose = Eigen::VectorXd::Random(62 * 3) * 0.3;
	//smal.SetPose(pose);
	//smal.UpdateVertices(); 

	RenderObjectColor* animal_model = new RenderObjectColor();
	Eigen::MatrixXf vertices_f = smal.GetVertices().cast<float>();
	vertices_f = vertices_f.colwise() + Eigen::Vector3f(0, 0, 0.21f);
	Eigen::MatrixXu faces_u = smal.GetFacesVert();
	animal_model->SetFaces(faces_u);
	animal_model->SetVertices(vertices_f);
	animal_model->SetColor(Eigen::Vector3f(0.5, 0.5, 0.1));

	//smal.testReadJoint("F:/projects/model_preprocess/designed_pig/extracted/framedata/joints_4.txt");

	Eigen::MatrixXf joints = smal.GetJoints().cast<float>();
	joints = joints.colwise() + Eigen::Vector3f(0, 0, 0.21f);

	std::cout << joints << std::endl;
	Eigen::VectorXi parents = smal.GetParents();
	std::vector<Eigen::Vector3f> balls;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	GetBallsAndSticks(joints, parents, balls, sticks);
	int jointnum = smal.GetJointNum();
	std::vector<int> color_ids = {
		0,0,0,0,0, // spine
		1,1,1,1,1,1,1,1, // right back
		2,2,2,2,2,2,2,2, // left back 
		0,0,0, 1,2, 1,1,1,1,1,2,2,2,2,2, 0,0,// head 
		3,3,3,3,3,3,3,3, // right front 
		5,5,5,5,5,5,5,5, // tail 
		4,4,4,4,4,4,4,4 // left front
	};
	std::vector<Eigen::Vector3f> colors;
	colors.resize(jointnum, CM[0]);
	for (int i = 0; i < color_ids.size(); i++)colors[i] = CM[color_ids[i]];
	BallStickObject* p_skel = new BallStickObject(ballObj, stickObj, balls, sticks, 0.01, 0.005, colors);

	//smal.testReadJoint("F:/projects/model_preprocess/designed_pig/extracted/framedata/joints_diff.txt");
	smal.testReadJoint("F:/projects/model_preprocess/designed_pig/pig_prior/tmp/testjoint.txt");
	Eigen::MatrixXf joints2 = smal.GetJoints().cast<float>();

	std::vector<Eigen::Vector3f> balls2;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks2;
	GetBallsAndSticks(joints2, parents, balls2, sticks2);
	for (int i = 0; i < colors.size(); i++)
	{
		colors[i] = colors[i] * 0.5;
	}
	BallStickObject* p_skel2 = new BallStickObject(ballObj, stickObj, balls2, sticks2, 0.01, 0.005, colors);

	m_renderer.colorObjs.push_back(animal_model);
	//m_renderer.skels.push_back(p_skel); 
	//m_renderer.skels.push_back(p_skel2); 


	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

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
	const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj");
	const ObjData cameraObj(conf_projectFolder + "/render/data/obj_model/camera.obj");

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, false);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
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
	BallStickObject* pointcloud = new  BallStickObject(ballObj, balls, sizes, colors);
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