#include "test_main.h" 

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

#include "../utils/obj_reader.h"

#include "test_main.h"
#include "../utils/timer_util.h"

#include "pigmodeldevice.h" 

void test_gpu()
{
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
	chess_floor->SetTransform({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	PigModelDevice smal(smal_config);

	// smal random pose 
	Eigen::VectorXf pose = Eigen::VectorXf::Random(smal.GetJointNum() * 3) * 0.3;
	smal.SetPose(pose);
	TimerUtil::Timer<std::chrono::milliseconds> tt;
	tt.Start();
	smal.UpdateVertices();
	std::cout << "Time elapsed: " << tt.Elapsed() << " ms" << std::endl;

	RenderObjectColor* animal_model = new RenderObjectColor();
	std::vector<Eigen::Vector3f> vertices_f = smal.GetVertices(); 
	std::vector<Eigen::Vector3u> faces_u = smal.GetFacesVert();
	animal_model->SetFaces(faces_u);
	animal_model->SetVertices(vertices_f);
	animal_model->SetColor(Eigen::Vector3f(0.5, 0.5, 0.1));

	m_renderer.colorObjs.push_back(animal_model);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

}