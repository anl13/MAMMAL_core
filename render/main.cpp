#include "renderer.h" 
#include "render_object.h" 
#include <iostream> 
#include <fstream> 
#include <sstream> 
#ifndef WIN32
#include <unistd.h> 
#else 
#ifndef _UNISTD_H
#define _UNISTD_H
#include <io.h>
#include <process.h>
#endif /* _UNISTD_H */
#endif 
#include "../utils/camera.h"
#include "../utils/math_utils.h"
#include "render_utils.h" 
#include "../utils/colorterminal.h" 
#include "../utils/model.h"

void main()
{
	const float kFloorDx = 0.28;
	const float kFloorDy = 0.2;
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");

	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
	std::cout << K << std::endl;

	Eigen::Vector3f up; up << 0, 0, 1;
	Eigen::Vector3f pos; pos << -1, 1.5, 0.8;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();

	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	// init element obj
	const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj");
	const ObjData cameraObj(conf_projectFolder + "/render/data/obj_model/camera.obj");

	//RenderObjectTexture* chess_floor = new RenderObjectTexture();
	//chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	//chess_floor->SetFaces(squareObj.faces, true);
	//chess_floor->SetVertices(squareObj.vertices);
	//chess_floor->SetTexcoords(squareObj.texcoords);
	//chess_floor->SetTransform({ kFloorDx, kFloorDy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	//m_renderer.texObjs.push_back(chess_floor);

	//std::string point_file = conf_projectFolder + "/results/points3d.txt";
	//std::vector<Eigen::Vector3d> points = read_points(point_file);
	//std::vector<float> sizes(points.size(), 0.05f);
	//std::vector<Eigen::Vector3f> balls, colors;
	//balls.resize(points.size());
	//colors.resize(points.size());
	//for (int i = 0; i < points.size(); i++)
	//{
	//	balls[i] = points[i].cast<float>();
	//	colors[i] = CM[0];
	//}
	//BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors);
	//m_renderer.skels.push_back(skelObject);

	Model obj;
	obj.Load("F:/projects/model_preprocess/designed_pig/pig_prior/tmp/samples/est0.obj");

	std::cout << obj.vertices.cols() << std::endl; 
	RenderObjectMesh* p_model = new RenderObjectMesh();
	p_model->SetVertices(obj.vertices.cast<float>());
	p_model->SetFaces(obj.faces);

	Eigen::MatrixXf normals = obj.normals.cast<float>(); 
	p_model->SetColors(normals); 
	p_model->SetNormal(normals); 

	m_renderer.meshObjs.push_back(p_model); 

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}

