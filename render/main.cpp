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

#include "render_utils.h" 
#include "../utils/camera.h"
#include "../utils/math_utils.h"
#include "../utils/colorterminal.h" 
#include "../utils/mesh.h"
#include "../utils/geometry.h"

std::vector<Camera> readCameras()
{
	std::vector<Camera> cams;
	std::vector<int> m_camids = {
		0,1,2,5,6,7,8,9,10,11
	};
	int m_camNum = m_camids.size();
	std::string m_camDir = "D:/Projects/animal_calib/data/calibdata/adjust/";
	for (int camid = 0; camid < m_camNum; camid++)
	{
		std::stringstream ss;
		ss << m_camDir << std::setw(2) << std::setfill('0') << m_camids[camid] << ".txt";
		std::ifstream camfile;
		camfile.open(ss.str());
		if (!camfile.is_open())
		{
			std::cout << "can not open file " << ss.str() << std::endl;
			exit(-1);
		}
		Eigen::Vector3f rvec, tvec;
		for (int i = 0; i < 3; i++) {
			float a;
			camfile >> a;
			rvec(i) = a;
		}
		for (int i = 0; i < 3; i++)
		{
			float a;
			camfile >> a;
			tvec(i) = a;
		}

		Camera camUndist = Camera::getDefaultCameraUndist();
		camUndist.SetRT(rvec, tvec);
		cams.push_back(camUndist);
		camfile.close();
	}
	return cams;
}

void test_cuda()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");

	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	Eigen::Vector3f up; up << 0.f, 0.f, 1.f;
	Eigen::Vector3f pos; pos << -1.f, 1.5f, 0.8f;
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
	//chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
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

	Mesh obj;
	obj.Load("F:/projects/model_preprocess/designed_pig/extracted/artist_model/model_triangle.obj");
	MeshEigen objeigen; 
	obj.GetMeshEigen(objeigen); 

	RenderObjectMesh* p_model = new RenderObjectMesh();
	p_model->SetVertices(obj.vertices_vec);
	p_model->SetFaces(obj.faces_v_vec);
	p_model->SetColors(obj.normals_vec); 
	p_model->SetNormal(obj.normals_vec); 

	m_renderer.meshObjs.push_back(p_model); 
	//m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 0.5f, 0.5f, 1.0f)); 

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	m_renderer.initResource();
	cv::Mat img; 
	img.create(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_32FC4); 
	cv::Mat depth; 
	depth.create(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_32FC1); 
	while (!glfwWindowShouldClose(windowPtr))
	{
		
		m_renderer.beginOffscreenRender(); 

		m_renderer.Draw();

		m_renderer.mapRenderingResults(); 
		cudaMemcpy2DFromArray(
			img.data, img.cols * img.elemSize(),
			m_renderer.m_colorArray, 0, 0, img.cols*img.elemSize(),
			img.rows,
			cudaMemcpyDeviceToHost
		);
		cv::flip(img, img, 0); 
		cv::extractChannel(img, depth, 2);

		cv::Mat depth_pseudo = pseudoColor(depth); 

		m_renderer.unmapRenderingResults(); 

		m_renderer.endOffscreenRender();

		cv::namedWindow("render", cv::WINDOW_NORMAL); 
		cv::imshow("render", depth_pseudo); 
		cv::waitKey(); 
		cv::destroyAllWindows(); 

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();

		break; 
	};

	m_renderer.releaseResource(); 
}



void test_depth()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/render";
	std::string smal_config = "D:/Projects/animal_calib/smal/smal2_config.json";
	auto CM = getColorMapEigen("anliang_rgb");
	std::vector<Camera> cams = readCameras();

	// init render
	Eigen::Matrix3f K =  cams[0].K;
	K.row(0) = K.row(0) / 1920.f; 
	K.row(1) = K.row(1) / 1080.f;
	//K << 0.698, 0, 0.502,
	//	0, 1.243, 0.483,
	//	0, 0, 1;
	std::cout << K << std::endl;

	Eigen::Vector3f up; up << 0, 0, -1;
	Eigen::Vector3f pos; pos << -1, 1.5, -0.8;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();

	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	// init element obj
	const ObjData ballObj(conf_projectFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/data/obj_model/square.obj");

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.f, 0.f, 0.f, 1.0f));

	Mesh obj;
	obj.Load("F:/projects/model_preprocess/designed_pig/extracted/artist_model/model_triangle.obj");

	RenderObjectMesh* p_model = new RenderObjectMesh();
	p_model->SetVertices(obj.vertices_vec);
	p_model->SetFaces(obj.faces_v_vec);
	p_model->SetColors(obj.normals_vec);
	p_model->SetNormal(obj.normals_vec);
	m_renderer.meshObjs.push_back(p_model); 

	m_renderer.s_camViewer.SetExtrinsic(cams[0].R, cams[0].T);
	// render depth 
	m_renderer.initResource();
	cv::Mat img;
	img.create(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_32FC4);
	cv::Mat depth;
	depth.create(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_32FC1);

	glEnable(GL_CULL_FACE); 
	m_renderer.beginOffscreenRender();
	m_renderer.Draw("depth");
	m_renderer.mapRenderingResults();
	cudaMemcpy2DFromArray(
		img.data, img.cols * img.elemSize(),
		m_renderer.m_colorArray, 0, 0, img.cols*img.elemSize(),
		img.rows,
		cudaMemcpyDeviceToHost
	);
	cv::flip(img, img, 0);
	cv::extractChannel(img, depth, 2);
	cv::Mat depth_pseudo = pseudoColor(-depth);
	m_renderer.unmapRenderingResults();
	m_renderer.endOffscreenRender();
	m_renderer.releaseResource();

	MeshEigen objeigen;
	obj.GetMeshEigen(objeigen); 
	cv::Mat mask = drawCVDepth(objeigen.vertices, objeigen.faces, cams[0]);
	cv::Mat blend = blend_images(depth_pseudo, mask, 0.5);

	cv::imshow("depth", depth_pseudo);
	cv::imshow("mask", mask);
	cv::imshow("blend", blend);
	cv::imwrite("E:/render_test/vis_align.png", depth_pseudo);
	cv::imwrite("E:/render_test/mask_align.png", mask);
	cv::imwrite("E:/render_test/blend_align.png", blend);

	cv::waitKey();
	cv::destroyAllWindows();

	int vertexNum = obj.vertex_num; 
	std::vector<Eigen::Vector3f> colors(vertexNum, Eigen::Vector3f(1.0f, 1.0f, 1.0f));
	//std::vector<float> sizes(vertexNum, 0.006);
	//std::vector<Eigen::Vector3f> balls = obj.vertices_vec;
	//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;

	for (int i = 0; i < vertexNum; i++)
	{
		Eigen::Vector3f v = obj.vertices_vec[i]; 
		Eigen::Vector3f uv = project(cams[0], v);
		float d = -queryDepth(depth, uv(0), uv(1));
		v = cams[0].R * v + cams[0].T; 
		std::cout << "d: " << d << "  gt: " << v(2) << std::endl;
		if (d > 0 && abs(d - v(2)) < 0.02f)
		{
			colors[i] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
		}
		else
		{
			colors[i] = Eigen::Vector3f(0.f, 0.f, 1.0f);
		}
	}
	//BallStickObject* pc = new BallStickObject(ballObj, balls, sizes, colors);
	
	m_renderer.meshObjs.clear(); 

	RenderObjectMesh* meshcolor = new RenderObjectMesh(); 
	meshcolor->SetVertices(obj.vertices_vec);
	meshcolor->SetFaces(obj.faces_v_vec);
	meshcolor->SetColors(colors);
	meshcolor->SetNormal(obj.normals_vec);

	m_renderer.meshObjs.push_back(meshcolor); 

	while (!glfwWindowShouldClose(windowPtr))
	{
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};


}

void main()
{
	test_depth(); 
}