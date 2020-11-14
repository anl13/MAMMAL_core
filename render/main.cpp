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

//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/core/cuda_stream_accessor.hpp>
//#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>

#include "render_utils.h" 
#include "../utils/camera.h"
#include "../utils/math_utils.h"
#include "../utils/colorterminal.h" 
#include "../utils/mesh.h"
#include "../utils/geometry.h"

#include "../utils/timer_util.h"

#include "test_kernel.h"

#include "../utils/skel.h"

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

void test_shader()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

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
		obj.vertices_vec[i] += Eigen::Vector3f(0, 0, 0.21); 
	}

	obj.ReMapTexture(); 

	RenderObjectColor * p_model = new RenderObjectColor(); 
	p_model->SetVertices(obj.vertices_vec); 
	p_model->SetFaces(obj.faces_v_vec); 
	p_model->SetNormal(obj.normals_vec); 
	p_model->SetColor(CM[0]); 
	p_model->isMultiLight = false;
	m_renderer.colorObjs.push_back(p_model); 

	//RenderObjectTexture* p_model = new RenderObjectTexture();
	//p_model->SetTexture(conf_projectFolder + "/render/data/white_tex.png");
	//p_model->SetFaces(obj.faces_t_vec);
	//p_model->SetVertices(obj.vertices_vec_t);
	//p_model->SetNormal(obj.normals_vec_t, 2);
	//p_model->SetTexcoords(obj.textures_vec, 1);
	//p_model->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	//p_model->isMultiLight = true; 
	//m_renderer.texObjs.push_back(p_model);

	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f)); 


	//m_renderer.createScene(conf_projectFolder); 
	//m_renderer.createPlane(conf_projectFolder);


	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
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
	Renderer::s_Init(false);
	Renderer m_renderer(conf_projectFolder + "/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.f, 0.f, 0.f, 1.0f));

	Mesh obj;
	obj.Load("F:/projects/model_preprocess/designed_pig/extracted/artist_model/model_triangle.obj");

	RenderObjectColor* p_model = new RenderObjectColor();
	p_model->SetVertices(obj.vertices_vec);
	p_model->SetFaces(obj.faces_v_vec);
	p_model->SetColor(Eigen::Vector3f(1.0f, 0.0f, 0.0f));
	p_model->SetNormal(obj.normals_vec);
	m_renderer.colorObjs.push_back(p_model); 

	m_renderer.s_camViewer.SetExtrinsic(cams[0].R, cams[0].T);

	//float * depth_device; 
	//cudaMalloc((void**)&depth_device, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float));
	//m_renderer.renderDepthDevice(); 
	//cv::Mat pseudo_out, depth_out; 
	//gpupseudo_color(m_renderer.m_device_renderData, WINDOW_WIDTH, WINDOW_HEIGHT, 2.9, 0, pseudo_out, depth_out, depth_device);
	float * depth_device = m_renderer.renderDepthDevice(); 

	// render depth 
	cv::Mat depth;
	depth.create(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_32FC1);
	cudaMemcpy(depth.data, depth_device, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);  

	cv::Mat depth_pseudo = pseudoColor(depth); 

	cv::imshow("depth", depth_pseudo);

	cv::waitKey();
	cv::destroyAllWindows();

	

	std::vector<uchar> visibility(obj.vertex_num, 0); 
	pcl::gpu::DeviceArray<Eigen::Vector3f> points_device; 
	points_device.upload(obj.vertices_vec); 

	TimerUtil::Timer<std::chrono::microseconds> tt1;
	tt1.Start();
	check_visibility(depth_device, WINDOW_WIDTH, WINDOW_HEIGHT, points_device,
		cams[0].K, cams[0].R, cams[0].T, visibility);
	std::cout << tt1.Elapsed() << std::endl; 

	int vertexNum = obj.vertex_num; 
	std::vector<Eigen::Vector3f> colors(vertexNum, Eigen::Vector3f(1.0f, 1.0f, 1.0f));

	TimerUtil::Timer<std::chrono::microseconds> tt; 
	tt.Start(); 
	for (int i = 0; i < vertexNum; i++)
	{
		Eigen::Vector3f v = obj.vertices_vec[i]; 
		Eigen::Vector3f uv = project(cams[0], v);
		float d = queryDepth(depth, uv(0), uv(1));
		v = cams[0].R * v + cams[0].T; 
		//std::cout << "d: " << d << "  gt: " << v(2) << std::endl;
		if (d > 0 && abs(d - v(2)) < 0.02f)
		{
			colors[i] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
		}
		else
		{
			colors[i] = Eigen::Vector3f(0.f, 0.f, 1.0f);
		}
		if (visibility[i] > 0) colors[i] = Eigen::Vector3f(1.0f, 0.0f, 0.0f); 
		else colors[i] = Eigen::Vector3f(0.f, 0.f, 1.0f); 
	}
	std::cout << tt.Elapsed() << " mcs" << std::endl; 
	m_renderer.clearAllObjs();

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

void test_color_table()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

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
	
	int num = CM.size() > 20 ? 20 : CM.size();
	for (int i = 0; i < num; i++)
	{
		Eigen::Vector3f translation = Eigen::Vector3f::Zero(); 
		translation(0) = i * 0.1 - 1;
		translation(2) = 0.2; 
		RenderObjectColor * p_model = new RenderObjectColor();
		Eigen::MatrixXf vertices = ballMeshEigen.vertices * 0.03;
		vertices = vertices.colwise() + translation; 
		p_model->SetVertices(vertices);
		p_model->SetFaces(ballMeshEigen.faces);
		p_model->SetNormal(ballMeshEigen.normals);
		p_model->SetColor(CM[i]);
		m_renderer.colorObjs.push_back(p_model);
	}

	//m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 0.5f, 0.5f, 1.0f));

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}


void test_discrete_scene()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

	// init a camera 
	auto cameras = readCameras();
	Eigen::Matrix3f K = cameras[0].K; 
	K.row(0) = K.row(0) / 1920; 
	K.row(1) = K.row(1) / 1080;
	std::cout << K << std::endl;

	// init renderer 
	Renderer::s_Init();

	Renderer m_renderer(conf_projectFolder + "/render/shader/");

	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	//m_renderer.s_camViewer.SetExtrinsic(cameras[0].R, cameras[0].T);
	//m_renderer.s_camViewer.SetExtrinsic(R, T); 

	Eigen::Vector3f pos(0, 0, 5);
	Eigen::Vector3f up(0, 1, 0); 
	Eigen::Vector3f center(0, 0, 0);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);


	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	std::string point_file = conf_projectFolder + "/data/calibdata/adjust/points3d.txt";
	std::vector<Eigen::Vector3f> points = read_points(point_file);
	std::vector<float> sizes(points.size(), 0.05f);
	std::vector<Eigen::Vector3f> balls, colors;
	balls = points;
	colors.resize(points.size());
	for (int i = 0; i < points.size(); i++)
	{
		colors[i] = CM[0];
	}
	int id = 49;
	colors[id] = CM[1]; 
	BallStickObject* skelObject = new BallStickObject(ballMeshEigen, balls, sizes, colors);
	m_renderer.skels.push_back(skelObject);

	m_renderer.createScene(conf_projectFolder); 

	//m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 0.5f, 0.5f, 1.0f));

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}

void test_artist_labeled_sample()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

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


	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	std::string point_file = conf_projectFolder + "/data/calibdata/adjust/points3d.txt";
	std::vector<Eigen::Vector3f> points = read_points(point_file);
	std::cout << "pointsize:  " << points.size() << std::endl;
	std::vector<float> sizes(points.size(), 0.05f);
	std::vector<Eigen::Vector3f> balls, colors;
	balls = points;
	colors.resize(points.size());
	for (int i = 0; i < points.size(); i++)
	{
		colors[i] = CM[0];
	}
	BallStickObject* skelObject = new BallStickObject(ballMeshEigen, balls, sizes, colors);
	m_renderer.skels.push_back(skelObject);

	Mesh obj;
	obj.Load("F:/projects/model_preprocess/designed_pig/extracted/artist_model/model_triangle.obj");
	MeshEigen objeigen(obj);

	//RenderObjectMesh* p_model = new RenderObjectMesh();
	//p_model->SetVertices(obj.vertices_vec);
	//p_model->SetFaces(obj.faces_v_vec);
	//p_model->SetColors(obj.normals_vec); 
	//p_model->SetNormal(obj.normals_vec); 

	RenderObjectColor * p_model = new RenderObjectColor();
	p_model->SetVertices(obj.vertices_vec);
	p_model->SetFaces(obj.faces_v_vec);
	p_model->SetNormal(obj.normals_vec);
	p_model->SetColor(Eigen::Vector3f(0.2f, 0.8f, 0.5f));

	m_renderer.colorObjs.push_back(p_model);
	//m_renderer.meshObjs.push_back(p_model); 
	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 0.5f, 0.5f, 1.0f));

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}


void test_mask()
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
	Renderer::s_Init(true);

	Renderer m_renderer(conf_projectFolder + "/render/shader/");

	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	Mesh obj;
	obj.Load("F:/projects/model_preprocess/designed_pig/extracted/artist_model/model_triangle.obj");
	MeshEigen objeigen(obj);

	RenderObjectColor * p_model = new RenderObjectColor();
	p_model->SetVertices(obj.vertices_vec);
	p_model->SetFaces(obj.faces_v_vec);
	p_model->SetNormal(obj.normals_vec);
	p_model->SetColor(Eigen::Vector3f(1.0f, 0.0f, 0.0f));

	RenderObjectColor *p_model2 = new RenderObjectColor(); 
	for (int i = 0; i < ballMesh.vertex_num; i++)
	{
		ballMesh.vertices_vec[i] = ballMesh.vertices_vec[i] * 0.6 + Eigen::Vector3f(0.2, 0, 0); 
	}
	p_model2->SetVertices(ballMesh.vertices_vec); 
	p_model2->SetFaces(ballMesh.faces_v_vec); 
	p_model2->SetNormal(ballMesh.normals_vec); 
	p_model2->SetColor(Eigen::Vector3f(0.0f, 1.f, 0.f)); 

	m_renderer.colorObjs.push_back(p_model);
	m_renderer.colorObjs.push_back(p_model2); 

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	//while (!glfwWindowShouldClose(windowPtr))
	//{
	//	m_renderer.Draw("mask");

	//	glfwSwapBuffers(windowPtr);
	//	glfwPollEvents();
	//};
	m_renderer.Draw("mask"); 
	cv::Mat mask = m_renderer.GetImage();
	for (int i = 0; i < mask.rows; i++)
	{
		for (int j = 0; j < mask.cols; j++)
		{
			if (mask.at<cv::Vec3b>(i, j)[0] != 0 && mask.at<cv::Vec3b>(i, j)[0] != 255)
			{
				mask.at<cv::Vec3b>(i, j)[0] = 255; 
				mask.at<cv::Vec3b>(i, j)[1] = 255;
				mask.at<cv::Vec3b>(i, j)[2] = 255;
				std::cout << i << ',' << j << std::endl;
			}
		}
	}
	cv::namedWindow("mask", cv::WINDOW_NORMAL); 
	cv::imshow("mask", mask); 
	cv::waitKey(); 
}

void test_trajectory()
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

	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
	SkelTopology topo = getSkelTopoByType("UNIV"); 
	std::vector<int> kpt_color_ids = {
			9,2,1,2,1, // face 
			2,1,2,1,2,1, // front legs 
			4,3,4,3,4,3, // back legs 
			5,9,5,6,5,5 // ceneter and tail 
	};

	for (int frameid = 750; frameid < 1500; frameid++)
	{
		std::stringstream ss;
		ss << "G:/pig_middle_data/teaser/joints/pig_" << 0 << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
		std::string point_file = ss.str();
		std::vector<Eigen::Vector3f> points = read_points(point_file);
		std::vector<float> sizes(points.size(), 0.005f);
		std::vector<Eigen::Vector3f> balls, colors;
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
		GetBallsAndSticks(points, topo.bones, balls, sticks);
		colors.resize(points.size());
		for (int i = 0; i < points.size(); i++)
		{
			colors[i] = CM[kpt_color_ids[i]];
		}
		BallStickObject* skelObject = new BallStickObject(ballMeshEigen,stickMeshEigen, balls,sticks, 0.005,0.001, colors);
		//BallStickObject* skelObject = new BallStickObject(ballMeshEigen, balls, sizes, colors);
		m_renderer.skels.push_back(skelObject);
	}

	m_renderer.createScene(conf_projectFolder); 
	//m_renderer.createPlane(conf_projectFolder);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		//glPolygonMode(GL_FRONT, GL_FILL);
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}

void main()
{
	test_trajectory();
}