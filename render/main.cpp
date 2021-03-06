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
#include <queue>
#include <deque>

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
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_bgr");

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
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_blend");
	std::vector<Eigen::Vector3f> CM2 = getColorMapEigenF("anliang_render");

	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	//Eigen::Vector3f up; up << 0.f, 0.f, 1.f;
	//Eigen::Vector3f pos; pos << -1.f, 1.5f, 0.8f;
	//Eigen::Vector3f center = Eigen::Vector3f::Zero();
	Eigen::Vector3f up; up << 0.260221, 0.36002, 0.895919;
	Eigen::Vector3f pos; pos << -1.91923, -2.12171, 1.37056;
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
	//std::vector<int> kpt_color_ids = {
	//		9,2,1,2,1, // face 
	//		2,1,2,1,2,1, // front legs 
	//		4,3,4,3,4,3, // back legs 
	//		5,9,5,6,5,5 // ceneter and tail 
	//};

	int start = 750; 
	int num = 2000;
	int window = 50; 

	cv::VideoWriter writer("G:/pig_middle_data/teaser/video/trajectory1.avi", cv::VideoWriter::fourcc('m', 'p', 'e', 'g'), 25.0, cv::Size(1920, 1080));
	if (!writer.isOpened())
	{
		std::cout << "not open" << std::endl;
		return; 
	}

	std::vector<Eigen::Vector2i> bones = {
	{0,1}, {0,2}, {1,2}, {1,3}, {2,4},
	 {5,7}, {7,9}, {6,8}, {8,10},
	{20,18},
	{18,11}, {18,12}, {11,13}, {13,15}, {12,14}, {14,16},
	{0,20},{5,20},{6,20}
	};
	std::vector<int> kpt_color_ids = {
		0,1,1,2,2,
		3,4,3,4,3,4,
		5,6,5,6,5,6,
		0,7, 0,7,0,0
	};
	std::vector<int> bone_color_ids = {
		1,2,0,1,2,3,3,4,4,
		7,5,6,5,5,6,6,
		7,3,4
	};

	std::vector<std::deque<std::vector<Eigen::Vector3f> > > joints_queues;
	joints_queues.resize(4); 
	for (int frameid = start; frameid < start + num; frameid++)
	{
		std::cout << frameid << std::endl; 
		// push to queue 
		for (int pid = 0; pid < 4; pid++)
		{
			std::stringstream ss;
			ss << "F:/pig_results_anchor_sil/joints/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
			std::string point_file = ss.str();
			std::vector<Eigen::Vector3f> points = read_points(point_file);
			joints_queues[pid].push_back(points);
			if (joints_queues[pid].size() > window) joints_queues[pid].pop_front(); 
		}

		m_renderer.clearAllObjs(); 

#if 1 // trajectory type1 
		for (int index = 0; index < joints_queues[0].size(); index++)
		{
			for (int pid = 0; pid < 4; pid++)
			{
				int ratio_index = window - joints_queues[0].size() + index;
				float ratio = (2 - (ratio_index / float(window)));

				std::vector<Eigen::Vector3f> skels = joints_queues[pid][index];
				std::vector<Eigen::Vector3f> balls;
				std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
				GetBallsAndSticks(skels, bones, balls, sticks);
				int jointnum = skels.size();
				std::vector<float> ball_sizes;
				ball_sizes.resize(jointnum, 0.005/ratio);
				std::vector<float> stick_sizes;
				stick_sizes.resize(sticks.size(), 0.002/ratio);
				std::vector<Eigen::Vector3f> ball_colors(jointnum);
				std::vector<Eigen::Vector3f> stick_colors(sticks.size());
				for (int i = 0; i < jointnum; i++)
				{
					ball_colors[i] = CM[kpt_color_ids[i]] * ratio;
				}
				for (int i = 0; i < sticks.size(); i++)
				{
					//stick_colors[i] = CM[bone_color_ids[i]] * ratio;
					stick_colors[i] = CM2[pid] * ratio;
				}

				BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
					balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
				p_skel->isMultiLight = false; 
				m_renderer.skels.push_back(p_skel);

			}
		}
#else 
		for (int index = 0; index < joints_queues[0].size(); index++)
		{
			if (index == joints_queues[0].size() - 1)
			{
				for (int pid = 0; pid < 4; pid++)
				{
					int ratio_index = window - joints_queues[0].size() + index; 
					float ratio = (2 - (ratio_index / float(window)));

					std::vector<Eigen::Vector3f> skels = joints_queues[pid][index];
					std::vector<Eigen::Vector3f> balls;
					std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
					GetBallsAndSticks(skels, bones, balls, sticks);
					int jointnum = skels.size();
					std::vector<float> ball_sizes;
					ball_sizes.resize(jointnum, 0.015);
					std::vector<float> stick_sizes;
					stick_sizes.resize(sticks.size(), 0.009);
					std::vector<Eigen::Vector3f> ball_colors(jointnum);
					std::vector<Eigen::Vector3f> stick_colors(sticks.size());
					for (int i = 0; i < jointnum; i++)
					{
						ball_colors[i] = CM[kpt_color_ids[i]] * ratio;
					}
					for (int i = 0; i < sticks.size(); i++)
					{
						//stick_colors[i] = CM[bone_color_ids[i]] * ratio;
						stick_colors[i] = CM2[pid] * ratio;
					}

					BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
						balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
					p_skel->isMultiLight = false; 
					m_renderer.skels.push_back(p_skel);
				}
			}
			else
			{
				for (int pid = 0; pid < 4; pid++)
				{
					float ratio = (2 - (index / float(window)));

					std::vector<Eigen::Vector3f> skels = joints_queues[pid][index];
					std::vector<Eigen::Vector3f> balls;
					std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
					GetBallsAndSticks(skels, bones, balls, sticks);
					int jointnum = skels.size();
					std::vector<float> ball_sizes;
					ball_sizes.resize(jointnum, 0.002);

					// sticks: connect last to current
					sticks.clear(); 
					sticks.resize(jointnum); 
					std::vector<float> stick_sizes;
					for (int k = 0; k < jointnum; k++)
					{
						sticks[k].first = joints_queues[pid][index][k];
						sticks[k].second = joints_queues[pid][index + 1][k];
					}
					stick_sizes.resize(sticks.size(), 0.001);
					std::vector<Eigen::Vector3f> ball_colors(jointnum);
					std::vector<Eigen::Vector3f> stick_colors(sticks.size());
					for (int i = 0; i < jointnum; i++)
					{
						ball_colors[i] = CM[kpt_color_ids[i]] * ratio;
					}
					for (int i = 0; i < sticks.size(); i++)
					{
						stick_colors[i] = CM[kpt_color_ids[i]] * ratio;
					}

					BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
						balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
					m_renderer.skels.push_back(p_skel);
				}
			}
		}
#endif 
		m_renderer.createScene(conf_projectFolder);

		cv::Mat img = m_renderer.GetImageOffscreen();
		writer.write(img);

		//std::stringstream output_ss; 
		//output_ss << "G:/pig_middle_data/teaser/video/trajectory_" << std::setw(6) << std::setfill('0') << frameid << ".png"; 
		//cv::imwrite(output_ss.str(), img);
		//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		//while (!glfwWindowShouldClose(windowPtr))
		//{
		//	//glPolygonMode(GL_FRONT, GL_FILL);
		//	m_renderer.Draw();

		//	glfwSwapBuffers(windowPtr);
		//	glfwPollEvents();
		//};
	}
	writer.release(); 
}

void generateFaceIndexMap()
{
	Mesh obj;
	obj.Load("D:/Projects/animal_calib/data/artist_model_sym3/manual_artist_sym.obj");
	obj.ReMapTexture(); 
	
	std::cout << "face v: " << obj.faces_v_vec.size() << std::endl; 
	std::cout << "face t: " << obj.faces_t_vec.size() << std::endl;

	cv::Mat img;
	img.create(cv::Size(2048, 2048), CV_8UC3); 
	img = img * 0;
	for (int findex = 0; findex < obj.faces_v_vec.size(); findex++)
	{
		int t1 = obj.faces_t_vec[findex](0);
		int t2 = obj.faces_t_vec[findex](1);
		int t3 = obj.faces_t_vec[findex](2);
		Eigen::Vector2f p1 = obj.textures_vec[t1] * 2048;
		Eigen::Vector2f p2 = obj.textures_vec[t2] * 2048;
		Eigen::Vector2f p3 = obj.textures_vec[t3] * 2048;
		cv::Point2i cvp1(round(p1(0)), round(p1(1)));
		cv::Point2i cvp2(round(p2(0)), round(p2(1)));
		cv::Point2i cvp3(round(p3(0)), round(p3(1)));
		std::vector<std::vector<cv::Point2i> > contour; 
		contour.resize(1); 
		contour[0].push_back(cvp1);
		contour[0].push_back(cvp2); 
		contour[0].push_back(cvp3);
		int b = findex / (32 * 32);
		int remain = findex - b * 32 * 32; 
		int g = remain / 32;
		int r = remain % 32;
		b = b * 8;
		g = g * 8;
		r = r * 8;

		cv::Scalar color(b, g, r);
		cv::fillPoly(img, contour, color, 1, 0);
	}
	cv::imwrite("D:/Projects/animal_calib/data/artist_model_sym3/face_index_texture.png", img); 
	//for (int x = 0; x < 2048; x++)
	//{
	//	for (int y = 0; y < 2048; y++)
	//	{
	//		float xf = x / 2048;
	//		float yf = y / 2048; 

	//	}
	//}
	
}

int color2faceid(const cv::Vec3b& c)
{
	int b = round(float(c[2]) / 8);
	int g = round(float(c[1]) / 8);
	int r = round(float(c[0]) / 8);
	int faceid = (b << 5 + g) << 5 + r;
	return faceid;
}

void test_indexrender()
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

	Mesh obj;
	obj.Load("D:/Projects/animal_calib/data/artist_model_sym3/manual_artist_sym.obj");
	for (int i = 0; i < obj.vertices_vec.size(); i++)
	{
		obj.vertices_vec[i] += Eigen::Vector3f(0, 0, 0.21);
	}

	obj.ReMapTexture();

	RenderObjectTexture* p_model = new RenderObjectTexture();
	p_model->SetTexture("D:/Projects/animal_calib/data/artist_model_sym3/face_index_texture.png");
	p_model->SetFaces(obj.faces_t_vec);
	p_model->SetVertices(obj.vertices_vec_t);
	p_model->SetNormal(obj.normals_vec_t, 2);
	p_model->SetTexcoords(obj.textures_vec, 1);
	p_model->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	p_model->isMultiLight = false;
	p_model->isFaceIndex = true; 
	m_renderer.texObjs.push_back(p_model);

	//m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));

	//m_renderer.createScene(conf_projectFolder); 
	//m_renderer.createPlane(conf_projectFolder);

	m_renderer.Draw(); 
	cv::Mat img = m_renderer.GetImage(); 
	int totalcount = 0;
	int correct = 0;
	for (int x = 0; x < 1920; x++)
	{
		for (int y = 0; y < 1080; y++)
		{
			cv::Vec3b pixel= img.at<cv::Vec3b>(y, x);
			int b = pixel[0];
			int g = pixel[1];
			int r = pixel[2];
			if (b > 0 || g > 0 || r > 0)
			{
				totalcount++;
				if (b % 8 != 0 || g % 8 != 0 || r % 8 != 0)
				{
					std::cout << "(b,g,r)=(" << b << "," << g << "," << r << ")" << std::endl;
				}
				else correct++;
			}
		}
	}
	std::cout << "total  : " << totalcount << std::endl; 
	std::cout << "correct: " << correct << std::endl;

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

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
	//test_trajectory();
	test_color_table();
	//test_indexrender(); 
	//generateFaceIndexMap();

}