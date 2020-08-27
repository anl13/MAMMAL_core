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

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareMesh.faces_v_vec);
	chess_floor->SetVertices(squareMesh.vertices_vec);
	chess_floor->SetNormal(squareMesh.normals_vec); 
	chess_floor->SetTexcoords(squareMesh.textures_vec);
	chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

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
	Renderer::s_Init(true);
	Renderer m_renderer(conf_projectFolder + "/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

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
	
	float4 * imgdata; 
	cudaMalloc( (void**)&imgdata, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float4));

	m_renderer.beginOffscreenRender();
	m_renderer.Draw("depth");
	m_renderer.mapRenderingResults();

	TimerUtil::Timer<std::chrono::microseconds> tt; 
	tt.Start(); 
	cudaMemcpy2DFromArray(
		img.data, img.cols * img.elemSize(),
		m_renderer.m_colorArray, 0, 0, img.cols*img.elemSize(),
		img.rows,
		cudaMemcpyDeviceToHost
	);
	std::cout << tt.Elapsed() << std::endl; 
	tt.Start(); 
	cudaMemcpy2DFromArray(imgdata, WINDOW_WIDTH * sizeof(float4), m_renderer.m_colorArray,
		0, 0, WINDOW_WIDTH * sizeof(float4), WINDOW_HEIGHT, cudaMemcpyDeviceToDevice);
	std::cout << tt.Elapsed() << std::endl; 

	cv::Mat depth_pseudo2;
	cv::Mat depth_out; 
	gpupseudo_color(imgdata, 1920, 1080, 2.48504, 0, depth_pseudo2, depth_out);

	//deviceimg.download(img); 
	cv::flip(img, img, 0);
	cv::extractChannel(img, depth, 2);
	cv::Mat depth_pseudo = pseudoColor(depth_out);
	m_renderer.unmapRenderingResults();
	m_renderer.endOffscreenRender();
	m_renderer.releaseResource();

	MeshEigen objeigen(obj);
	cv::Mat mask = drawCVDepth(objeigen.vertices, objeigen.faces, cams[0]);
	cv::Mat blend = blend_images(depth_pseudo, mask, 0.5);

	cv::imshow("depth", depth_pseudo2);
	cv::imshow("mask", mask);
	cv::imshow("blend", blend);
	cv::imwrite("E:/render_test/vis_align.png", depth_pseudo);
	cv::imwrite("E:/render_test/mask_align.png", mask);
	cv::imwrite("E:/render_test/blend_align.png", blend);

	cv::waitKey();
	cv::destroyAllWindows();

	return; 

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