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


void test_inmeshtest()
{
	// render config 
	std::cout << "In render scene now!" << std::endl;

	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM3 = getColorMapEigenF("anliang_paper");
	std::vector<Camera> cams = readCameras();

	// init a camera 
	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	//Eigen::Vector3f up; up << -0.289519, -0.293115, 0.911188;
	//Eigen::Vector3f pos; pos << 0.78681, 0.706331, 0.402439;
	//Eigen::Vector3f center; center << 0.131863, 0.0613784, -0.0129145;

	Eigen::Vector3f pos(0.319969, 1.25653, 0.708613); 
	Eigen::Vector3f up(-0.0388474, -0.505415, 0.862002);
	Eigen::Vector3f center(0.131863, 0.0613784, -0.0129145); 

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
	PigSolverDevice smal(smal_config);
	std::string state_file = "D:/results/paper_teaser/0704_demo/state/pig_1_frame_007888.txt"; 
	smal.readState(state_file); 
	smal.UpdateVertices();
	smal.UpdateNormalFinal();
	smal.map_reduced_vertices(); 

	std::vector<Eigen::Vector3f> colors_mesh(smal.m_reduced_vertices.size(), Eigen::Vector3f(0, 0, 0));


	//// smal random pose 
	RenderObjectMesh * animal_model = new RenderObjectMesh();
	animal_model->SetFaces(smal.m_reduced_faces);
	animal_model->SetVertices(smal.m_reduced_vertices);
	animal_model->SetNormal(smal.m_reduced_normals);
	animal_model->SetColors(colors_mesh);
	animal_model->isMultiLight = false;
	animal_model->isFill = false;
	m_renderer.meshObjs.push_back(animal_model);
	
#if 0 // 2021.10.11 demo video to show how we do collision detection. 
	std::vector<Eigen::Vector3f> skels;
	for (int i = 0; i < 500; i++)
	{
		Eigen::Vector3f a = Eigen::Vector3f::Random() / 18;
		a(0) += 0.5;
		a(2) -= 0.03;
		a(2) *= 3;
		skels.push_back(a);
	}

	cv::VideoWriter writer("D:/video_book/inmeshtest.mp4", cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 25, cv::Size(1920, 1080));
	for (int frame = 0; frame < 125; frame++)
	{
		m_renderer.skels.clear(); 
		std::vector<Eigen::Vector3f> balls = skels;
		for (int k = 0; k < 500; k++)
			skels[k](0) -= (0.7 / 125); 
		std::vector<float> ball_sizes(balls.size(), 0.007);

		std::vector<Eigen::Vector3f> ball_colors(balls.size(), CM3[0]);
		for (int i = 0; i < balls.size(); i++)
		{
			bool is_in;
			auto mesh_vertices = smal.m_reduced_vertices;
			auto mesh_faces = smal.m_reduced_faces;
			Eigen::Vector3f a = Eigen::Vector3f(0, 0, 0);
			Eigen::Vector3f b = balls[i];

			is_in = inMeshTest_cpu(mesh_vertices, mesh_faces, a, balls[i]);

			if (is_in) {
				ball_colors[i] = CM3[1];
			}
			else ball_colors[i] = CM3[0];
		}

		BallStickObject* p_skel = new BallStickObject(ballMeshEigen,
			balls, ball_sizes, ball_colors);
		m_renderer.skels.push_back(p_skel);
		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));

		m_renderer.Draw();
		cv::Mat img = m_renderer.GetImageOffscreen();
		//cv::imwrite("E:/pig_middle_data/picture_model/skel.png", img);
		writer.write(img);

		//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
		//while (!glfwWindowShouldClose(windowPtr))
		//{
		//	m_renderer.Draw();
		//	glfwSwapBuffers(windowPtr);
		//	glfwPollEvents();
		//};
	}
	writer.release(); 
#endif 
}
