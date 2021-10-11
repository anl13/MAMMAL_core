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

void add_graph(PigSolverDevice& smal, Renderer& m_renderer, const MeshEigen& ballMeshEigen, const MeshEigen& stickMeshEigen)
{
	std::vector<Eigen::Vector3u> reduced_faces = smal.m_reduced_faces;
	std::vector<Eigen::Vector3f> reduced_verts = smal.m_reduced_vertices;
	std::vector<Eigen::Vector3f> balls;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	std::vector<Eigen::Vector2i> bones;
	for (int i = 0; i < reduced_faces.size(); i++)
	{
		int a = reduced_faces[i](0);
		int b = reduced_faces[i](1);
		int c = reduced_faces[i](2);
		std::vector<Eigen::Vector2i> lines = {
			{a,b}, {b,c}, {c,a}
		};
		for (int k = 0; k < 3; k++)
		{
			int valid = true;
			for (int m = 0; m < bones.size(); m++)
			{
				if ((bones[m](0) == lines[k](0) && bones[m](1) == lines[k](1))
					|| (bones[m](0) == lines[k](1) && bones[m](1) == lines[k](0)))
				{
					valid = false;
					break;
				}
			}
			if (valid) bones.push_back(lines[k]);
		}
	}
	GetBallsAndSticks(reduced_verts, bones, balls, sticks);
	sticks.clear(); 
	std::vector<float> ball_sizes;
	ball_sizes.resize(reduced_verts.size(), 0.003);
	std::vector<float> stick_sizes;
	stick_sizes.resize(sticks.size(), 0.001);
	std::vector<Eigen::Vector3f> ball_colors(reduced_verts.size(), Eigen::Vector3f(0.3, 0.3, 1));
	std::vector<Eigen::Vector3f> stick_colors(sticks.size(), Eigen::Vector3f(0.3, 0.3, 0.3));

	BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
		balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
	p_skel->isFill = true; 
	p_skel->isMultiLight = false; 
	m_renderer.skels.push_back(p_skel);

}

// 2021.10.11 this function is used to render collision state of two pigs 
void test_collision_render()
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

	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);

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
	std::string state_file = "D:/results/paper_teaser/0704_demo/debug/anchor_1.txt";
	smal.readState(state_file);
	smal.UpdateVertices();
	smal.UpdateNormalFinal();
	smal.map_reduced_vertices();

	PigSolverDevice smal2(smal_config); 
	std::string state_file2 = "D:/results/paper_teaser/0704_demo/debug/anchor_3.txt";
	smal2.readState(state_file2);
	smal2.UpdateVertices();
	smal2.UpdateNormalFinal();
	smal2.map_reduced_vertices();



	//// smal random pose 
	//RenderObjectColor * animal_model = new RenderObjectColor();
	//animal_model->SetFaces(smal.GetFacesVert());
	//animal_model->SetVertices(smal.GetVertices());
	//animal_model->SetNormal(smal.GetNormals());
	//animal_model->SetColor(Eigen::Vector3f(0.9,0.9,0.9));
	//animal_model->isMultiLight = false;
	//animal_model->isFill = true;
	//m_renderer.colorObjs.push_back(animal_model);

	RenderObjectColor * animal_model2 = new RenderObjectColor();
	animal_model2->SetFaces(smal2.GetFacesVert());
	animal_model2->SetVertices(smal2.GetVertices());
	animal_model2->SetNormal(smal2.GetNormals());
	animal_model2->SetColor(Eigen::Vector3f(0.9, 0.9, 0.9));
	animal_model2->isMultiLight = false;
	animal_model2->isFill = true;
	m_renderer.colorObjs.push_back(animal_model2);


	//add_graph(smal, m_renderer, ballMeshEigen, stickMeshEigen);
	//std::cout << "add graph done. " << std::endl;
	//add_graph(smal2, m_renderer, ballMeshEigen, stickMeshEigen);
	//std::cout << "add graph done. " << std::endl;



#if 0 // 2021.10.11 demo video to show how we do collision detection. 
	std::vector<Eigen::Vector3u> reduced_faces = smal.m_reduced_faces;
	std::vector<Eigen::Vector3f> reduced_verts = smal.m_reduced_vertices;
	std::vector<Eigen::Vector3f> balls;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	std::vector<Eigen::Vector2i> bones;
	for (int i = 0; i < reduced_faces.size(); i++)
	{
		int a = reduced_faces[i](0);
		int b = reduced_faces[i](1);
		int c = reduced_faces[i](2);
		std::vector<Eigen::Vector2i> lines = {
			{a,b}, {b,c}, {c,a}
		};
		for (int k = 0; k < 3; k++)
		{
			int valid = true;
			for (int m = 0; m < bones.size(); m++)
			{
				if ((bones[m](0) == lines[k](0) && bones[m](1) == lines[k](1))
					|| (bones[m](0) == lines[k](1) && bones[m](1) == lines[k](0)))
				{
					valid = false;
					break;
				}
			}
			if (valid) bones.push_back(lines[k]);
		}
	}
	GetBallsAndSticks(reduced_verts, bones, balls, sticks);
	sticks.clear();
	std::vector<float> ball_sizes;
	ball_sizes.resize(reduced_verts.size(), 0.003);
	std::vector<float> stick_sizes;
	stick_sizes.resize(sticks.size(), 0.001);
	std::vector<Eigen::Vector3f> ball_colors(reduced_verts.size(), Eigen::Vector3f(0.3, 0.3, 1));
	for (int i = 0; i < balls.size(); i++)
	{
		bool is_in;
		auto mesh_vertices = smal.m_reduced_vertices;
		auto mesh_faces = smal.m_reduced_faces;
		Eigen::Vector3f a = Eigen::Vector3f(0, 0, 0);
		Eigen::Vector3f b = balls[i];

		is_in = inMeshTest_cpu(smal2.m_reduced_vertices, mesh_faces, a, balls[i]);

		if (is_in) {
			ball_colors[i] = Eigen::Vector3f(0,0,1);
		}
		else ball_colors[i] = Eigen::Vector3f(1,0,0);
	}
	std::vector<Eigen::Vector3f> stick_colors(sticks.size(), Eigen::Vector3f(0.3, 0.3, 0.3));

	BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
		balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
	p_skel->isFill = true;
	p_skel->isMultiLight = false;
	m_renderer.skels.push_back(p_skel);

#endif 
	m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1)); 
	m_renderer.s_camViewer.SetExtrinsic(cams[8].R, cams[8].T + Eigen::Vector3f(0, 0, 0.125));

	glEnable(GL_BLEND);
	//glDisable(GL_DEPTH_TEST);
	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	m_renderer.Draw();

	cv::Mat img = m_renderer.GetImageOffscreen();
	cv::imwrite("D:/results/paper_teaser/0704_demo/render_all/collision3.png", img);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}
