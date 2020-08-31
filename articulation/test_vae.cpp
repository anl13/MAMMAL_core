
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
#include "../utils/mesh.h"

#include "test_main.h"



/// 20200801: pass numeric test
int test_vae()
{
	// render config 
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");
	std::vector<Camera> cams = readCameras();

	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
	std::cout << K << std::endl;

	Eigen::Vector3f pos = Eigen::Vector3f(0.873302, -0.961363, 0.444287);
	Eigen::Vector3f up = Eigen::Vector3f(-0.220837, 0.236654, 0.946164);
	Eigen::Vector3f center = Eigen::Vector3f::Zero();
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

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareMesh.faces_v_vec);
	chess_floor->SetVertices(squareMesh.vertices_vec);
	chess_floor->SetNormal(squareMesh.normals_vec, 2);
	chess_floor->SetTexcoords(squareMesh.textures_vec, 1);
	chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

	std::string pig_config = "D:/Projects/animal_calib/articulation/artist_config.json";

	PigModel gtpig(pig_config);
	gtpig.setIsLatent(false);
	Eigen::VectorXf pose = Eigen::VectorXf::Random(62 * 3) * 0.1;
	gtpig.SetPose(pose);
	gtpig.UpdateVertices();
	gtpig.SaveObj("G:/debug_pig4/poseiter/gt.obj");

	PigSolver pig(pig_config);
	pig.m_targetVSameTopo = gtpig.GetVertices();
	pig.FitPoseToVerticesSameTopoLatent();
	pig.SaveObj("G:/debug_pig4/poseiter/estimation.obj");

	RenderObjectColor* animal_model = new RenderObjectColor();
	Eigen::MatrixXf vertices_f = pig.GetVertices().cast<float>();
	vertices_f = vertices_f.colwise() + Eigen::Vector3f(0, 0, 0.21f);

	Eigen::MatrixXu faces_u = pig.GetFacesVert();
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

	return 0;
}