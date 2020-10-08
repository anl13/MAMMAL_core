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

#include "../utils/mesh.h"

#include "test_main.h"
#include "../utils/timer_util.h"

#include "pigmodeldevice.h" 
#include "pigsolverdevice.h"
#include "pigsolver.h"

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

	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	PigSolverDevice smal(smal_config);

	system("pause"); 
	return; 

	smal.SetScale(0.01); 

	PigSolver smalcpu(smal_config); 
	smalcpu.SetScale(0.01); 

	// smal random pose 
	Eigen::VectorXf pose = Eigen::VectorXf::Random(smal.GetJointNum() * 3) * 0.3;
	smal.SetPose(pose);
	smalcpu.SetPose(pose); 
	smal.UpdateVertices();
	smal.UpdateNormalFinal(); 

	smalcpu.UpdateVertices(); 

	Eigen::MatrixXf h_J_joint_cpu, h_J_vert_cpu, h_J_joint_gpu, h_J_vert_gpu;
	int jointNum = smal.GetJointNum(); 
	int vertexNum = smal.GetVertexNum(); 

	h_J_joint_cpu.resize(3 * jointNum, 3 + 3 * jointNum);
	h_J_joint_gpu.resize(3*jointNum, 3+3*jointNum);
	h_J_vert_cpu.resize(3 * vertexNum, 3 + 3 * jointNum); 
	h_J_vert_gpu.resize(3*vertexNum, 3+3*jointNum);

	pcl::gpu::DeviceArray2D<float> d_J_joint, d_J_vert; 
	d_J_joint.create(3 + 3 * jointNum, 3 * jointNum); 
	d_J_vert.create(3 + 3 * jointNum, 3 * vertexNum); 

	smal.calcPoseJacobiFullTheta_device(d_J_joint, d_J_vert); 
	d_J_joint.download(h_J_joint_gpu.data(),3*jointNum * sizeof(float)); 
	d_J_vert.download(h_J_vert_gpu.data(), 3 * vertexNum * sizeof(float)); 

	smal.CalcPoseJacobiFullTheta_cpu(h_J_joint_cpu, h_J_vert_cpu, true);

	Eigen::MatrixXf h_J_joint_2, h_J_vert_2;
	smalcpu.CalcPoseJacobiFullTheta(h_J_joint_2, h_J_vert_2, true); 

	std::cout << "joint norm: " << (h_J_joint_cpu - h_J_joint_gpu).norm() << std::endl; 
	std::cout << "vert  norm: " << (h_J_vert_cpu - h_J_vert_gpu).norm() << std::endl; 
	std::cout << "cpu comp: " << (h_J_joint_cpu - h_J_joint_2).norm() << ", " << (h_J_vert_cpu - h_J_vert_2).norm() << std::endl; 

	RenderObjectColor* animal_model = new RenderObjectColor();
	std::vector<Eigen::Vector3f> vertices_f = smal.GetVertices(); 
	std::vector<Eigen::Vector3u> faces_u = smal.GetFacesVert();
	std::vector<Eigen::Vector3f> normals = smal.GetNormals(); 
	animal_model->SetFaces(faces_u);
	animal_model->SetVertices(vertices_f);
	animal_model->SetNormal(normals); 
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