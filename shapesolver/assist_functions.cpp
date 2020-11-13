#include "assist_functions.h"
#include <vector>
#include <string> 
#include <fstream> 

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "shapesolver.h"
#include "../posesolver/framedata.h" 
#include "../posesolver/framesolver.h"
#include "../utils/show_gpu_param.h"

#include "../utils/Hungarian.h" 

void compute_procrutes()
{
	std::vector<int> flip_index = {
		0,1,2,3,4,13,14,15,16,17,18,19,20,5,6,7,8,9,10,11,12,21,22,23,25,24,31,32,33,34,35,
		26,27,28,29,30,36,37,54,55,56,57,58,59,60,61,46,47,48,49,50,51,52,53,
		38,39,40,41,42,43,44,45
	};

	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config.json";
	ShapeSolver solver(pig_config);

	Eigen::MatrixXf joints = solver.GetJoints();
	for (int i = 0; i < 62; i++)
	{
		std::cout << std::setw(2) << i << " : " << joints(1, i) << ", " <<
			joints(1, flip_index[i]) << std::endl;
	}

	Mesh rawmodel;
	rawmodel.Load("D:/Projects/animal_calib/data/artist_model/model_triangle.obj"); 

	MeshEigen symmesh(rawmodel); 
	symmesh.vertices = solver.GetVertices();
	symmesh.faces = solver.GetFacesVert();
	symmesh.CalcNormal();

	Eigen::Vector3f mean;
	mean(0) = symmesh.vertices.row(0).mean();
	mean(1) = symmesh.vertices.row(1).mean();
	mean(2) = symmesh.vertices.row(2).mean();
	symmesh.vertices = symmesh.vertices.colwise() - mean;



	std::cout << "mean: " << mean.transpose() << std::endl; 

	
	MeshEigen symmesh2(rawmodel);
	Eigen::Vector3f mean2;
	mean2(0) = symmesh2.vertices.row(0).mean(); 
	mean2(1) = symmesh2.vertices.row(1).mean(); 
	mean2(2) = symmesh2.vertices.row(2).mean(); 
	symmesh2.vertices = symmesh2.vertices.colwise() - mean2; 

	float totaldist1 = 0;
	float totaldist2 = 0;
	for (int i = 0; i < symmesh.faces.cols(); i++)
	{
		int a = symmesh.faces(0, i);
		int b = symmesh.faces(1, i);
		int c = symmesh.faces(2, i);
		totaldist1 += (symmesh.vertices.col(a) - symmesh.vertices.col(b)).norm(); 
		totaldist1 += (symmesh.vertices.col(a) - symmesh.vertices.col(c)).norm();
		totaldist1 += (symmesh.vertices.col(b) - symmesh.vertices.col(c)).norm();
	}

	for (int i = 0; i < symmesh2.faces.cols(); i++)
	{
		int a = symmesh2.faces(0, i);
		int b = symmesh2.faces(1, i);
		int c = symmesh2.faces(2, i);
		totaldist2 += (symmesh2.vertices.col(a) - symmesh2.vertices.col(b)).norm();
		totaldist2 += (symmesh2.vertices.col(a) - symmesh2.vertices.col(c)).norm();
		totaldist2 += (symmesh2.vertices.col(b) - symmesh2.vertices.col(c)).norm();
	}

	//float scale = totaldist1 / totaldist2; 
	float scale = 0.405927;
	symmesh2.vertices *= scale; 

	//Eigen::Matrix3f S = symmesh2.vertices * symmesh.vertices.transpose(); 
	//Eigen::JacobiSVD<Eigen::Matrix3f> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
	//Eigen::MatrixXf U = svd.matrixU();
	//Eigen::MatrixXf V = svd.matrixV();
	//Eigen::Matrix3f R = V * U.transpose();
	Eigen::Matrix3f R;
	R << 0, 0, 1, 1, 0, 0, 0, 1, 0;
	symmesh2.vertices = R * symmesh2.vertices; 

	//symmesh2.vertices.row(0) = -symmesh.vertices.row(0); 
	//symmesh2.faces.row(0) = symmesh.faces.row(1); 
	//symmesh2.faces.row(1) = symmesh.faces.row(0); 
	//symmesh2.CalcNormal();

	std::cout << "R: " << std::endl << R << std::endl; 
	std::cout << "scale: " << scale << std::endl; 


	// init renderer
	Camera cam = Camera::getDefaultCameraUndist();
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(false);
	Renderer m_renderer("D:/Projects/animal_calib/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));



	RenderObjectColor* p_model = new RenderObjectColor();
	solver.UpdateNormalFinal();
	p_model->SetVertices(symmesh.vertices);
	p_model->SetNormal(symmesh.normals);
	p_model->SetFaces(symmesh.faces);

	//p_model->SetVertices(solver.GetVertices()); 
	//p_model->SetNormal(solver.GetNormals());
	//p_model->SetFaces(solver.GetFacesVert()); 
	p_model->SetColor(CM[0]);


	RenderObjectColor* p_model2 = new RenderObjectColor();
	p_model2->SetVertices(symmesh2.vertices); 
	p_model2->SetFaces(symmesh2.faces); 
	p_model2->SetNormal(symmesh2.normals); 
	p_model2->SetColor(CM[1]); 

	m_renderer.clearAllObjs();
	m_renderer.createScene("D:/Projects/animal_calib/");
	m_renderer.colorObjs.push_back(p_model);
	m_renderer.colorObjs.push_back(p_model2); 

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}

void compute_symmetry()
{
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");
	std::string conf_projectFolder = "D:/Projects/animal_calib/"; 

	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config.json";
	ShapeSolver solver(pig_config);

	Mesh rawmodel;
	rawmodel.Load("D:/Projects/animal_calib/data/artist_model/model_triangle.obj");

	MeshEigen symmesh(rawmodel);
	//float scale = 0.405927;
	float scale = 1; 
	symmesh.vertices = symmesh.vertices * scale; 
	Eigen::Matrix3f R; 
	R << 0, 0, 1, 1, 0, 0, 0, 1, 0;
	symmesh.vertices = R * symmesh.vertices; 

	Eigen::Vector3f mean;
	mean(0) = symmesh.vertices.row(0).mean();
	mean(1) = symmesh.vertices.row(1).mean();
	mean(2) = symmesh.vertices.row(2).mean();
	symmesh.vertices = symmesh.vertices.colwise() - mean;
	symmesh.CalcNormal(); 

	std::cout << "mean: " << mean.transpose() << std::endl;

	MeshEigen symmesh2 = symmesh;
	Eigen::Vector3f mean2;
	mean2(0) = symmesh2.vertices.row(0).mean();
	mean2(1) = symmesh2.vertices.row(1).mean();
	mean2(2) = symmesh2.vertices.row(2).mean();
	symmesh2.vertices = symmesh2.vertices.colwise() - mean2;
	std::cout << "mean2: " << mean2.transpose() << std::endl; 

	symmesh2.vertices.row(1) = -symmesh.vertices.row(1); 
	symmesh2.faces.row(0) = symmesh.faces.row(1); 
	symmesh2.faces.row(1) = symmesh.faces.row(0); 
	symmesh2.CalcNormal();

	std::cout << "R: " << std::endl << R << std::endl;
	std::cout << "scale: " << scale << std::endl;

	int VN = symmesh.vertices.cols(); 
	Eigen::MatrixXf E = Eigen::MatrixXf::Zero(VN, VN); 
	for (int i = 0; i < VN; i++)
	{
		for (int j = 0; j < VN; j++)
		{
			Eigen::Vector3f diff = symmesh.vertices.col(i) - symmesh2.vertices.col(j);
			E(i, j) = diff.norm(); 
		}
	}
	std::vector<int> sym = solveHungarian(E); 
	std::ofstream outfile("D:/Projects/animal_calib/data/artist_model_sym/sym.txt"); 
	for (int i = 0; i < sym.size(); i++)
	{
		outfile << sym[i] << std::endl; 
	}
	outfile.close(); 
	

	// init renderer
	Camera cam = Camera::getDefaultCameraUndist();
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(false);
	Renderer m_renderer("D:/Projects/animal_calib/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));


	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	int N = symmesh.vertices.cols(); 
	std::vector<float> sizes(N, 0.003f);
	std::vector<Eigen::Vector3f> balls, colors;
	for (int i = 0; i < symmesh.vertices.cols(); i++)balls.push_back(symmesh.vertices.col(i));

	colors.resize(balls.size());
	for (int i = 0; i < colors.size(); i++)
	{
		colors[i] = CM[0];
	}
	BallStickObject* skelObject = new BallStickObject(ballMeshEigen, balls, sizes, colors);
	m_renderer.skels.push_back(skelObject);

	std::vector<Eigen::Vector3f> balls2, colors2;
	for (int i = 0; i < symmesh2.vertices.cols(); i++)
		balls2.push_back(symmesh2.vertices.col(i));
	colors2.resize(balls2.size()); 
	for (int i = 0; i < colors2.size(); i++)
		colors2[i] = CM[1]; 
	BallStickObject* skelObject2 = new BallStickObject(ballMeshEigen, balls2, sizes, colors2);
	m_renderer.skels.push_back(skelObject2); 

	//RenderObjectColor* p_model = new RenderObjectColor();
	//solver.UpdateNormalFinal();
	//p_model->SetVertices(symmesh.vertices);
	//p_model->SetNormal(symmesh.normals);
	//p_model->SetFaces(symmesh.faces);
	//p_model->SetColor(CM[0]);


	//RenderObjectColor* p_model2 = new RenderObjectColor();
	//p_model2->SetVertices(symmesh2.vertices);
	//p_model2->SetFaces(symmesh2.faces);
	//p_model2->SetNormal(symmesh2.normals);
	//p_model2->SetColor(CM[1]);

	m_renderer.createScene("D:/Projects/animal_calib/");
	//m_renderer.colorObjs.push_back(p_model);
	//m_renderer.colorObjs.push_back(p_model2);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}

void flip_symmetry()
{
	std::vector<int> flip_index = {
	0,1,2,3,4,13,14,15,16,17,18,19,20,5,6,7,8,9,10,11,12,21,22,23,25,24,31,32,33,34,35,
	26,27,28,29,30,36,37,54,55,56,57,58,59,60,61,46,47,48,49,50,51,52,53,
	38,39,40,41,42,43,44,45
	};

	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");
	std::string conf_projectFolder = "D:/Projects/animal_calib/";

	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config.json";
	PigModel solver(pig_config);
	
	std::cout << solver.m_jointsOrigin.transpose() << std::endl; 
	std::cout << std::endl; 
	std::cout << "pose: " << std::endl << solver.m_poseParam << std::endl; 

	std::cout << "final: " << std::endl << solver.m_jointsFinal.transpose() << std::endl; 

	std::cout << "deform: " << std::endl << solver.m_jointsDeformed.transpose() << std::endl; 
	std::cout << "scaled: " << std::endl << solver.m_jointsShaped.transpose() << std::endl; 

	MeshEigen symmesh;
	symmesh.vertices = solver.GetVertices(); 
	symmesh.faces = solver.GetFacesVert(); 
	symmesh.CalcNormal(); 

	std::vector<int> sym; 
	std::ifstream infile("D:/Projects/animal_calib/data/artist_model_sym/sym.txt"); 
	int VN = symmesh.vertices.cols(); 
	sym.resize(VN); 
	for (int i = 0; i < VN; i++) infile >> sym[i];
	
	MeshEigen symmesh2 = symmesh; 
	for (int i = 0; i < VN; i++)
	{
		Eigen::Vector3f flipped = symmesh.vertices.col(sym[i]); 
		flipped(1) *= -1;
		symmesh2.vertices.col(i) = (symmesh.vertices.col(i) + flipped) / 2;
	}
	std::ofstream outfile("D:/Projects/animal_calib/data/artist_model_sym/vertices.txt"); 
	outfile << symmesh2.vertices.transpose();
	outfile.close(); 

	int JN = solver.GetJointNum(); 
	Eigen::MatrixXf joints = Eigen::MatrixXf::Zero(3, JN); 
	Eigen::MatrixXf rawjoints = solver.GetJoints(); 
	std::cout << rawjoints.transpose() << std::endl; 
	for (int i = 0; i < JN; i++)
	{
		Eigen::Vector3f flipped = rawjoints.col(flip_index[i]); 
		flipped(1) *= -1;
		joints.col(i) = (
			rawjoints.col(i) + flipped
			) / 2;
	}
	std::ofstream outfilejoint("D:/Projects/animal_calib/data/artist_model_sym/t_pose_joints.txt");
	outfilejoint << joints.transpose(); 
	outfilejoint.close(); 

	// init renderer
	Camera cam = Camera::getDefaultCameraUndist();
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(false);
	Renderer m_renderer("D:/Projects/animal_calib/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));


	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	RenderObjectColor* p_model = new RenderObjectColor();
	solver.UpdateNormalFinal();
	p_model->SetVertices(symmesh.vertices);
	p_model->SetNormal(symmesh.normals);
	p_model->SetFaces(symmesh.faces);
	p_model->SetColor(CM[0]);


	RenderObjectColor* p_model2 = new RenderObjectColor();
	p_model2->SetVertices(symmesh2.vertices);
	p_model2->SetFaces(symmesh2.faces);
	p_model2->SetNormal(symmesh2.normals);
	p_model2->SetColor(CM[1]);

	m_renderer.createScene("D:/Projects/animal_calib/");
	m_renderer.colorObjs.push_back(p_model);
	m_renderer.colorObjs.push_back(p_model2);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}


void test_symmetry()
{

	std::vector<int> flip_index = {
0,1,2,3,4,13,14,15,16,17,18,19,20,5,6,7,8,9,10,11,12,21,22,23,25,24,31,32,33,34,35,
26,27,28,29,30,36,37,54,55,56,57,58,59,60,61,46,47,48,49,50,51,52,53,
38,39,40,41,42,43,44,45
	};

	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");
	std::string conf_projectFolder = "D:/Projects/animal_calib/";

	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config.json";
	PigModel solver(pig_config);
	Eigen::MatrixXf regressedjoints = solver.m_verticesOrigin * solver.m_jregressor;

	std::cout << regressedjoints.transpose() << std::endl; 
	std::cout << regressedjoints.row(1).mean() << std::endl; 

	MeshEigen symmesh;
	symmesh.vertices = solver.GetVertices();
	symmesh.faces = solver.GetFacesVert();
	symmesh.CalcNormal();

	std::vector<int> sym;
	std::ifstream infile("D:/Projects/animal_calib/data/artist_model_sym/sym.txt");
	int VN = symmesh.vertices.cols();
	sym.resize(VN);
	for (int i = 0; i < VN; i++) infile >> sym[i];
	infile.close(); 


	// init renderer
	Camera cam = Camera::getDefaultCameraUndist();
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(false);
	Renderer m_renderer("D:/Projects/animal_calib/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));


	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	RenderObjectColor* p_model = new RenderObjectColor();
	solver.UpdateNormalFinal();
	p_model->SetVertices(symmesh.vertices);
	p_model->SetNormal(symmesh.normals);
	p_model->SetFaces(symmesh.faces);
	p_model->SetColor(CM[0]);


	//RenderObjectColor* p_model2 = new RenderObjectColor();
	//p_model2->SetVertices(symmesh2.vertices);
	//p_model2->SetFaces(symmesh2.faces);
	//p_model2->SetNormal(symmesh2.normals);
	//p_model2->SetColor(CM[1]);

	std::vector<Eigen::Vector3f> balls;
	std::vector<Eigen::Vector3f> colors;
	std::vector<float> sizes; 
	for (int i = 0; i < regressedjoints.cols(); i++)
	{
		balls.push_back(regressedjoints.col(i)); 
		sizes.push_back(0.01); 
		colors.push_back(CM[2]);
	}
	BallStickObject* joints_rend = new BallStickObject(
		ballMeshEigen, balls, sizes, colors); 
	

	m_renderer.createScene("D:/Projects/animal_calib/");
	m_renderer.colorObjs.push_back(p_model);
	//m_renderer.colorObjs.push_back(p_model2);
	m_renderer.skels.push_back(joints_rend); 

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}

void generate_nodegraph()
{
	show_gpu_param();
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config2.json";
	ShapeSolver solver(pig_config);
	solver.SaveObj("D:/Projects/animal_calib/shapesolver/data/model.obj");
	NodeGraphGenerator graph;
	Mesh model;
	model.Load("data/model.obj");

	//graph.Generate(model); 

	graph.model = model;
	graph.CalcGeodesic();
	//graph.SaveGeodesic("data/geodesic.txt");
	graph.LoadGeodesic("data/geodesic.txt");

	graph.SampleNode(); 
	//graph.SampleNodeFromObj("data/manual_artist_head_node.obj");
	
	graph.GenKnn();
	graph.GenNodeNet();

	graph.Save("data/node_graph.txt");
	graph.VisualizeNodeNet("data/nodenet.obj");
	graph.VisualizeKnn("data/knn.obj");
}