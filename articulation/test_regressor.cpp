#include "test_main.h"

#include "pigmodeldevice.h"
#include "../render/render_object.h"
#include "../render/renderer.h"
#include "../render/render_utils.h"
#include <time.h>
#include <fstream> 
#include <iostream> 
#include <sstream> 
#include <string>

void savepoints(std::string filename, const std::vector<Eigen::Vector3f>& points)
{
	std::ofstream outfile(filename); 
	if (!outfile.is_open())
	{
		std::cout << "can not open file " << filename << std::endl; 
		return; 
	}
	for (int i = 0; i < points.size(); i++)
	{
		outfile << points[i].transpose() << std::endl;
	}
	return; 
}

void generate_samples()
{
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	PigModelDevice pig(smal_config);

	for (int sampleid = 0; sampleid < 500; sampleid++)
	{
		srand(time(NULL));
		Eigen::VectorXf pose = Eigen::VectorXf::Random(62 * 3) * 0.35;
		pig.SetPose(pose);
		pig.SetTranslation(Eigen::Vector3f::Random()*0.35);
		pig.UpdateVertices();

		std::vector<Eigen::Vector3f> joints = pig.GetJoints();
		std::vector<Eigen::Vector3f> verts = pig.GetVertices();

		std::stringstream jointname;
		jointname << "D:/Projects/animal_python/regressor_data/joint_" << sampleid << ".txt";
		savepoints(jointname.str(), joints); 
		std::stringstream vertname; 
		vertname << "D:/Projects/animal_python/regressor_data/vert_" << sampleid << ".txt";
		savepoints(vertname.str(), verts); 
	}

	
}

void test_regressor()
{
	//generate_samples(); 
	//return; 

	// render config 
	const float kFloorDx = 0;
	const float kFloorDy = 0;

	std::cout << "In render scene now!" << std::endl;

	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");
	std::vector<Eigen::Vector3f> CM2 = getColorMapEigenF("anliang_render");
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
	PigModelDevice pig(smal_config);

	srand(time(NULL)); 
	Eigen::VectorXf pose = Eigen::VectorXf::Random(62 * 3) * 0.4;
	pig.SetPose(pose);
	pig.SetTranslation(Eigen::Vector3f(0, 0, 0.21)); 
	
	pig.UpdateVertices();
	pig.UpdateNormalFinal();
	std::vector<Eigen::Vector3f> regressedjoints = pig.RegressJointsPosed(); 

	//RenderObjectColor* animal_model = new RenderObjectColor();
	//std::vector<Eigen::Vector3f> verts = pig.GetVertices(); 
	//std::vector<Eigen::Vector3u> faces_u = pig.GetFacesVert();
	//std::vector<Eigen::Vector3f> normals = pig.GetNormals();
	//animal_model->SetFaces(faces_u);
	//animal_model->SetVertices(verts);
	//animal_model->SetNormal(normals);
	//animal_model->SetColor(Eigen::Vector3f(0.5, 0.5, 0.1));


	std::vector<Eigen::Vector3f> joints = pig.GetJoints();
	std::vector<int> parents = pig.GetParents(); 
	std::vector<Eigen::Vector3f> balls;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	GetBallsAndSticks(joints, parents, balls, sticks);
	int jointnum = pig.GetJointNum();
	std::vector<int> color_ids = {
		0,0,0,0,0, // spine
		1,1,1,1,1,1,1,1, // right back
		2,2,2,2,2,2,2,2, // left back 
		0,0,0, 1,2, 1,1,1,1,1,2,2,2,2,2, 0,0,// head 
		3,3,3,3,3,3,3,3, // right front 
		5,5,5,5,5,5,5,5, // tail 
		4,4,4,4,4,4,4,4 // left front
	};
	std::vector<Eigen::Vector3f> colors;
	colors.resize(jointnum, CM[0]);
	for (int i = 0; i < color_ids.size(); i++)colors[i] = CM[color_ids[i]];
	BallStickObject* p_skel = new BallStickObject(ballMeshEigen, ballMeshEigen, balls, sticks, 0.01, 0.005, colors);

	std::vector<Eigen::Vector3f> balls2;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks2;
	GetBallsAndSticks(regressedjoints, parents, balls2, sticks2);
	std::vector<Eigen::Vector3f> colors2;
	colors2.resize(jointnum, CM2[0]);
	for (int i = 0; i < color_ids.size(); i++)colors2[i] = CM2[color_ids[i]];
	BallStickObject* p_skel2 = new BallStickObject(ballMeshEigen, ballMeshEigen, balls2, sticks2, 0.01, 0.005, colors2);


	//m_renderer.colorObjs.push_back(animal_model);
	m_renderer.skels.push_back(p_skel); 
	m_renderer.skels.push_back(p_skel2); 


	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

}

