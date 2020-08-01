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
#include "../utils/obj_reader.h"

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
		Vec3 rvec, tvec;
		for (int i = 0; i < 3; i++) {
			double a;
			camfile >> a;
			rvec(i) = a;
		}
		for (int i = 0; i < 3; i++)
		{
			double a;
			camfile >> a;
			tvec(i) = a;
		}

		Camera camUndist = getDefaultCameraUndist();
		camUndist.SetRT(rvec, tvec);
		cams.push_back(camUndist);
		camfile.close();
	}
	return cams;
}

std::vector<cv::Mat> readImgs()
{
	std::string folder = "D:/Projects/animal_calib/";
	std::string m_imgDir = folder + "/data/backgrounds/bg";
	std::vector<int> m_camids = {
		0,1,2,5,6,7,8,9,10,11
	};
	int m_camNum = m_camids.size();
	std::vector<cv::Mat> m_imgs;
	for (int camid = 0; camid < m_camNum; camid++)
	{
		std::stringstream ss;
		ss << m_imgDir << m_camids[camid] << "_undist.png";
		cv::Mat img = cv::imread(ss.str());
		if (img.empty())
		{
			std::cout << "img is empty! " << ss.str() << std::endl;
			exit(-1);
		}
		m_imgs.push_back(img);
	}
	return m_imgs;
}

int test_mean_pose()
{
	// render config 
	const float kFloorDx = 0;
	const float kFloorDy = 0;

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
	const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj");
	const ObjData cameraObj(conf_projectFolder + "/render/data/obj_model/camera.obj");

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, false);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ kFloorDx, kFloorDy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	PigModel smal(smal_config);

	//// smal random pose 
	//Eigen::VectorXd pose = Eigen::VectorXd::Random(62 * 3) * 0.3;
	//smal.SetPose(pose);
	//smal.UpdateVertices(); 

	RenderObjectColor* animal_model = new RenderObjectColor();
	Eigen::MatrixXf vertices_f = smal.GetVertices().cast<float>();
	vertices_f = vertices_f.colwise() + Eigen::Vector3f(0, 0, 0.21f); 
	Eigen::MatrixXu faces_u = smal.GetFacesVert();
	animal_model->SetFaces(faces_u);
	animal_model->SetVertices(vertices_f);
	animal_model->SetColor(Eigen::Vector3f(0.5, 0.5, 0.1));
	
	//smal.testReadJoint("F:/projects/model_preprocess/designed_pig/extracted/framedata/joints_4.txt");
	
	Eigen::MatrixXf joints = smal.GetJoints().cast<float>();
	joints = joints.colwise() + Eigen::Vector3f(0, 0, 0.21f); 

	std::cout << joints << std::endl; 
	Eigen::VectorXi parents = smal.GetParents(); 
	std::vector<Eigen::Vector3f> balls;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	GetBallsAndSticks(joints, parents, balls, sticks);
	int jointnum = smal.GetJointNum(); 
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
	BallStickObject* p_skel = new BallStickObject(ballObj, stickObj, balls, sticks, 0.01, 0.005, colors);

	//smal.testReadJoint("F:/projects/model_preprocess/designed_pig/extracted/framedata/joints_diff.txt");
	smal.testReadJoint("F:/projects/model_preprocess/designed_pig/pig_prior/tmp/testjoint.txt");
	Eigen::MatrixXf joints2 = smal.GetJoints().cast<float>(); 

	std::vector<Eigen::Vector3f> balls2;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks2; 
	GetBallsAndSticks(joints2, parents, balls2, sticks2);
	for (int i = 0; i < colors.size(); i++)
	{
		colors[i] = colors[i] * 0.5;
	}
	BallStickObject* p_skel2 = new BallStickObject(ballObj, stickObj, balls2, sticks2, 0.01, 0.005, colors); 

	m_renderer.colorObjs.push_back(animal_model); 
	//m_renderer.skels.push_back(p_skel); 
	//m_renderer.skels.push_back(p_skel2); 

	
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

	return 0; 
}

void read_obj(std::string filename, Eigen::MatrixXf& vertices, Eigen::MatrixXu& faces)
{
	std::vector<Eigen::Vector3f> vs; 
	std::vector<Eigen::Vector3u> fs; 
	std::fstream reader;
	reader.open(filename.c_str(), std::ios::in);

	if (!reader.is_open())
	{
		std::cout << "[ObjData] file not exist!" << std::endl;
		exit(-1);
	}
	while (!reader.eof())
	{
		std::string dataType;
		reader >> dataType;

		if (reader.eof()) break;

		if (dataType == "v")
		{
			Eigen::Vector3f temp;
			reader >> temp.x() >> temp.y() >> temp.z();
			vs.push_back(temp);
		}
		else if (dataType == "f")
		{
			Eigen::Vector3u temp; 
			reader >> temp.x() >> temp.y() >> temp.z(); 
			temp[0] = temp[0] - 1; 
			temp[1] = temp[1] - 1;

			temp[2] = temp[2] - 1;

			fs.push_back(temp);
		}
		else
		{
			continue;
		}
	}
	int vertexnum = vs.size();
	int facenum = fs.size();
	vertices.resize(3, vertexnum);
	faces.resize(3, facenum);
	for (int i = 0; i < vertexnum; i++)vertices.col(i) = vs[i];
	for (int i = 0; i < facenum; i++)faces.col(i) = fs[i];
}

int test_write_video()
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
	const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj");
	const ObjData cameraObj(conf_projectFolder + "/render/data/obj_model/camera.obj");

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, false);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

	std::string tmp_folder = "F:/projects/model_preprocess/designed_pig/pig_prior/tmp/";
	
	//std::vector<cv::Mat> images;
	for (int k = 0; k < 100; k++)
	{
		m_renderer.colorObjs.clear(); 
		std::stringstream ss;
		ss << tmp_folder << "demo2_lie/demo" << k << ".obj";
		Eigen::MatrixXu faces_u;
		Eigen::MatrixXf vertices_f;
		read_obj(ss.str(), vertices_f, faces_u);
		RenderObjectColor* animal_model = new RenderObjectColor();

		vertices_f = vertices_f.colwise() + Eigen::Vector3f(0, 0, 0.21f);

		animal_model->SetFaces(faces_u);
		animal_model->SetVertices(vertices_f);
		animal_model->SetColor(Eigen::Vector3f(0.5, 0.5, 0.1));

		m_renderer.colorObjs.push_back(animal_model);

		m_renderer.Draw();
		cv::Mat img = m_renderer.GetImage();
		//images.push_back(img); 
		std::stringstream ss_out;
		ss_out << tmp_folder << "demo2_lie/img" << k << ".png";
		cv::imwrite(ss_out.str(), img);
	}
	//cv::Mat packed; 
	//packImgBlock(images, packed);
	//cv::Mat small;
	//cv::resize(packed, small, cv::Size(1920, 1080));
	//cv::imwrite(tmp_folder + "range/all.png", small);

	//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	//while (!glfwWindowShouldClose(windowPtr))
	//{
	//	m_renderer.Draw();
	//	glfwSwapBuffers(windowPtr);
	//	glfwPollEvents();
	//};

	return 0; 
}

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
	const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj");
	const ObjData cameraObj(conf_projectFolder + "/render/data/obj_model/camera.obj");

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, false);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

	std::string pig_config = "D:/Projects/animal_calib/articulation/artist_config.json";

	PigModel gtpig(pig_config); 
	gtpig.setIsLatent(false); 
	Eigen::VectorXd pose = Eigen::VectorXd::Random(62 * 3) * 0.1; 
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
void main()
{
	test_vae(); 
}