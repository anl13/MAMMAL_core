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
#include "../utils/image_utils_gpu.h"

#include "pigmodeldevice.h"
#include "pigsolverdevice.h"
#include "../utils/mesh.h"

#include "test_main.h"
#include "../utils/timer_util.h"

#include "../posesolver/framedata.h"

std::vector<std::pair<int, int> > config = {
	{405, 151}, // {startid, framenum}
{600, 401},
{1010, 449},
{5130, 350}, 
{6650, 300},
{8600, 325},
{9600, 250},
{9300, 500}
}; 

//std::vector <std::vector<Eigen::Vector3f> > readAllJoints()
//{
//	std::vector<std::vector<Eigen::Vector3f> > all_joints; 
//	all_joints.resize(1); 
//	std::ifstream inputfile("F:/projects/model_preprocess/designed_pig/pig_prior/c++/joints.txt");
//	for (int i = 0; i < 1; i++)
//	{
//		all_joints[i].resize(62); 
//		for (int k = 0; k < 62; k++)
//		{
//			for(int m = 0; m < 3)
//		}
//	}
//}

Eigen::Vector3f alignByRotz(Eigen::Vector3f rotvec)
{
	Eigen::Matrix3f R = GetRodrigues(rotvec); 
	Eigen::Vector3f euler = Mat2Euler(R); 
	euler(0) = 0; 
	Eigen::Matrix3f R2 = EulerToRotRad(euler); 
	return Mat2Rotvec(R2); 
}

int test_visdesigned()
{

	std::vector<int> flip_index = {
0,1,2,3,4,13,14,15,16,17,18,19,20,5,6,7,8,9,10,11,12,21,22,23,25,24,31,32,33,34,35,
26,27,28,29,30,36,37,54,55,56,57,58,59,60,61,46,47,48,49,50,51,52,53,
38,39,40,41,42,43,44,45
	};

	std::cout << "In render scene now!" << std::endl;

	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");
	std::vector<Camera> cams = readCameras();

	// init a camera 
	Eigen::Matrix3f K = cams[0].K; 
	K.row(0) /= 1920;
	K.row(1) /= 1080;
	std::cout << K << std::endl;

	Eigen::Vector3f up; up << 0.182088, 0.260274,  0.94821;
	Eigen::Vector3f pos; pos << -1.13278, - 1.31876, 0.559579;
	Eigen::Vector3f center; center << -0.223822, -0.243763, 0.0899532;


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
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	PigModelDevice smal(smal_config);
	cv::VideoWriter writer("artist_animation_cluster.avi", cv::VideoWriter::fourcc('m', 'p', 'e', 'g'), 1.0, cv::Size(1920, 1080));
	int state_id = 0;
	FrameData frame; 
	frame.configByJson("D:/Projects/animal_calib/posesolver/config.json"); 
	//for (int i = 0; i < config.size(); i++)
	//{
	//	for (int t = 0; t < config[i].second; t++)
	//	{
	//		int frameid = config[i].first + t; 
	//		//frame.set_frame_id(frameid); 
	//		//frame.fetchData(); 
	//		//std::vector<cv::Mat> imgs = frame.get_imgs_undist(); 
	//		m_renderer.clearAllObjs(); 

	//		std::string stateFolder = "F:/projects/model_preprocess/designed_pig/pig_prior/c++/";
	//		std::stringstream ss;
	//		ss << stateFolder << "state_modify_" << std::setw(6) << std::setfill('0') << state_id << ".txt";
	//		smal.readState(ss.str());

	//		//Eigen::Vector3f trans = smal.GetTranslation(); 
	//		//std::vector<Eigen::Vector3f> pose = smal.GetPose(); 
	//		//trans(0) = 0; 
	//		//trans(1) = 0; 
	//		//smal.SetTranslation(trans); 
	//		//pose[0] = alignByRotz(pose[0]);
	//		//smal.SetPose(pose);

	//		smal.UpdateVertices();
	//		smal.UpdateNormalFinal();

	//		RenderObjectColor* animal_model = new RenderObjectColor();
	//		std::vector<Eigen::Vector3u> faces_u = smal.GetFacesVert();
	//		std::vector<Eigen::Vector3f> normals = smal.GetNormals();
	//		animal_model->SetFaces(faces_u);
	//		animal_model->SetVertices(smal.GetVertices());
	//		animal_model->SetNormal(normals);
	//		animal_model->SetColor(Eigen::Vector3f(0.5, 0.5, 0.1));

	//		m_renderer.colorObjs.push_back(animal_model);
	//		//m_renderer.s_camViewer.SetExtrinsic(cams[0].R, cams[0].T); 
	//		//m_renderer.Draw();
	//		//cv::Mat img0 = m_renderer.GetImage(); 
	//		//cv::Mat out; 
	//		//overlay_render_on_raw_gpu(img0, imgs[0], out);
	//		//cv::Mat outsmall;
	//		//cv::resize(out, outsmall, cv::Size(480, 270)); 


	//		m_renderer.createScene(conf_projectFolder);
	//		m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
	//		m_renderer.Draw();
	//		cv::Mat img = m_renderer.GetImage();

	//		//outsmall.copyTo(img(cv::Rect(1440, 810, 480, 270)));

	//		std::stringstream ss_text; 
	//		ss_text << "frame " << std::setw(6) << frameid << "  seq " << i << " index " << t;
	//		cv::putText(img, ss_text.str(), cv::Point(100, 100), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
	//		writer.write(img);
	//		std::cout << "write img " << i << "," << t << std::endl;
	//		state_id++;
	//	}
	//}

	for (int i = 0; i < 30; i++)
	{
		m_renderer.clearAllObjs();

		std::vector<Eigen::Vector3f> joints;
		std::stringstream ss; 
		ss << "F:/projects/model_preprocess/designed_pig/pig_prior/c++/centeroid_" << i;
		std::ifstream inputfile(ss.str());
		joints.resize(62); 
		for (int k = 0; k < 62; k++)
		{
			for (int m = 0; m < 3; m++) inputfile >> joints[k](m);
		}
		inputfile.close(); 

		for (int k = 46; k < 54; k++)joints[k] = joints[0];

		std::vector<int> parents = smal.GetParents();
		std::vector<Eigen::Vector3f> balls;
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
		GetBallsAndSticks(joints, parents, balls, sticks);
		int jointnum = 62;
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
		m_renderer.skels.push_back(p_skel); 

		m_renderer.createScene(conf_projectFolder);
		m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
		m_renderer.Draw();
		cv::Mat img = m_renderer.GetImage();
		writer.write(img); 
		std::stringstream outss; 
		outss << "cluster_" << i << ".png";
		cv::imwrite(outss.str(), img); 
	}
	writer.release(); 

	return 0; 
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

	return 0;
}

