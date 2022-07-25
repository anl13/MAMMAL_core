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

Eigen::Vector3f alignByRotz(Eigen::Vector3f rotvec)
{
	Eigen::Matrix3f R = GetRodrigues(rotvec); 
	Eigen::Vector3f euler = Mat2Euler(R); 
	euler(0) = 0; 
	Eigen::Matrix3f R2 = EulerToRotRad(euler); 
	return Mat2Rotvec(R2); 
}

int visualize_artist_design()
{

	std::cout << "In render scene now!" << std::endl;

	std::string conf_projectFolder = "H:/MAMMAL_core/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

	FrameData frame; 
	frame.configByJson(conf_projectFolder + "configs/config_BamaPig3D_main.json");
	std::vector<Camera> cams = frame.get_cameras(); 

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
	std::string smal_config = "H:/MAMMAL_core/articulation/artist_config_sym.json";
	PigModelDevice smal(smal_config);
	smal.SetGlobalRotType("axis-angle");
	cv::VideoWriter writer("I:/pig_model_design/animation_new/artist_animation_curcial_detailedscene.avi", cv::VideoWriter::fourcc('m', 'p', 'e', 'g'), 25.0, cv::Size(1920, 1080));
	int state_id = 0;
 
	std::vector<int> crucial_joints = {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 38, 39, 40, 41, 54, 55, 56, 57, 21, 22, 23
	}; 
	state_id = 0; 
	std::vector<int> render_seq = {2}; 
	for (int i = 0; i < 8; i++)
	{
		if (!in_list(i, render_seq))
		{
			state_id += config[i].second; 
			continue; 
		}
		for (int t = 0; t < config[i].second; t++)
		{
			//int frameid = config[i].first + t;
			m_renderer.clearAllObjs();
			std::string stateFolder = "I:/pig_model_design/model_preprocess/designed_pig/pig_prior/c++/";
			std::stringstream ss;
			ss << stateFolder << "state_" << std::setw(6) << std::setfill('0') << state_id << ".txt";
			smal.readState(ss.str());
			//std::vector<Eigen::Vector3f> poses = smal.GetPose(); 
			//for (int k = 0; k < 62; k++)
			//{
			//	if (!in_list(k, crucial_joints)) poses[k].setZero(); 
			//}
			//smal.SetPose(poses); 
			smal.UpdateVertices();
			smal.saveObj("I:/pig_model_design/animation_new/obj/" + std::to_string(state_id) + ".obj"); 
			state_id++; 
			continue; 
			/// render skel 
			//std::vector<int> parents = smal.GetParents();
			//std::vector<Eigen::Vector3f> balls;
			//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
			//GetBallsAndSticks(joints, parents, balls, sticks);
			//int jointnum = 62;
			//std::vector<int> color_ids = {
			//	0,0,0,0,0, // spine
			//	1,1,1,1,1,1,1,1, // right back
			//	2,2,2,2,2,2,2,2, // left back 
			//	0,0,0, 1,2, 1,1,1,1,1,2,2,2,2,2, 0,0,// head 
			//	3,3,3,3,3,3,3,3, // right front 
			//	5,5,5,5,5,5,5,5, // tail 
			//	4,4,4,4,4,4,4,4 // left front
			//};
			//std::vector<Eigen::Vector3f> colors;
			//colors.resize(jointnum, CM[0]);
			//for (int i = 0; i < color_ids.size(); i++)colors[i] = CM[color_ids[i]];
			//BallStickObject* p_skel = new BallStickObject(ballMeshEigen, ballMeshEigen, balls, sticks, 0.01, 0.005, colors);
			//m_renderer.skels.push_back(p_skel);
			//m_renderer.createScene(conf_projectFolder);
			//m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
			//m_renderer.Draw();
			//cv::Mat img = m_renderer.GetImage();
			//writer.write(img);
			//std::cout << "finish " << state_id << std::endl;
			//state_id++;
			//continue;

			smal.UpdateNormalFinal(); 
			RenderObjectColor* animal_model = new RenderObjectColor();
			std::vector<Eigen::Vector3u> faces_u = smal.GetFacesVert();
			std::vector<Eigen::Vector3f> normals = smal.GetNormals();
			animal_model->SetFaces(faces_u);
			animal_model->SetVertices(smal.GetVertices());
			animal_model->SetNormal(normals);
			animal_model->SetColor(Eigen::Vector3f(0.5, 0.5, 0.1));
			animal_model->isMultiLight = false;
			m_renderer.colorObjs.push_back(animal_model);
			//m_renderer.s_camViewer.SetExtrinsic(cams[0].R, cams[0].T); 
			//m_renderer.Draw();
			//cv::Mat img0 = m_renderer.GetImage(); 
			//cv::Mat out; 
			//overlay_render_on_raw_gpu(img0, imgs[0], out);
			//cv::Mat outsmall;
			//cv::resize(out, outsmall, cv::Size(480, 270)); 
			//m_renderer.createScene(conf_projectFolder);
			m_renderer.createSceneHalf(conf_projectFolder, 1.08); 
			m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
			m_renderer.Draw();
			cv::Mat img = m_renderer.GetImageOffscreen();
			//outsmall.copyTo(img(cv::Rect(1440, 810, 480, 270)));
			std::stringstream ss_text;
			//ss_text << "frame " << std::setw(6) << frameid << "  seq " << i << " index " << t;
			//cv::putText(img, ss_text.str(), cv::Point(100, 100), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0));
			writer.write(img);
			std::cout << "write img " << i << "," << t << std::endl;
			state_id++;
		}
	}
	return 0; 

}

