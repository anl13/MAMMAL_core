#include <iostream>
#include "calibration.h"
#include "../render/renderer.h"
#include "../render/render_object.h"
#include "../render/render_utils.h"

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

void test_calib(std::string folder)
{
	Calibrator calibrator(folder);
	calibrator.calib_pipeline();
}

void test_epipolar(std::string folder)
{
	Calibrator calib(folder);
	calib.test_epipolar();
}

void show_scene()
{
	const float kFloorDx = 0;
	const float kFloorDy = 0;
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");
	std::vector<Camera> cams = readCameras(); 
	std::vector<cv::Mat> bgs = readImgs();

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
	m_renderer.s_camViewer.SetExtrinsic(cams[0].R.cast<float>(), cams[0].T.cast<float>());

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

	std::string point_file = conf_projectFolder + "/data/calibdata/adjust/points3d.txt";
	std::vector<Eigen::Vector3d> points = read_points(point_file);
	std::vector<Eigen::Vector3d> selected_points = points;

	std::vector<float> sizes(selected_points.size(), 0.05f);
	std::vector<Eigen::Vector3f> balls, colors;
	balls.resize(selected_points.size());
	colors.resize(selected_points.size());
	for (int i = 0; i < selected_points.size(); i++)
	{
		balls[i] = selected_points[i].cast<float>();
		colors[i] = CM[0];
	}
	BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors);
	m_renderer.skels.push_back(skelObject);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
	//m_renderer.Draw(); 
	//cv::Mat render = m_renderer.GetImage();
	//cv::Mat overlay = overlay_renders(bgs[0], render);
	//cv::namedWindow("show", cv::WINDOW_NORMAL); 
	//cv::imshow("show", overlay);
	//cv::waitKey(); 
	//cv::destroyAllWindows(); 

}

int main()
{
	show_scene();
	//test_epipolar("D:/Projects/animal_calib/"); 

	return 0; 
}