#include <iostream>
#include "calibration.h"
#include "../render/renderer.h"
#include "../render/render_object.h"
#include "../render/render_utils.h"

#include "../utils/image_utils.h"
#include "../utils/math_utils.h" 

std::vector<Camera> readCameras()
{
	std::vector<Camera> cams; 
	std::vector<int> m_camids = {
		//1,2,5,6,7,8,9
		0,1,2,5,6,7,8,9,10,11
	};
	int m_camNum = m_camids.size(); 
	//std::string m_camDir = "D:/Projects/animal_calib/data/calibdata_animal_6/extrinsic/"; 
	//std::string m_camDir = "D:/Projects/animal_calibration/calib_batch3/result_batch3/";
	std::string m_camDir = "D:/Projects/animal_calib/data/calibdata/extrinsic/"; 
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
		Eigen::Vector3f rvec, tvec;
		for (int i = 0; i < 3; i++) {
			float a;
			camfile >> a;
			rvec(i) = a;
		}
		for (int i = 0; i < 3; i++)
		{
			double a;
			camfile >> a;
			tvec(i) = a;
		}

		Camera camUndist = Camera::getDefaultCameraUndist();
		//Camera camUndist = Camera::getFarCameraUndist(); 
		camUndist.SetRT(rvec, tvec);
		cams.push_back(camUndist);
		camfile.close();
	}
	return cams; 
}

std::vector<cv::Mat> readImgs()
{
	std::string folder = "D:/Projects/animal_calib/";
	//std::string m_imgDir = folder + "/data/calibdata_animal_5/backgrounds/bg";
	//std::string m_imgDir = "D:/Projects/animal_calibration/calib_batch3/result_batch3/undist/bg";
	std::string m_imgDir = "D:/Projects/animal_calibration/calib10/undist/bg"; 
	std::vector<int> m_camids = {
		//1,2,5,6,7,8,9
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


void show_scene()
{
	const float kFloorDx = 0;
	const float kFloorDy = 0;
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");
	std::vector<Camera> cams = readCameras(); 
	std::vector<cv::Mat> bgs = readImgs();

	// init a camera 
	Eigen::Matrix3f K = cams[0].K;
	K.row(0) /= 1920;
	K.row(1) /= 1080;
	std::cout << K << std::endl;

	// init renderer 
	Renderer::s_Init(false);
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(cams[0].R.cast<float>(), cams[0].T.cast<float>());
	//Eigen::Vector3f pos(0, 0, 5);
	//Eigen::Vector3f up(0, 1, 0);
	//Eigen::Vector3f center(0, 0, 0);
	//m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
	// init element obj
	const Mesh ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const Mesh stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const Mesh squareObj(conf_projectFolder + "/render/data/obj_model/square.obj");
	const Mesh cameraObj(conf_projectFolder + "/render/data/obj_model/camera.obj");
	const MeshEigen ballObjEigen(ballObj); 

	std::string point_file = conf_projectFolder + "/data/calibdata/extrinsic/points3d.txt";
	//std::string point_file = "D:/Projects/animal_calibration/calib_batch3/points3d_flipx.txt"; 
	std::vector<Eigen::Vector3f> points = read_points(point_file);
	std::vector<Eigen::Vector3f> selected_points = points;
	std::vector<float> sizes(selected_points.size(), 0.03f);
	std::vector<Eigen::Vector3f> balls, colors;
	balls.resize(selected_points.size());
	colors.resize(selected_points.size());
	for (int i = 0; i < selected_points.size(); i++)
	{
		balls[i] = selected_points[i].cast<float>();
		colors[i] = CM[0];
		//if (i == 74) colors[i] = CM[1];
	}
	BallStickObject* skelObject = new BallStickObject(ballObjEigen, balls, sizes, colors);
	m_renderer.skels.push_back(skelObject);


	//Mesh output;
	//for (int k = 0; k < balls.size(); k++)
	//{
	//	Mesh tmp = ballObj;
	//	for (int i = 0; i < tmp.vertices_vec.size(); i++)
	//	{
	//		tmp.vertices_vec[i] = tmp.vertices_vec[i] * sizes[k] + balls[k];
	//	}
	//	composeMesh(output, tmp); 
	//}
	//output.Save("D:/results/points.obj"); 


//	Mesh camObj("C:/Users/BBNC/Documents/maya/projects/default/scenes/pigscene/camera_big.obj");
//#if 1  // re-build scene model
//	//for (int i = 1; i < 7; i++)
//	//{
//		//Mesh sceneObj("C:/Users/BBNC/Documents/maya/projects/default/scenes/pigscene/zhujuan_new_part" + std::to_string(i) + ".obj", false);
//		Mesh sceneObj("C:/Users/BBNC/Documents/maya/projects/default/scenes/pigscene/zhujuan_halfwall3.obj", false);
//
//		Eigen::Matrix3f R = EulerToRotDegree(-90, 0, 90);
//		for (int i = 0; i < sceneObj.vertices_vec.size(); i++)
//		{
//			sceneObj.vertices_vec[i] = R * sceneObj.vertices_vec[i];
//		}
//		sceneObj.CalcNormal();
//		sceneObj.Save("D:/Projects/animal_calib/render/data/obj_model/zhujuan_halfwall3.obj");
//	//}
//	
//#endif 

#if 0
	Mesh sceneObj("C:/Users/BBNC/Documents/maya/projects/default/scenes/pigscene/zhujuan_new_part4.obj", false);
	Eigen::Matrix3f R = EulerToRotDegree(-90, 0, 90);
	for (int i = 0; i < sceneObj.vertices_vec.size(); i++)
	{
		sceneObj.vertices_vec[i] = R * sceneObj.vertices_vec[i];
	}
	sceneObj.CalcNormal();
	RenderObjectColor *p_scene = new RenderObjectColor();
	p_scene->SetVertices(sceneObj.vertices_vec);
	p_scene->SetNormal(sceneObj.normals_vec);
	p_scene->SetColor(Eigen::Vector3f(0.9, 0.9, 0.9));
	p_scene->isMultiLight = false;
	p_scene->SetFaces(sceneObj.faces_v_vec);
	m_renderer.colorObjs.push_back(p_scene);

#endif

	m_renderer.createSceneDetailed("D:/Projects/animal_calib", 1, -1);
	//RenderObjectColor *p_cam = new RenderObjectColor();
	//p_cam->SetVertices(camObj.vertices_vec);
	//p_cam->SetNormal(camObj.normals_vec);
	//p_cam->SetFaces(camObj.faces_v_vec);
	//p_cam->SetColor(Eigen::Vector3f(0.9, 0.9, 0.9));
	//p_cam->isMultiLight = false;


	//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	//while (!glfwWindowShouldClose(windowPtr))
	//{
	//	glDisable(GL_CULL_FACE);
	//	m_renderer.Draw();
	//	glfwSwapBuffers(windowPtr);
	//	glfwPollEvents();
	//};

	//std::vector<int> camids = {1,2,5,6,7,8,9};
	//std::vector<int> camids = { 4,6,9,10,12 };
	std::vector<int> camids = { 0,1,2,5,6,7,8,9,10,11 };
	for (int i = 0; i < cams.size(); i++)
	{
		m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1)); 
		m_renderer.s_camViewer.SetExtrinsic(cams[i].R.cast<float>(), cams[i].T.cast<float>());
		glDisable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		m_renderer.Draw();
		//cv::Mat render = m_renderer.GetImage();
		cv::Mat render = m_renderer.GetImageOffscreen(); 
		cv::Mat overlay = overlay_renders(bgs[i], render,0.5);
		std::stringstream ss_out; 
		//ss_out << "D:/Projects/animal_calibration/calib_batch3/result_batch3/undist/bg" << camids[i] << "_overlay.png"; 
		ss_out << "D:/Projects/animal_calibration/calib10/undist_new/bg" << camids[i] << "_overlay.png"; 
		cv::imwrite(ss_out.str(), overlay);
		//cv::namedWindow("show", cv::WINDOW_NORMAL);
		//cv::imshow("show", overlay);
		//cv::waitKey();
		//cv::destroyAllWindows();
	}
}

// map between artist designed scene mesh to 
void show_artist_scene()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	Eigen::Vector3f up; up << 0.f, 0.f, 1.f;
	Eigen::Vector3f pos; pos << -1.f, 1.5f, 0.8f;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();

	// init renderer 
	Renderer::s_Init();

	Renderer m_renderer(conf_projectFolder + "/render/shader/");

	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);


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

	Eigen::Matrix3f tmp;
	tmp << 50.7, 0, -225, -320, 0, -225, -320, 0, 190;
	Eigen::Matrix3f pointsRaw = tmp.transpose();
	Eigen::Matrix3f pointsNew;
	tmp << -1, 0.91, 0, -1, -0.91, 0, 1, -0.91, 0;
	pointsNew = tmp.transpose();
	float scale = (pointsNew.col(0) - pointsNew.col(1)).norm() /
		(pointsRaw.col(0) - pointsNew.col(1)).norm();
	pointsRaw = pointsRaw * scale;
	Eigen::Matrix3f R;
	R << 0, 0, 1, 1, 0, 0, 0, 1, 0;
	pointsRaw = R * pointsRaw;
	Eigen::Vector3f T = (pointsNew.col(0) + pointsNew.col(2)) / 2 -
		(pointsRaw.col(0) + pointsRaw.col(2)) / 2;
	pointsRaw = pointsRaw.colwise() + T;

	float scale2 = ((pointsNew.col(0) - pointsNew.col(1)).norm() + (pointsNew.col(0) - pointsNew.col(2)).norm() + (pointsNew.col(1) - pointsNew.col(2)).norm())
		/ ((pointsRaw.col(0) - pointsRaw.col(1)).norm() + (pointsRaw.col(0) - pointsRaw.col(2)).norm() + (pointsRaw.col(1) - pointsRaw.col(2)).norm());
	pointsRaw = pointsRaw * scale2;

	std::cout << "pointsRaw: " << std::endl << pointsRaw << std::endl;
	std::cout << "pointsNew: " << std::endl << pointsNew << std::endl;
	std::cout << "R: " << std::endl << R << std::endl;

	std::vector<Eigen::Vector3f> points;
	points.push_back(pointsRaw.col(0));
	points.push_back(pointsRaw.col(1));
	points.push_back(pointsRaw.col(2));
	points.push_back(pointsNew.col(0));
	points.push_back(pointsNew.col(1));
	points.push_back(pointsNew.col(2));

	std::vector<float> sizes(6, 0.05f);
	std::vector<Eigen::Vector3f> balls, colors;
	balls = points;
	colors.resize(points.size());
	for (int i = 0; i < points.size(); i++)
	{
		if (i < 3)
			colors[i] = CM[0];
		else colors[i] = CM[1];
	}
	BallStickObject* skelObject = new BallStickObject(ballMeshEigen, balls, sizes, colors);
	m_renderer.skels.push_back(skelObject);

	Mesh sceneMesh("F:/projects/model_preprocess/designed_pig/scenes/scene_triangle.obj", false);
	for (int i = 0; i < sceneMesh.vertex_num; i++)
	{
		sceneMesh.vertices_vec[i] = scale2 * (R * (scale * sceneMesh.vertices_vec[i]) + T);
		//sceneMesh.normals_vec[i] = R * sceneMesh.normals_vec[i];
	}
	sceneMesh.CalcNormal();

	RenderObjectColor* p_scene = new RenderObjectColor();
	p_scene->SetFaces(sceneMesh.faces_v_vec);
	p_scene->SetVertices(sceneMesh.vertices_vec);
	p_scene->SetNormal(sceneMesh.normals_vec);

	p_scene->SetColor(CM[0]);

	m_renderer.colorObjs.push_back(p_scene);
	
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	while (!glfwWindowShouldClose(windowPtr))
	{
		//glEnable(GL_CULL_FACE);
		glDisable(GL_CULL_FACE);
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}

// map between artist designed scene mesh to 
void adjust_calibration()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");
	std::vector<Eigen::Vector3f> CM0 = getColorMapEigenF("anliang_rgb"); 
	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	Eigen::Vector3f up; up << 0.f, 0.f, 1.f;
	Eigen::Vector3f pos; pos << -1.f, 1.5f, 0.8f;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();

	// init renderer 
	Renderer::s_Init();

	Renderer m_renderer(conf_projectFolder + "/render/shader/");

	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);


	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	//RenderObjectTexture* chess_floor = new RenderObjectTexture();
	//chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	//chess_floor->SetFaces(squareMesh.faces_v_vec);
	//chess_floor->SetVertices(squareMesh.vertices_vec);
	//chess_floor->SetNormal(squareMesh.normals_vec, 2);
	//chess_floor->SetTexcoords(squareMesh.textures_vec, 1);
	//chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	//m_renderer.texObjs.push_back(chess_floor);

	std::string point_file = conf_projectFolder + "/data/calibdata/tmp/points3d.txt";
	std::vector<Eigen::Vector3f> points = read_points(point_file);
	std::vector<Eigen::Vector3f> selected_points;

	Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
	R(0, 0) = 1; 
	R(1, 1) = -1; 
	R(2, 2) = -1;

	for (int i = 0; i < points.size(); i++)
	{
		points[i] = R * (points[i]);
	}
	Eigen::Vector3f translation;
	translation = -(points[45] + points[48]) / 2;
	for (int i = 0; i < points.size(); i++)
	{
		points[i] = points[i] + translation; 
	}

	//save 
	std::ofstream pointfile("D:/Projects/animal_calib/data/calibdata/adjust_new/points3d.txt"); 
	for (int i = 0; i < points.size(); i++)
	{
		pointfile << points[i].transpose() << std::endl; 
	}
	pointfile.close(); 
	std::vector<int> m_camids = {
		0,1,2,5,6,7,8,9,10,11
	};
	int m_camNum = m_camids.size();
	std::string m_camDir = "D:/Projects/animal_calib/data/calibdata/tmp/";
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
		Eigen::Vector3f rvec, tvec;
		for (int i = 0; i < 3; i++) {
			float a;
			camfile >> a;
			rvec(i) = a;
		}
		for (int i = 0; i < 3; i++)
		{
			double a;
			camfile >> a;
			tvec(i) = a;
		}

		Eigen::Matrix3f Rmat = GetRodrigues(rvec);
		Eigen::Matrix3f R2 = Rmat * R.transpose(); 
		Eigen::Vector3f T = -R.transpose() * translation;
		T = Rmat * T + tvec;
		Eigen::Vector3f rvec_2 = Mat2Rotvec(R2); 
		std::stringstream ss_out; 
		ss_out << "D:/Projects/animal_calib/data/calibdata/adjust_new/" << std::setw(2) << std::setfill('0') << m_camids[camid] << ".txt";
		std::ofstream outfile(ss_out.str()); 
		outfile << rvec_2 << std::endl << T;
		outfile.close(); 
	}


	selected_points.push_back(points[45]);
	selected_points.push_back(points[47]);
	selected_points.push_back(points[48]);
	
	std::vector<float> sizes(points.size(), 0.05f);
	std::vector<Eigen::Vector3f> balls, colors;
	balls = points;
	colors.resize(points.size());
	for (int i = 0; i < points.size(); i++)
	{
		colors[i] = CM[1];
	}
	colors[45] = CM0[0];
	colors[47] = CM0[0];
	colors[48] = CM0[0];
	std::cout << "45: " << points[45].transpose() << std::endl; 
	std::cout << "47: " << points[47].transpose() << std::endl; 
	std::cout << "48: " << points[48].transpose() << std::endl; 
	std::cout << "42: " << points[42].transpose() << std::endl; 
	std::cout << "43: " << points[43].transpose() << std::endl;
	std::cout << "44: " << points[44].transpose() << std::endl; 
	std::cout << "49: " << points[49].transpose() << std::endl; 
	std::cout << "50: " << points[50].transpose() << std::endl; 

	colors[50] = CM0[1];
	std::cout << "55: " << points[55].transpose() << std::endl; 
	BallStickObject* skelObject = new BallStickObject(ballMeshEigen, balls, sizes, colors);
	m_renderer.skels.push_back(skelObject);

	m_renderer.createScene("D:/Projects/animal_calib"); 

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	std::vector<Camera> cams = readCameras(); 
	std::vector<cv::Mat> imgs(cams.size()); 
	for (int i = 0; i < cams.size(); i++)
	{
		m_renderer.s_camViewer.SetExtrinsic(cams[i].R, cams[i].T);
		m_renderer.Draw(); 
		imgs[i] = m_renderer.GetImage(); 
	}
	cv::Mat packed; 
	packImgBlock(imgs, packed); 
	cv::imwrite("D:/Projects/animal_calib/data/calibdata/adjust_new/img.png", packed); 

	while (!glfwWindowShouldClose(windowPtr))
	{
		//glEnable(GL_CULL_FACE);
		glDisable(GL_CULL_FACE);
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}

int main()
{
	//adjust_calibration(); 

	// ---- render scene model and points onto undistorted images 
	// to see whether the manually designed scene model could roughly 
	// fit image features. 
	show_scene();

	//// ---- calibrating the 10 camera system. 
	//std::string folder = "D:/Projects/animal_calib/"; 
	//Calibrator calib(folder); 
	////calib.test_epipolar(); 
	//calib.calib_pipeline(); 

	return 0; 
}