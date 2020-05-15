#include <iostream> 
#include <fstream> 
#include <sstream> 
#ifndef WIN32
#include <unistd.h> 
#else 
#ifndef _UNISTD_H
#define _UNISTD_H
#include <io.h>
#include <process.h>
#endif /* _UNISTD_H */
#endif 
#include "../utils/camera.h"
#include "../utils/image_utils.h"
#include "../associate/framedata.h" 
#include "../utils/math_utils.h"
#include "../utils/colorterminal.h" 
#include "../smal/pigmodel.h"
#include "../utils/node_graph.h"
#include "../utils/model.h"
#include "../utils/dataconverter.h" 
#include "../nanorender/NanoRenderer.h"
#include <vector_functions.hpp>
#include "../utils/timer.hpp" 
#include "main.h"

int render_smal_test()
{
    std::cout << "In smal render now!" << std::endl; 

    std::string conf_projectFolder = "D:/projects/animal_calib/render";
	std::string smal_config = "D:/Projects/animal_calib/smal/smal2_config.json";
	SkelTopology m_topo = getSkelTopoByType("UNIV");
    /// read smal model 
	PigSolver smal(smal_config);
	std::string folder = smal.GetFolder();
    smal.UpdateVertices();

	FrameData framedata; 
	framedata.configByJson("D:/Projects/animal_calib/associate/config.json");
	framedata.set_frame_id(0);
	framedata.fetchData();
	auto cams = framedata.get_cameras();

	Eigen::Vector3f up; up << 0, 0, -1;
	Eigen::Vector3f pos; pos << -1, 1.5, -0.8;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();

	Model m3c;
	m3c.vertices = smal.GetVertices(); 
	m3c.faces = smal.GetFacesVert();
	m3c.CalcNormal();
	ObjModel m4c;
	convert3CTo4C(m3c, m4c);

	Camera cam = cams[0];
	NanoRenderer renderer; 
	renderer.Init(1920, 1080, float(cam.K(0, 0)), float(cam.K(1, 1)), float(cam.K(0, 2)), float(cam.K(1, 2)));
	std::cout << "cam.K: " << cam.K << std::endl;

	auto human_model = renderer.CreateRenderObject("human_model", vs_phong_geometry, fs_phong_geometry);
	human_model->SetIndices(m4c.indices);
	human_model->SetBuffer("positions", m4c.vertices);
	human_model->SetBuffer("normals", m4c.normals);
	
	Eigen::Matrix4f view_eigen = calcRenderExt(cam.R.cast<float>(), cam.T.cast<float>());
	nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
	//human_model->SetView(view_nano);
	
	renderer.UpdateCanvasView(view_eigen);

	int frameIdx = 0;
	while (!renderer.ShouldClose())
	{
		// rotate box1 along z axis
		//human_model->SetModelRT(Matrix4f::translate(Vector3f(-0.1, 0, 0.1)) * Matrix4f::rotate(Vector3f(0, 0, 1), glfwGetTime()));
		renderer.Draw();
		++frameIdx;
	}

    return 0; 
}

/*
void renderScene()
{
	const float kFloorDx = 0.28;
	const float kFloorDy = 0.2;


    std::cout << "In render scene now!" << std::endl; 

    std::string conf_projectFolder = "D:/projects/animal_calib/render";
    auto CM = getColorMapEigen("anliang_rgb"); 

    // init a camera 
    Eigen::Matrix3f K; 
    K << 0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0, 1;
    std::cout << K << std::endl; 

    Eigen::Vector3f up; up << 0,0, -1; 
    Eigen::Vector3f pos; pos << -1, 1.5, -0.8; 
    Eigen::Vector3f center = Eigen::Vector3f::Zero(); 

    // init renderer 
    Renderer::s_Init(); 
    Renderer m_renderer(conf_projectFolder + "/shader/"); 
    m_renderer.s_camViewer.SetIntrinsic(K, 1, 1); 
    m_renderer.s_camViewer.SetExtrinsic(pos, up, center); 

    // init element obj
    const ObjData ballObj(conf_projectFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/data/obj_model/cylinder.obj");
    const ObjData squareObj(conf_projectFolder + "/data/obj_model/square.obj"); 
    const ObjData cameraObj(conf_projectFolder + "/data/obj_model/camera.obj"); 

    RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, true);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ kFloorDx, kFloorDy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor); 

    std::string point_file = conf_projectFolder + "/../build/results/points3d.txt";
    std::vector<Eigen::Vector3d> points = read_points(point_file);
    std::vector<float> sizes(points.size(), 0.05f); 
    std::vector<Eigen::Vector3f> balls, colors; 
    balls.resize(points.size()); 
    colors.resize(points.size());
    for(int i = 0; i < points.size(); i++)
    {
        balls[i] = points[i].cast<float>(); 
        colors[i] = CM[0]; 
    }
    BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors); 
    m_renderer.skels.push_back(skelObject); 

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 

    while(!glfwWindowShouldClose(windowPtr))
    {
        m_renderer.Draw(); 
        glfwSwapBuffers(windowPtr); 
        glfwPollEvents(); 
    };
}
*/

int test_depth()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/render";
	std::string smal_config = "D:/Projects/animal_calib/smal/smal2_config.json";
	SkelTopology m_topo = getSkelTopoByType("UNIV");
	/// read smal model 
	//PigModel smal(smal_config); 
	PigSolver smal(smal_config);
	std::string folder = smal.GetFolder();
	smal.UpdateVertices();

	// init render
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
	std::cout << K << std::endl;

	FrameData frame;
	frame.configByJson("D:/Projects/animal_calib/associate/config.json");
	frame.set_frame_id(0); 
	frame.fetchData();
	auto cams = frame.get_cameras();
	Camera cam = cams[0]; 
	Model m3c;
	m3c.vertices = smal.GetVertices();
	m3c.faces = smal.GetFacesVert();
	m3c.CalcNormal();
	ObjModel m4c;
	convert3CTo4C(m3c, m4c);

	NanoRenderer renderer;
	renderer.Init(1920, 1080, float(cam.K(0, 0)), float(cam.K(1, 1)), float(cam.K(0, 2)), float(cam.K(1, 2)));

	auto human_model = renderer.CreateRenderObject("human_model", vs_phong_geometry, fs_phong_geometry);
	human_model->SetIndices(m4c.indices);
	human_model->SetBuffer("positions", m4c.vertices);
	human_model->SetBuffer("normals", m4c.normals);

	Eigen::Matrix4f view_eigen = calcRenderExt(cam.R.cast<float>(), cam.T.cast<float>());
	nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
	renderer.UpdateCanvasView(view_eigen);

	// create offscreen render object, you can render this object to a cuda texture or a cv::Mat
	// In this example code, I render this object to a cv::Mat and then use cv::imshow to demonstrate the rendering results
	// See interfaces of OffScreenRenderObject for more details
	auto human_offscreen = renderer.CreateOffscreenRenderObject(
		"box3", vs_vertex_position, fs_vertex_position, 1920,1080, cam.K(0,0), cam.K(1,1), cam.K(0,2), cam.K(1,2), 1, true);
	human_offscreen->SetIndices(human_model);
	human_offscreen->SetBuffer("positions", human_model);
	human_offscreen->SetBuffer("normals", human_model);
	human_offscreen->_SetViewByCameraRT(cam.R, cam.T);

	cv::Mat rendered_img(1920, 1080, CV_32FC4);
	std::vector<cv::Mat> rendered_imgs;
	rendered_imgs.push_back(rendered_img);

	int frameIdx = 0;
	while (!renderer.ShouldClose())
	{
		renderer.Draw();
		++frameIdx;
		// box3_offscreen to a cv::Mat (offscreen rendering)
		human_offscreen->DrawOffscreen();
		human_offscreen->DownloadRenderingResults(rendered_imgs);
		cv::imshow("rendered img", rendered_imgs[0]);
		std::cout << "img: " << rendered_imgs[0].cols << "," << rendered_imgs[0].rows << std::endl; 

		std::vector<cv::Mat> channels(4);
		cv::split(rendered_imgs[0], channels);
		cv::Mat vis = pseudoColor(channels[2]);
		cv::imshow("depth", vis); 

		cv::waitKey(1);
	}

	//cv::Mat capture = m_renderer.GetFloatImage();
	//cv::Mat depth_vis = pseudoColor(capture); 
	//cv::namedWindow("depth", cv::WINDOW_NORMAL);
	//cv::imshow("depth", depth_vis);
	//cv::waitKey();
	//cv::destroyAllWindows();


	//for (int i = 0; i < vs.cols(); i++)
	//{
	//	Eigen::Vector3d v = vs.col(i).cast<double>();
	//	Eigen::Vector3d uv = project(cams[0], v);
	//	double d = queryDepth(capture, uv(0), uv(1)) * RENDER_FAR_PLANE;
	//	v = cams[0].R * v + cams[0].T; 
	//	std::cout << "d: " << d << "  gt: " << v(2) << std::endl;
	//	if (d > 0 && abs(d - v(2)) < 0.1)
	//	{
	//		colors[i] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
	//	}
	//	else
	//	{
	//		colors[i] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
	//	}
	//}

	return 0;
}