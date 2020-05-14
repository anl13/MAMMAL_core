#include "../render/renderer.h" 
#include "../render/render_object.h" 
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
#include "../render/eigen_util.h"
#include "../render/render_utils.h" 
#include "../utils/colorterminal.h" 
#include "../smal/pigmodel.h"
#include "../utils/node_graph.h"
#include "../utils/model.h"

int render_smal_test()
{
    std::cout << "In smal render now!" << std::endl; 

    std::string conf_projectFolder = "D:/projects/animal_calib/render";
	std::string smal_config = "D:/Projects/animal_calib/smal/smal2_config.json";
	auto CM = getColorMapEigen("anliang_rgb"); 
	SkelTopology m_topo = getSkelTopoByType("UNIV");
    /// read smal model 
	//PigModel smal(smal_config); 
	PigSolver smal(smal_config);
	std::string folder = smal.GetFolder();
	//smal.readShapeParam(folder + "pigshape.txt");
    smal.UpdateVertices();

	// init render
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
	Renderer m_renderer(conf_projectFolder + "/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	// init element obj
	const ObjData ballObj(conf_projectFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/data/obj_model/square.obj");

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.2f, 0.8f, 0.2f, 1.0f));

	//RenderObjectTexture* chess_floor = new RenderObjectTexture();
	//chess_floor->SetTexture(conf_projectFolder + "/data/chessboard.png");
	//chess_floor->SetFaces(squareObj.faces, true);
	//chess_floor->SetVertices(squareObj.vertices);
	//chess_floor->SetTexcoords(squareObj.texcoords);
	//chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	//m_renderer.texObjs.push_back(chess_floor);

    Eigen::Matrix<unsigned int,-1,-1, Eigen::ColMajor> faces = smal.GetFacesVert(); 
    Eigen::MatrixXf vs    = smal.GetVertices().cast<float>(); 
	Eigen::MatrixXd vsd = smal.GetVertices(); 

	RenderObjectColor* smal_render = new RenderObjectColor();
    smal_render->SetFaces(faces); 
    smal_render->SetVertices(vs);
    Eigen::Vector3f color; 
    color << 0.8f, 0.8f, 0.8f; 
    smal_render->SetColor(color); 
    m_renderer.colorObjs.push_back(smal_render); 

	//// // show joints
	//Eigen::MatrixXf joints = smal.GetJoints().cast<float>(); 
	//int jointNum = smal.GetJointNum(); 
	//Eigen::VectorXi parents = smal.GetParents(); 
	//std::vector<float> sizes(jointNum, 0.03);
	//std::vector<Eigen::Vector3f> balls;
	//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	//std::vector<Eigen::Vector3f> colors(jointNum, Eigen::Vector3f(0.8, 0.8, 0.8));
	//for (int i = 0; i < jointNum; i++)
	//{
	//	colors[i] = CM[i];
	//}
	//GetBallsAndSticks(joints, parents, balls, sticks);
	//BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors);
	//m_renderer.skels.push_back(skelObject); 

	//// // show joints
	//Eigen::MatrixXf joints = smal.getRegressedSkelbyPairs().cast<float>();
	//int jointNum = m_topo.joint_num;
	//Eigen::VectorXi parents;
	//std::vector<float> sizes(jointNum, 0.03);
	//std::vector<Eigen::Vector3f> balls;
	//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	//std::vector<Eigen::Vector3f> colors(jointNum, Eigen::Vector3f(0.8, 0.8, 0.8));
	//for (int i = 0; i < jointNum; i++)
	//{
	//	colors[i] = CM[i];
	//}
	//GetBallsAndSticks(joints, parents, balls, sticks);
	//BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors);
	//m_renderer.skels.push_back(skelObject);

    while(!glfwWindowShouldClose(windowPtr))
    {
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        m_renderer.Draw(); 
        glfwSwapBuffers(windowPtr); 
        glfwPollEvents(); 
    };
	
    return 0; 
}

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


int test_depth()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/render";
	std::string smal_config = "D:/Projects/animal_calib/smal/smal2_config.json";
	auto CM = getColorMapEigen("anliang_rgb");
	SkelTopology m_topo = getSkelTopoByType("UNIV");
	/// read smal model 
	//PigModel smal(smal_config); 
	PigSolver smal(smal_config);
	std::string folder = smal.GetFolder();
	//smal.readShapeParam(folder + "pigshape.txt");
	smal.UpdateVertices();

	// init render
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
	Renderer m_renderer(conf_projectFolder + "/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	// init element obj
	const ObjData ballObj(conf_projectFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/data/obj_model/square.obj");

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.f, 0.f, 0.f, 1.0f));

	//RenderObjectColor* floor = new RenderObjectColor();
	//floor->SetFaces(squareObj.faces);
	//floor->SetVertices(squareObj.vertices);
	//Eigen::Vector3f c(0.8f, 0.8f, 0.8f);
	//floor->SetColor(c);
	//m_renderer.colorObjs.push_back(floor);

	Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces = smal.GetFacesVert();
	Eigen::MatrixXf vs = smal.GetVertices().cast<float>();
	Eigen::MatrixXd vsd = smal.GetVertices();

	RenderObjectColor* smal_render = new RenderObjectColor();
	smal_render->SetFaces(faces);
	smal_render->SetVertices(vs);
	Eigen::Vector3f color;
	color << 0.8f, 0.8f, 0.8f;
	smal_render->SetColor(color);
	m_renderer.colorObjs.push_back(smal_render);

	//while (!glfwWindowShouldClose(windowPtr))
	//{
	//	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//	m_renderer.Draw();
	//	glfwSwapBuffers(windowPtr);
	//	glfwPollEvents();
	//};
	FrameData frame;
	frame.configByJson("D:/Projects/animal_calib/associate/config.json");
	frame.set_frame_id(0); 
	frame.fetchData();
	auto cams = frame.get_cameras();
	m_renderer.s_camViewer.SetExtrinsic(cams[0].R.cast<float>(), cams[0].T.cast<float>());
	m_renderer.Draw("depth");
	
	cv::Mat capture = m_renderer.GetFloatImage();
	cv::Mat depth_vis = pseudoColor(capture); 
	cv::namedWindow("depth", cv::WINDOW_NORMAL);
	cv::imshow("depth", depth_vis);
	cv::waitKey();
	cv::destroyAllWindows();

	int vertexNum = vs.cols();
	std::vector<Eigen::Vector3f> colors(vertexNum, Eigen::Vector3f(1.0f, 1.0f, 1.0f));
	std::vector<float> sizes(vertexNum, 0.006);
	std::vector<Eigen::Vector3f> balls;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	Eigen::VectorXi parents;
	GetBallsAndSticks(vs, parents, balls, sticks);

	for (int i = 0; i < vs.cols(); i++)
	{
		Eigen::Vector3d v = vs.col(i).cast<double>();
		Eigen::Vector3d uv = project(cams[0], v);
		double d = queryDepth(capture, uv(0), uv(1)) * RENDER_FAR_PLANE;
		v = cams[0].R * v + cams[0].T; 
		std::cout << "d: " << d << "  gt: " << v(2) << std::endl;
		if (d > 0 && abs(d - v(2)) < 0.1)
		{
			colors[i] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
		}
		else
		{
			colors[i] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
		}
	}
	BallStickObject* pc = new BallStickObject(ballObj, balls, sizes, colors);
	m_renderer.skels.push_back(pc);

	while (!glfwWindowShouldClose(windowPtr))
	{
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

	return 0;
}