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
// #define DEBUG_RENDER

int render_smal_test()
{
    std::cout << "In smal render now!" << std::endl; 

    std::string conf_projectFolder = "D:/projects/animal_calib/render";
	std::string pig_config = "D:/Projects/animal_calib/smal/smal_config.json";
	auto CM = getColorMapEigen("jet"); 

    // init a camera 
    Eigen::Matrix3f K; 
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
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

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 
	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));

    /// read smal model 
    Eigen::VectorXd pose = Eigen::VectorXd::Random(99) * 0.1; 
    Eigen::VectorXd shape = Eigen::VectorXd::Random(41) * 0.1; 
    //SMAL smal(smal_folder); 
	PigModel smal(pig_config); 
    //smal.SetPose(pose); 
    //smal.SetShape(shape); 
    smal.UpdateVertices();

	std::string type = "smal";
    if(type == "smal")
    {
        RenderObjectColor* smal_render = new RenderObjectColor(); 
        auto faces = smal.GetFaces(); 
        Eigen::MatrixXf vs    = smal.GetVertices().cast<float>(); 
		vs = vs.colwise() - vs.col(0); 
		std::cout << faces.transpose() << std::endl;
        smal_render->SetFaces(faces); 
        smal_render->SetVertices(vs);
        Eigen::Vector3f color; 
        color << 0.8f, 0.5f, 0.4f; 
        smal_render->SetColor(color); 
        m_renderer.colorObjs.push_back(smal_render); 
    }
    else if (type=="skel")
    {
        std::vector<Eigen::Vector3f> colors; 
        for(int i = 0; i < 33; i++) colors.push_back(CM[7 * i]); 
        colors[0] = Eigen::Vector3f(1.0,0.0,0.0); 
        std::vector<float> sizes(33, 0.03); 
        std::vector<Eigen::Vector3f> balls; 
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks; 
        Eigen::VectorXi parents = smal.GetParents(); 
        Eigen::Matrix3Xf joints = smal.GetJoints().cast<float>(); 
        std::cout << "parents: " << parents.transpose() << std::endl;
        std::cout << "joints: " << std::endl << joints << std::endl; 
        GetBallsAndSticks(joints, parents, balls, sticks); 
        
        // BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks, 0.02f, 0.01f, color); 
        BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors); 
        m_renderer.skels.push_back(skelObject); 
    }


    while(!glfwWindowShouldClose(windowPtr))
    {
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

