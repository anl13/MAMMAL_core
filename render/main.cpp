#include "renderer.h" 
#include "render_object.h" 
#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <unistd.h> 
#include "../associate/camera.h"
#include "../associate/image_utils.h"
#include "../associate/framedata.h" 
#include "../associate/math_utils.h"
#include "eigen_util.h"
#include "render_utils.h" 
#include "../smal/smal.h" 

#include <gflags/gflags.h> 

DEFINE_string(type, "smal", "hint which func to use"); 

// #define DEBUG_RENDER

int render_animal_skels() 
{
    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
    auto CM = getColorMapEigen(); 

#ifndef DEBUG_RENDER
    FrameData frame; 
    std::string jsonfile = "/home/al17/animal/animal_calib/build/results/json/003732.json"; 
    frame.configByJson("/home/al17/animal/animal_calib/associate/config.json"); 
    frame.setFrameId(3732); 
    frame.fetchData(); 
    frame.readSkelfromJson(jsonfile); 
    std::cout << frame.m_skels.size() << std::endl; 
    Camera frameCam = frame.getCamUndistById(1); 
#endif 

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
	const ObjData cubeObj(conf_projectFolder + "/data/obj_model/cube.obj");
    const ObjData squareObj(conf_projectFolder + "/data/obj_model/square.obj"); 

    RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, true);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor); 

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 

#ifdef DEBUG_RENDER
    Eigen::Vector3f color;
    color(0) = 1; 
    color(1) = 0.1; 
    color(2) = 0.1; 
    std::vector<BallStickObject*> skels; 
    std::vector<Eigen::Vector3f> balls; 
    std::vector< std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks; 
    Eigen::Vector3f p1 = Eigen::Vector3f::Zero(); 
    Eigen::Vector3f p2 = Eigen::Vector3f::Ones();
    balls.push_back(p1); balls.push_back(p2); 
    std::pair<Eigen::Vector3f, Eigen::Vector3f> stick; stick.first = p1; stick.second = p2; 
    sticks.push_back(stick);
    
    BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks, 0.045f, 0.02f, color); 
    skels.push_back(skelObject); 

    m_renderer.skels = skels; 
#else
    for(int i = 0; i < 4; i++)
    {
        std::vector<Eigen::Vector3f> balls; 
        std::vector< std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks; 
        GetBallsAndSticks(frame.m_skels[i], BONES, balls, sticks); 
        Eigen::Vector3f color = CM[i]; 
        BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks, 0.02f, 0.01f, color); 
        m_renderer.skels.push_back(skelObject); 
    }

#endif 
    while(!glfwWindowShouldClose(windowPtr))
    {
        m_renderer.Draw(); 

        glfwSwapBuffers(windowPtr); 
        glfwPollEvents(); 
    };

    return 0; 
}

int render_smal_test()
{
    std::cout << "In smal render now!" << std::endl; 

    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
    auto CM = getColorMapEigen(); 

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

    // RenderObjectTexture* chess_floor = new RenderObjectTexture();
	// chess_floor->SetTexture(conf_projectFolder + "/data/chessboard.png");
	// chess_floor->SetFaces(squareObj.faces, true);
	// chess_floor->SetVertices(squareObj.vertices);
	// chess_floor->SetTexcoords(squareObj.texcoords);
	// chess_floor->SetTransform({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	// m_renderer.texObjs.push_back(chess_floor); 

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 

    // for(int i = 0; i < 4; i++)
    // {
    //     std::vector<Eigen::Vector3f> balls; 
    //     std::vector< std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks; 
    //     GetBallsAndSticks(frame.m_skels[i], BONES, balls, sticks); 
    //     Eigen::Vector3f color = CM[i]; 
    //     BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks, 0.02f, 0.01f, color); 
    //     m_renderer.skels.push_back(skelObject); 
    // }

    /// read smal model 
    std::string smal_folder = "/home/al17/animal/smal/smal_online_V1.0/smalr_txt";
    SMAL smal(smal_folder); 

    if(FLAGS_type == "smal")
    {
        RenderObjectColor* smal_render = new RenderObjectColor(); 
        auto faces = smal.GetFaces(); 
        Eigen::MatrixXf vs    = smal.GetVertices().cast<float>(); 
        Eigen::Vector3f c = vs.col(0); 
        smal_render->SetFaces(faces); 
        smal_render->SetVertices(vs);
        Eigen::Vector3f color; 
        color << 1.0f, 0.0f, 0.0f; 
        smal_render->SetColor(color); 
        m_renderer.colorObjs.push_back(smal_render); 
    }
    else if (FLAGS_type=="skel")
    {
        std::vector<Eigen::Vector3f> balls; 
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks; 
        Eigen::VectorXi parents = smal.GetParents(); 
        Eigen::Matrix3Xf joints = smal.GetJoints().cast<float>(); 
        std::cout << "parents: " << parents.transpose() << std::endl;
        std::cout << "joints: " << std::endl << joints << std::endl; 
        GetBallsAndSticks(joints, parents, balls, sticks); 
        Eigen::Vector3f color = CM[0]; 
        BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks, 0.02f, 0.01f, color); 
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

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    render_smal_test(); 
}