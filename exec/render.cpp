#include "../render/renderer.h" 
#include "../render/render_object.h" 
#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <unistd.h> 
#include "../associate/camera.h"
#include "../associate/image_utils.h"
#include "../associate/framedata.h" 
#include "../associate/math_utils.h"
#include "../render/eigen_util.h"
#include "../render/render_utils.h" 
#include "../smal/smal.h" 
#include "../associate/colorterminal.h" 

#include <gflags/gflags.h> 

DEFINE_string(type, "smal", "hint which func to use"); 

// #define DEBUG_RENDER

const float c_floor_dx = 0.28; 
const float c_floor_dy = 0.2; 

int render_animal_skels() 
{
    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
    auto CM = getColorMapEigen("anliang_rgb"); 

#ifndef DEBUG_RENDER
    FrameData frame; 
    std::string jsonfile = "/home/al17/animal/animal_calib/build/results/json/003732.json"; 
    frame.configByJson("/home/al17/animal/animal_calib/associate/config.json"); 
    frame.setFrameId(3732); 
    frame.fetchData(); 
    frame.readSkelfromJson(jsonfile); 
    std::cout << frame.m_skels.size() << std::endl; 
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
	chess_floor->SetTransform({ c_floor_dx, c_floor_dy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
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


int render_proposals()
{
    FrameData frame; 
    std::string configFile = "/home/al17/animal/animal_calib/associate/config.json"; 
    frame.configByJson(configFile); 

    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
    auto CM = getColorMapEigen("anliang_rgb"); 
    auto CM2 = getColorMapEigen("jet"); 
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

    int frameid = frame.startid; 
    for(frameid = frame.startid; frameid < frame.startid + frame.framenum; frameid++)
    {
        int delta = frameid - frame.startid; 
        frame.setFrameId(frameid); 
        std::cout << "set frame id: " << frameid << std::endl; 
        frame.fetchData(); 
        std::cout << "fetch data" << std::endl; 

        frame.epipolarSimilarity(); 
        frame.ransacProposals(); 
        
        
        std::vector<int> color_kpt_id = frame.m_kpt_color_id; 
        std::vector<std::vector<ConcensusData> > data = frame.m_concensus;

        std::vector<Eigen::Vector3f> balls; 
        std::vector<float> sizes; 
        std::vector<int> color_ids; 
        GetBalls(data, color_kpt_id, balls, sizes, color_ids); 
        std::vector<Eigen::Vector3f> colors; 
        // for(int i = 0; i < color_ids.size(); i++) colors.push_back(CM[ color_ids[i]]); 
        for(int i = 0; i < color_ids.size(); i++) colors.push_back(CM2[ delta ]); 

        std::cout << "balls: " << balls.size() << std::endl; 
        
        BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors); 
        m_renderer.skels.push_back(skelObject); 

    }
    
    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 
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
    std::cout << "In render scene now!" << std::endl; 

    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
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
	chess_floor->SetTransform({ c_floor_dx, c_floor_dy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
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

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // render_smal_test(); 
    render_proposals(); 
    // renderScene(); 
}