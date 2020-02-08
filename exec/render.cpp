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
#include "../smal/smal.h" 
#include "../utils/colorterminal.h" 

#include <gflags/gflags.h> 

DEFINE_string(type, "smal", "smal: show smal");
// #define DEBUG_RENDER

const float kFloorDx = 0.28; 
const float kFloorDy = 0.2; 

int render_animal_skels() 
{

    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
    auto CM = getColorMapEigen("anliang_rgb", true); 

#ifndef DEBUG_RENDER
    FrameData frame; 
    frame.configByJson("/home/al17/animal/animal_calib/associate/config.json"); 
#endif 
    std::string videoname_render = "/home/al17/animal/animal_calib/result_data/render.avi"; 
    cv::VideoWriter writer_render(videoname_render, cv::VideoWriter::fourcc('H', '2', '6', '4'), 25.0, cv::Size(1024, 1024)); 
    if(!writer_render.isOpened())
    {
        std::cout << "can not open video file " << videoname_render << std::endl; 
        return -1; 
    }
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
	chess_floor->SetTransform({ kFloorDx, kFloorDy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor); 

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 

    int startid = frame.get_start_id(); 
    int framenum = frame.get_frame_num(); 
    SkelTopology topo = frame.get_topo(); 
    vector<int> kpt_color_ids = topo.kpt_color_ids; 
    for(int frameid = startid; frameid < startid + framenum; frameid++)
    {
        m_renderer.skels.clear(); 

        std::stringstream ss; 
        ss << "/home/al17/animal/animal_calib/result_data/skels3d/skel_" << std::setw(6) << std::setfill('0') << frameid << ".json"; 
        frame.readSkel3DfromJson(ss.str()); 
        auto data = frame.get_skels3d(); 
        for(int i = 0; i < data.size(); i++)
        {
            std::vector<Eigen::Vector3f> balls; 
            std::vector< std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks; 
            GetBallsAndSticks(data[i], topo.bones, balls, sticks); 
            Eigen::Vector3f color = CM[i]; 
            BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks, 0.02f, 0.01f, color); 
            m_renderer.skels.push_back(skelObject); 
        }

        Eigen::Vector3f camPos; camPos << -0.94904, 0.692267, -1.43164;
        Eigen::Vector3f camUp; camUp << 0.728626, -0.323766, -0.603555;
        Eigen::Vector3f camCen; camCen << 0.133968,  0.318453, 0.0778542;

        m_renderer.s_camViewer.SetExtrinsic(camPos, camUp, camCen); 
        // while(!glfwWindowShouldClose(windowPtr))
        // {
            m_renderer.Draw(); 
            cv::Mat capture = m_renderer.GetImage(); 
            // std::stringstream ss_img;
            writer_render.write(capture); 
            // ss_img << "/home/al17/animal/animal_calib/result_data/render/render_" << std::setw(6) << std::setfill('0') << frameid << ".png";
            // cv::imwrite(ss_img.str(), capture);  
            glfwSwapBuffers(windowPtr); 
            glfwPollEvents(); 
        // };
    }

    return 0; 
}

int render_smal_test()
{
    std::cout << "In smal render now!" << std::endl; 

    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
    auto CM = getColorMapEigen("jet"); 

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

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 

    /// read smal model 
    std::string smal_folder = "/home/al17/animal/smal/smal_online_V1.0/smalr_txt";
    Eigen::VectorXd pose = Eigen::VectorXd::Random(99) * 0.1; 
    Eigen::VectorXd shape = Eigen::VectorXd::Random(41) * 0.1; 
    SMAL smal(smal_folder); 
    smal.SetPose(pose); 
    smal.SetShape(shape); 
    smal.UpdateVertices();

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

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    render_animal_skels(); 
    // render_smal_test(); 
}