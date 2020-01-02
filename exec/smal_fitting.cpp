#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include "../smal/smal.h" 
#include "../utils/colorterminal.h" 
#include "../utils/obj_reader.h"
#include "../render/render_object.h"
#include "../render/render_utils.h"
#include "../render/renderer.h"
#include "../smal/smal_solver.h"
#include "../smal/smal_jsolver.h" 
#include "../associate/framedata.h"

using std::vector; 
const float kFloorDx = 0.28; 
const float kFloorDy = 0.2; 

int TestVerticesAlign()
{   
    std::string target_file = "../data/pig_B_tpose.obj"; 
    OBJReader objreader; 
    objreader.read(target_file); 

    /// read smal model 
    std::string smal_folder = "/home/al17/animal/smal/smal_online_V1.0/smalr_txt";
    Eigen::VectorXd pose = Eigen::VectorXd::Random(99) * 0.2;
    pose.segment<9>(24); 
    pose.segment<9>(36); 
    Eigen::VectorXd shape = Eigen::VectorXd::Random(41) * 0.1; 
    SMAL smal(smal_folder); 
    smal.readState(); 
    smal.UpdateVertices();

    SMAL_SOLVER solver(smal_folder); 
    solver.setY(objreader.vertices_eigen.cast<double>()); 
    solver.globalAlignByVertices(); 
    // solver.OptimizePose(50, 0.001); 
    solver.OptimizeShape(50, 0.00000001); 
    solver.saveState("pig_shape.txt"); 

    //// rendering pipeline. 
    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
    auto CM = getColorMapEigen("jet"); 

    // init a camera 
    Eigen::Matrix3f K; 
    K << 0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0, 1;
    // std::cout << K << std::endl; 

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

    RenderObjectColor* smal_render = new RenderObjectColor(); 
    Eigen::Matrix<unsigned int,-1,-1,Eigen::ColMajor> faces = solver.GetFaces(); 
    Eigen::MatrixXf vs = solver.GetVertices().cast<float>(); 
    smal_render->SetFaces(faces); 
    smal_render->SetVertices(vs);
    Eigen::Vector3f color; 
    color << 1.0f, 0.0f, 0.0f; 
    smal_render->SetColor(color); 
    m_renderer.colorObjs.push_back(smal_render); 

    RenderObjectColor* smal_target = new RenderObjectColor(); 
    Eigen::MatrixXf vs_gt = objreader.vertices_eigen; 
    // Eigen::MatrixXf vs_gt = smal.GetVertices().cast<float>(); 
    smal_target->SetFaces(faces); 
    smal_target->SetVertices(vs_gt); 
    color << 0.0f, 0.6f, 0.0f;
    smal_target->SetColor(color); 
    m_renderer.colorObjs.push_back(smal_target);  

    while(!glfwWindowShouldClose(windowPtr))
    {
        m_renderer.Draw(); 

        glfwSwapBuffers(windowPtr); 
        glfwPollEvents(); 
    };
    return 0; 
}

std::vector<std::pair<int,int> > M = {
    {1, 1863},
    {1, 1089}, 
    {1, 3050}, 
    {1, 486}, 
    {1, 2451}, 
    {0, 8},
    {0, 12},
    {0, 9},
    {0, 13}, 
    {0, 10},
    {0, 14}, 
    {0, 18},
    {0, 22},
    {0, 19},
    {0, 23}, 
    {0, 20},
    {0, 24}, 
    {-1, -1}, 
    {0, 25}, 
    {0, 6}, 
    {0, 4},
    {-1, -1},
    {-1, -1}
}; 


int TestKeypointsAlign()
{
    SkelTopology topo = getSkelTopoByType("UNIV"); 
    FrameData frame;
    frame.configByJson("/home/al17/animal/animal_calib/associate/config.json");
    int frameid = 0; 
    std::stringstream ss; 
    ss << "/home/al17/animal/animal_calib/result_data/skels3d/skel_" 
       << std::setw(6) << std::setfill('0') << frameid << ".json"; 
    frame.readSkel3DfromJson(ss.str()); 
    std::vector<std::vector<Eigen::Vector3d> > data = frame.get_skels3d(); 
    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(3, data[0].size()); 
    for(int i = 0; i < data[0].size(); i++) Y.col(i) = data[0][i]; 
    
    std::string smal_folder = "/home/al17/animal/smal/smal_online_V1.0/smalr_txt";
    SMAL_JSOLVER solver(smal_folder); 
    solver.readState("../data/pig_shape.txt");
    solver.setY(Y); 
    solver.setMapper(M); 
    solver.globalAlign(); 
    solver.optimizePose(50, 0.001); 

    //// rendering pipeline. 
    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
    auto CM = getColorMapEigen("jet"); 

    // init a camera 
    Eigen::Matrix3f K; 
    K << 0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0, 1;
    // std::cout << K << std::endl; 

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

    RenderObjectColor* smal_render = new RenderObjectColor(); 
    Eigen::Matrix<unsigned int,-1,-1,Eigen::ColMajor> faces = solver.GetFaces(); 
    Eigen::MatrixXf vs = solver.GetVertices().cast<float>(); 
    smal_render->SetFaces(faces); 
    smal_render->SetVertices(vs);
    smal_render->SetColor(CM[220]); 
    m_renderer.colorObjs.push_back(smal_render); 

    std::vector<Eigen::Vector3f> balls; 
    std::vector< std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks; 
    GetBallsAndSticks(data[0], topo.bones, balls, sticks); 
    BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks, 
                                0.02f, 0.01f, CM[0]); 
    m_renderer.skels.push_back(skelObject); 

    Eigen::Vector3f camPos; camPos << -0.94904, 0.692267, -1.43164;
    Eigen::Vector3f camUp; camUp << 0.728626, -0.323766, -0.603555;
    Eigen::Vector3f camCen; camCen << 0.133968,  0.318453, 0.0778542;

    m_renderer.s_camViewer.SetExtrinsic(camPos, camUp, camCen); 

    while(!glfwWindowShouldClose(windowPtr))
    {
        m_renderer.Draw(); 

        glfwSwapBuffers(windowPtr); 
        glfwPollEvents(); 
    };
    return 0; 
}

int main()
{
    TestKeypointsAlign(); 
}
