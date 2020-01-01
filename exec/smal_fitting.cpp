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

using std::vector; 

int main()
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
    solver.setY(smal.GetVertices()); 
    // solver.SetPose(pose); 
    // solver.readState("gt.txt"); 
    // solver.debug(); 
    // return -1; 
    // solver.globalAlignByVertices(); 
    solver.OptimizePose(50, 0.001); 

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
    // Eigen::MatrixXf vs_gt = objreader.vertices_eigen; 
    Eigen::MatrixXf vs_gt = smal.GetVertices().cast<float>(); 
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