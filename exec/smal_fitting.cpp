#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include "../utils/colorterminal.h" 
#include "../utils/obj_reader.h"
#include "../render/render_object.h"
#include "../render/render_utils.h"
#include "../render/renderer.h"

#include "../associate/framedata.h"
#include "../smal/pigmodel.h"
#include "../smal/pigsolver.h"
#include "main.h"

using std::vector; 

int TestVerticesAlign()
{   
    std::string target_file = "D:/Projects/animal_calib/data/pig_B_tpose.obj"; 
    OBJReader objreader; 
    objreader.read(target_file); 

    /// read smal model 
    std::string smal_config = "D:/Projects/animal_calib/smal/smal_config.json";
    Eigen::VectorXd pose = Eigen::VectorXd::Random(99) * 0.2;
    pose.segment<9>(24); 
    pose.segment<9>(36); 
    Eigen::VectorXd shape = Eigen::VectorXd::Random(41) * 0.1; 
    PigModel smal(smal_config); 
    smal.readState(); 
    smal.UpdateVertices();

    PigSolver solver(smal_config); 
    solver.setTargetVSameTopo(objreader.vertices_eigen.cast<double>()); 
    solver.globalAlignToVerticesSameTopo(); 
    // solver.OptimizePose(50, 0.001); 
    solver.FitShapeToVerticesSameTopo(50, 0.00000001); 
    solver.saveState("pig_shape.txt"); 

    //// rendering pipeline. 
    std::string renderFolder = "D:/Projects/animal_calib/render";
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
    Renderer m_renderer(renderFolder + "/shader/"); 
    m_renderer.s_camViewer.SetIntrinsic(K, 1, 1); 
    m_renderer.s_camViewer.SetExtrinsic(pos, up, center); 

    // init element obj
    const ObjData ballObj(renderFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(renderFolder + "/data/obj_model/cylinder.obj");
    const ObjData squareObj(renderFolder + "/data/obj_model/square.obj"); 

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 

    RenderObjectColor* smal_render = new RenderObjectColor(); 
    Eigen::Matrix<unsigned int,-1,-1,Eigen::ColMajor> faces = solver.GetFacesTex(); 
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