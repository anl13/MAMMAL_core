#include "renderer.h" 
#include "render_object.h" 
#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <unistd.h> 

int main() 
{
    std::string conf_projectFolder = "/home/al17/animal/animal_calib";
    // init a smpl 

    // init renderer 
    Renderer::s_Init(); 
    Renderer::s_InitCamViewer(Eigen::Vector3f(10.0f, 2.5f, 10.0f), 
                              Eigen::Vector3f(0.0f, 1.0f, 0.0f), 
                              Eigen::Vector3f::Zero()); 
    Renderer *m_rendererPtr = new Renderer(Eigen::Vector3f(2.0f, 4.0f, 2.0f), 
                                 50.0f, 
                                 conf_projectFolder+"/render/"); 

    RenderObject* room = new RenderObject(
        ObjData(conf_projectFolder + "/render/data/cube.obj", true),
        conf_projectFolder + "/render/data/chessboard.png",
        MaterialParam(1.5f, 1.0f, 1.0f, 1.0f));
    room->SetTransform({ 0.0f, 3.8f,0.0f }, { 0.0f, 0.0f, 0.0f }, 5.0f);
    m_rendererPtr->UploadStaticObject<RenderObject>(room);

    GLFWwindow* windowPtr = m_rendererPtr->s_windowPtr; 
    std::string ballObjPath = conf_projectFolder + "/render/data/ball.obj"; 
    std::string stickObjPath = conf_projectFolder + "/render/data/cylinder.obj"; 
    ObjData ballObj(ballObjPath, false);
	ObjData stickObj(stickObjPath, false);
    std::vector<Eigen::Vector3f> balls;
    std::vector<std::pair<int, int>> sticks;
    balls.emplace_back(Eigen::Vector3f::Zero()); 
    balls.emplace_back(Eigen::Vector3f::Ones()); 
    sticks.emplace_back({0,1});

    // std::vector<RenderObject*> objectPtrs; 
    // while (!glfwWindowShouldClose(windowPtr))
	// {
        
	// }
    // if(m_rendererPtr)
    //     delete m_rendererPtr; 
}