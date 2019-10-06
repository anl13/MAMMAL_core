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

#define DEBUG_RENDER

void GetBallsAndSticks(
    const PIG_SKEL& joints, 
    const std::vector<Eigen::Vector2i>& bones, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks 
    )
{
    balls.clear(); 
    sticks.clear(); 
    for(int i = 0; i < joints.cols(); i++)
    {
        Eigen::Vector4d p = joints.col(i); 
        if(p(3) > 0) // joint is valid 
        {
            Eigen::Vector3f pf = p.segment<3>(0).cast<float>();
            pf(0) = pf(0); 
            pf(1) = -pf(1); 
            pf(2) = -pf(2);
            balls.push_back(pf);
        }
    }
    
    for(int i = 0; i < bones.size(); i++)
    {
        int sid = bones[i](0);
        int eid = bones[i](1); 
        Eigen::Vector4d ps = joints.col(sid); 
        Eigen::Vector4d pe = joints.col(eid); 
        ps(1) = -ps(1); 
        ps(2) = -ps(2); 
        pe(1) = -pe(1); 
        pe(2) = -pe(2); 
        if(ps(3) > 0 && pe(3) > 0)
        {
            std::pair<Eigen::Vector3f, Eigen::Vector3f> stick = 
            {
                ps.segment<3>(0).cast<float>(), 
                pe.segment<3>(0).cast<float>()
            }; 
            sticks.push_back(stick); 
        }
    }
}

std::vector<Eigen::Vector3f> getColorMapEigen()
{
    std::vector<Eigen::Vector3i> cm; 
    getColorMap("anliang", cm); 
    std::vector<Eigen::Vector3f> cm_float;
    cm_float.resize(cm.size()); 
    for(int i = 0; i < cm.size(); i++)
    {
        Eigen::Vector3f c;
        c(0) = float(cm[i](2)); 
        c(1) = float(cm[i](1)); 
        c(2) = float(cm[i](0));
        c = c / 255.0f; 
        cm_float[i] = c;  
    }
    return cm_float; 
}

class CombinedObj
{
public: 
    CombinedObj(){facenum=0; vertexnum=0;}; 
    ~CombinedObj(){}; 
    int facenum; 
    int vertexnum; 
    
};

int main() 
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
    Camera cam;
    Eigen::Matrix3d K; 
    K << 1499.44, 0, 1014.23, 0, 1499.13, 1003.77, 0,0,1 ; 
    std::cout << K << std::endl; 
    cam.SetK(K); 
    Vec3 rvec; rvec << 0, -1.57,0; 
    Eigen::Matrix3d R1 = GetRodrigues(rvec);
    Vec3 rvec2; rvec2 << 1.57, 0, 0; 
    Eigen::Matrix3d R2 = GetRodrigues(rvec2);   
    Eigen::Vector3d T = Eigen::Vector3d::Zero(); 
    T(2) = 2; 
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity(); 
    cam.SetRT(I,T); 
    // cam.SetRT(frameCam.R, frameCam.T); 

    // init renderer 
    Renderer::s_Init(); 
    Renderer m_renderer(conf_projectFolder + "/shader/"); 
    m_renderer.s_camViewer.SetIntrinsic(cam.K.cast<float>(), 2048); 
    m_renderer.s_camViewer.SetExtrinsic(cam.R.cast<float>(), cam.T.cast<float>()); 

    // init element obj
    const ObjData ballObj(conf_projectFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/data/obj_model/cylinder.obj");
	const ObjData cubeObj(conf_projectFolder + "/data/obj_model/cube.obj");

    // init scene background 
    RenderObjectTexture* room = new RenderObjectTexture();
	room->SetTexture(conf_projectFolder + "/data/chessboard.png");
	room->SetFaces(cubeObj.faces, true);
	room->SetVertices(cubeObj.vertices);
	room->SetTexcoords(cubeObj.texcoords);
	room->SetTransform({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.InsertObject<RenderObjectTexture>(1, room);

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
        std::cout << frame.m_skels[i] << std::endl;
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

}