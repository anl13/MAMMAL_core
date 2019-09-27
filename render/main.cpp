#include "renderer.h" 
#include "render_object.h" 
#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <unistd.h> 
#include "../associate/camera.h" 
#include "eigen_util.h"

int main() 
{
    std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
    // init a smpl 
    Camera cam;
    Eigen::Matrix3d K; 
    K << 1499.44, 0, 1014.23, 0, 1499.13, 1003.77, 0,0,1 ; 
    std::cout << K << std::endl; 
    cam.SetK(K); 
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity(); 
    Eigen::Vector3d T = Eigen::Vector3d::Zero(); 
    T(2) = 3; 
    cam.SetRT(R,T); 

    // init renderer 
    Renderer::s_Init(); 
    Renderer m_renderer(conf_projectFolder + "/shader/"); 
    m_renderer.s_camViewer.SetIntrinsic(cam.K.cast<float>(), 2048); 
    m_renderer.s_camViewer.SetExtrinsic(cam.R.cast<float>(), cam.T.cast<float>()); 

    // obj
    const ObjData ballObj(conf_projectFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/data/obj_model/cylinder.obj");
	const ObjData cubeObj(conf_projectFolder + "/data/obj_model/cube.obj");

    RenderObjectTexture* room = new RenderObjectTexture();
	room->SetTexture(conf_projectFolder + "/data/chessboard.png");
	room->SetFaces(cubeObj.faces, true);
	room->SetVertices(cubeObj.vertices);
	room->SetTexcoords(cubeObj.texcoords);
	room->SetTransform({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.InsertObject<RenderObjectTexture>(1, room);

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 
    Shader& depthShader = m_renderer.depthShader; 
    Shader& colorShader = m_renderer.colorShader; 
    Shader& meshShader  = m_renderer.meshShader; 
    Shader& textureShader = m_renderer.textureShader; 

    GLuint& shadowTexture = m_renderer.shadowTexture; 

    Eigen::Vector3f color;
    color(0) = 1; 
    color(1) = 0.1; 
    color(2) = 0.1; 
    std::vector<BallStickObject*> skels; 
    std::vector<Eigen::Vector3f> balls; 
    std::vector< std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks; 
    balls.emplace_back(Eigen::Vector3f::Zero()); 
    balls.emplace_back(Eigen::Vector3f::Ones() / 2); 
    std::pair<Eigen::Vector3f, Eigen::Vector3f> stick = {balls[0], balls[1]};
    sticks.push_back(stick); 
    BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks, 0.045f, 0.02f, color); 
    skels.push_back(skelObject); 

    m_renderer.skels = skels; 
    
    while(!glfwWindowShouldClose(windowPtr))
    {
        // /************************rendering now!!****************************/
        // // set background 
        // glClearColor(0.5f, 0.5f, 0.5f, 1.0f); 
        // // glClearColor(1.0f, 1.0f, 1.0f, 1.0f); 
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

        // // 1. render depth of scene to texture (from light's perspective)
        // // TODO: 
        // glViewport(0, 0, SHADOW_WINDOW_WIDTH, SHADOW_WINDOW_HEIGHT); 
        // glBindFramebuffer(GL_FRAMEBUFFER, m_renderer.shadowFBO); 
        // glClear(GL_DEPTH_BUFFER_BIT); 
        // depthShader.Use(); 

        // Eigen::Vector3f lightPos = - m_renderer.s_camViewer.GetPos(); 
        // // std::cout << "lightPos: " << lightPos.transpose() << std::endl; 
		// depthShader.SetVec3("light_pos", lightPos);
		// depthShader.SetFloat("far_plane", RENDER_FAR_PLANE);
		// {
        //     Eigen::Matrix<float, 4, 4, Eigen::ColMajor> perspective = EigenUtil::Perspective(0.5f*EIGEN_PI, 1.0f, RENDER_NEAR_PLANE, RENDER_FAR_PLANE);

        //     std::vector<Eigen::Vector3f> frontList = { 
        //         {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        //         {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f} };

        //     std::vector<Eigen::Vector3f> upList = {
        //         {0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f },{0.0f, 0.0f, 1.0f},
        //         {0.0f, 0.0f, -1.0f },{ 0.0f, -1.0f, 0.0f },{ 0.0f, -1.0f, 0.0f} };

        //     for (int i = 0; i < 6; i++)
        //     {
        //         const Eigen::Matrix<float, 4, 4, Eigen::ColMajor> shadowTransform = perspective * EigenUtil::LookAt(lightPos, lightPos + frontList[i], upList[i]);
        //         depthShader.SetMat4("shadow_matrices[" + std::to_string(i) + "]", shadowTransform);
        //     }
		// }


        // for(auto iter = m_renderer.renderObjects.begin(); iter!=m_renderer.renderObjects.end(); iter++)
        // {
        //     iter->second->DrawDepth(depthShader); 
        // }
        // for(int i = 0; i < skels.size(); i++)
        // {
        //     skels[i]->Draw(depthShader); 
        // }
		// glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// // 2. render scene as normal using the generated depth/shadow map  
		// // --------------------------------------------------------------

        // for(auto iter = m_renderer.renderObjects.begin(); iter!=m_renderer.renderObjects.end(); iter++)
        // {
        //     textureShader.Use(); 
        //     m_renderer.s_camViewer.ConfigShader(textureShader); 
        //     textureShader.SetVec3("light_pos", lightPos); 
        //     textureShader.SetFloat("far_plane", RENDER_FAR_PLANE); 
        //     textureShader.SetInt("depth_cube", 1); 
        //     glActiveTexture(GL_TEXTURE1); 
        //     glBindTexture(GL_TEXTURE_CUBE_MAP, shadowTexture); 
        //     textureShader.SetInt("object_texture", 2); 
        //     glActiveTexture(GL_TEXTURE2); 
        //     iter->second->DrawWhole(textureShader); 
        // }
        // for(int i = 0; i < skels.size(); i++)
        // {
        //     colorShader.Use(); 
        //     m_renderer.s_camViewer.ConfigShader(colorShader); 
        //     colorShader.SetVec3("light_pos", lightPos); 
        //     colorShader.SetFloat("far_plane", RENDER_FAR_PLANE); 
        //     colorShader.SetInt("depth_cube", 1); 
        //     glActiveTexture(GL_TEXTURE1); 
        //     glBindTexture(GL_TEXTURE_CUBE_MAP, shadowTexture); 
            
        //     skels[i]->Draw(colorShader); 
        // }
        m_renderer.Draw(); 

        glfwSwapBuffers(windowPtr); 
        glfwPollEvents(); 
    };

}