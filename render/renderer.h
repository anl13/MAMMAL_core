#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <Eigen/Eigen>
#include <map> 

#include "camviewer.h"
#include "render_object.h"
#include "shader.h"
#include "eigen_util.h"

// #define WINDOW_SIZE 800
// #define SHADOW_WINDOW_SIZE 1024


// #define WINDOW_SIZE 512
// #define SHADOW_WINDOW_SIZE 512
 #define WINDOW_HEIGHT 1080
 #define WINDOW_WIDTH  1920
 #define SHADOW_WINDOW_HEIGHT  1080
 #define SHADOW_WINDOW_WIDTH   1920

//#define WINDOW_HEIGHT 1024
//#define WINDOW_WIDTH  1024
//#define SHADOW_WINDOW_HEIGHT  1024
//#define SHADOW_WINDOW_WIDTH   1024

class Renderer
{
public:
	enum MOUSE_ACTION
	{
		MOUSE_LEFT,
		MOUSE_MIDDLE,
		MOUSE_RIGHT,
		MOUSE_NONE
	};

	static CamViewer s_camViewer;
	static GLFWwindow* s_windowPtr;

	static void s_Init();

	// --------------------------------------------------
	Renderer(const std::string &_shaderFolder);
	~Renderer();
	Renderer(const Renderer& _) = delete;
	Renderer& operator=(const Renderer& _) = delete;

	void InitShader();

	Shader colorShader; 
	Shader textureShader; 
	Shader depthShader; 
	Shader normalShader;

	//GLuint shadowFBO;
	//GLuint shadowTexture;
	
	std::vector<RenderObjectColor*> colorObjs; 
	std::vector<RenderObjectMesh*> meshObjs; 
	std::vector<RenderObjectTexture*> texObjs; 
	std::vector<BallStickObject*> skels; 

	void Draw(std::string type="color"); 
	cv::Mat GetImage(); 
	cv::Mat GetFloatImage();

	void SetBackgroundColor(const Eigen::Vector4f& _color) {
		m_backgroundColor = _color;
	}
	
private:
	static MOUSE_ACTION s_mouseAction;
	static Eigen::Vector2f s_beforePos; 
	static float s_arcballRadius; 
	static double s_leftClickTimeSeconds; 
	
	static void s_InitGLFW();
	static void s_InitGLAD();
	static void s_InitMouse();


	static void s_MouseButtonCallBack(GLFWwindow* _windowPtr, int button, int cation, int mods); 
	static void s_CursorPoseCallBack(GLFWwindow* _windowPtr, double x, double y); 
	static void s_ScrollCallBack(GLFWwindow* _windowPtr, double xOffset, double yOffset);
	static void s_KeyCallBack(GLFWwindow *_windowPtr, int key, int scancode, int action, int mods);

	std::string m_shaderFolder;

	Eigen::Vector4f m_backgroundColor; 
};





