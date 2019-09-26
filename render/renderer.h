#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <Eigen/Eigen>
#include <map> 

#include "camviewer.h"
#include "render_object.h"
#include "shader.h"

// #define WINDOW_SIZE 800
// #define SHADOW_WINDOW_SIZE 1024


#define WINDOW_SIZE 512
#define SHADOW_WINDOW_SIZE 512

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
	Shader meshShader; 
	Shader depthShader; 

	GLuint shadowFBO;
	GLuint shadowTexture;

	std::map<int, RenderObject*> renderObjects; 
	template<class T> 
	void InsertObject(const int id, T* renderObject){
		renderObjects.insert(std::make_pair(id, renderObject)); 
	}

	void Draw(); 
	
private:
	static MOUSE_ACTION s_mouseAction;
	static Eigen::Vector2f s_beforePos; 

	static void s_InitGLFW();
	static void s_InitGLAD();
	static void s_InitMouse();


	static void s_MouseButtonCallBack(GLFWwindow* _windowPtr, int button, int cation, int mods); 
	static void s_CursorPoseCallBack(GLFWwindow* _windowPtr, double x, double y); 
	static void s_ScrollCallBack(GLFWwindow* _windowPtr, double xOffset, double yOffset);
	static void s_KeyCallBack(GLFWwindow *_windowPtr, int key, int scancode, int action, int mods);

	std::string m_shaderFolder;
};





