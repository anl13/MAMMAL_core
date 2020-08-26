#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <Eigen/Eigen>
#include <map> 
#include <opencv2/opencv.hpp>

#include "camviewer.h"
#include "render_object.h"
#include "shader.h"
#include "../utils/math_utils.h"

#include "../utils/safe_call.hpp"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/point_types.h>


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

	SimpleShader colorShader; 
	SimpleShader textureShader; 
	SimpleShader depthShader; 
	SimpleShader normalShader;
	SimpleShader meshShader; 
	SimpleShader positionShader; 

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

	// offscreen rendering 
	std::vector<cudaGraphicsResource_t> m_cuda_gl_resources; 
	 
	void initResource(); 

	bool is_useResource; 
	GLuint m_renderbuffers[2]; 
	GLuint m_framebuffer; 
	cudaArray_t m_colorArray;
	void beginOffscreenRender(); 
	void endOffscreenRender(); 
	void mapRenderingResults();
	void unmapRenderingResults(); 

	void releaseResource(); 
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

