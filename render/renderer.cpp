#include <iostream>
#include <string>

#include "renderer.h"
#include "GLFW/glfw3.h" 
#include <cfloat> 
#include "eigen_util.h" 

CamViewer               Renderer::s_camViewer;
GLFWwindow*             Renderer::s_windowPtr;
Renderer::MOUSE_ACTION  Renderer::s_mouseAction;
Eigen::Vector2f         Renderer::s_beforePos;


void Renderer::s_Init()
{
	s_InitGLFW();
	s_InitGLAD();
	s_InitMouse(); 
}


void Renderer::s_InitGLFW()
{
	// glfw: initialize and configure
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	// glfw window creation
	s_windowPtr = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Renderer", NULL, NULL);
	if (s_windowPtr == NULL)
	{
		std::cout << "Failed to create GLFW window." << std::endl; 
		glfwTerminate();
		exit(-1); 
	}

	glfwMakeContextCurrent(s_windowPtr);

	// bind callback
	glfwSetMouseButtonCallback(s_windowPtr, s_MouseButtonCallBack);
	glfwSetCursorPosCallback(s_windowPtr, s_CursorPoseCallBack);
	glfwSetScrollCallback(s_windowPtr, s_ScrollCallBack);
	glfwSetKeyCallback(s_windowPtr, s_KeyCallBack);

}


void Renderer::s_InitGLAD()
{
	// glad: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		// throw "Failed to initialize GLAD";
		std::cout << "failed to initialize GLAD" << std::endl; 
		exit(-1); 
		glfwTerminate();
	}

	// configure global opengl state
	glEnable(GL_DEPTH_TEST);
	
	glEnable(GL_CULL_FACE);

}


void Renderer::s_InitMouse()
{
	s_mouseAction = MOUSE_NONE;
	s_beforePos = Eigen::Vector2f::Zero();
}

void Renderer::s_MouseButtonCallBack(GLFWwindow* _windowPtr, int button, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		switch (button)
		{
		case GLFW_MOUSE_BUTTON_LEFT:
			s_mouseAction = MOUSE_LEFT;
			break;
		case GLFW_MOUSE_BUTTON_MIDDLE:
			s_mouseAction = MOUSE_MIDDLE;
			break;
		case GLFW_MOUSE_BUTTON_RIGHT:
			s_mouseAction = MOUSE_RIGHT;
			break;
		default:
			break;
		}
	}
	else
	{
		s_mouseAction = MOUSE_NONE;
	}
	
}

// move scene with arcball 
void Renderer::s_CursorPoseCallBack(GLFWwindow* _windowPtr, double xPos, double yPos)
{
	Eigen::Vector2f nowPos = Eigen::Vector2f(float(xPos), float(yPos));

	if (nowPos.x() < 0.0f || nowPos.x() > float(WINDOW_WIDTH) || nowPos.y() < 0.0f || nowPos.y() > float(WINDOW_HEIGHT))
	{
		return;
	}

	auto GetArcballCoord = [](
		const Eigen::Vector2f& planeCoord,
		const Eigen::Vector3f& front,
		const Eigen::Vector3f& up,
		const Eigen::Vector3f& right
		)->Eigen::Vector3f {
		// Attention: planeCoord should between [-1, 1]
		float x = planeCoord.x() / sqrtf(2.0f);
		float y = planeCoord.y() / sqrtf(2.0f);
		float z = sqrtf(1 - powf(x, 2) - powf(y, 2));
		return (right * x + up * y - front * z);
	};

	const Eigen::Vector3f camCenter = s_camViewer.GetCenter();
	const Eigen::Vector3f camPos = s_camViewer.GetPos();
	const Eigen::Vector3f camUp = s_camViewer.GetUp();
	const Eigen::Vector3f camRight = s_camViewer.GetRight();
	const Eigen::Vector3f camFront = s_camViewer.GetFront();

	Eigen::Vector2f _nowPos = nowPos; _nowPos(0) /= float(WINDOW_WIDTH); _nowPos(1) /= float(WINDOW_HEIGHT); 
	const Eigen::Vector3f nowArcCoord = GetArcballCoord(
		_nowPos * 2 - Eigen::Vector2f::Ones(), camFront, camUp, camRight);
	Eigen::Vector2f _before_pos = s_beforePos; _before_pos(0) /= float(WINDOW_WIDTH); _before_pos(1) /= float(WINDOW_HEIGHT); 
	const Eigen::Vector3f beforeArcCoord = GetArcballCoord(
		_before_pos * 2 - Eigen::Vector2f::Ones(), camFront, camUp, camRight);

	switch (s_mouseAction)
	{
	case Renderer::MOUSE_LEFT:
	{
		float sensitivity = 2.0f;
		const float theta = acos(beforeArcCoord.dot(nowArcCoord));
		const Eigen::Vector3f rotationAxis = theta < FLT_EPSILON ? Eigen::Vector3f(0.0f, 0.0f, 1.0f) : (beforeArcCoord.cross(nowArcCoord)).normalized();

		const Eigen::Vector3f nowCamPos = Eigen::AngleAxisf(sensitivity * theta, rotationAxis) * (camPos - camCenter) + camCenter;
		s_camViewer.SetExtrinsic(nowCamPos, camUp, camCenter);
		break;
	}

	case Renderer::MOUSE_MIDDLE:
		break;

	case Renderer::MOUSE_RIGHT:
	{
		const float distance = (camPos - camCenter).norm();
		Eigen::Vector3f nowCamcenter = camCenter + distance * (nowArcCoord - beforeArcCoord);
		s_camViewer.SetExtrinsic(camPos, camUp, nowCamcenter);
		break;
	}

	case Renderer::MOUSE_NONE:
		break;

	default:
		break;
	}
	s_beforePos = nowPos;
}

void Renderer::s_ScrollCallBack(GLFWwindow* _windowPtr, double xOffset, double yOffset)
{
	float sensitivity = 0.2f;
	const Eigen::Vector3f pos = Renderer::s_camViewer.GetPos();
	const Eigen::Vector3f front = Renderer::s_camViewer.GetFront();
	const Eigen::Vector3f up = Renderer::s_camViewer.GetUp();
	const Eigen::Vector3f center = Renderer::s_camViewer.GetCenter();
	
	const Eigen::Vector3f newPos = pos + sensitivity * float(yOffset) * front;
	if ((newPos - center).dot(pos - center) > 0.0f)
	{
		s_camViewer.SetExtrinsic(newPos, up, center);
	}
}


void Renderer::s_KeyCallBack(GLFWwindow *_windowPtr, int key, int scancode, int action, int mods)
{
	float sensitivity = 0.1f;
	const Eigen::Vector3f pos = Renderer::s_camViewer.GetPos();
	const Eigen::Vector3f front = Renderer::s_camViewer.GetFront();
	const Eigen::Vector3f up = Renderer::s_camViewer.GetUp();
	const Eigen::Vector3f center = Renderer::s_camViewer.GetCenter();

    if( key == GLFW_KEY_ESCAPE && action == GLFW_PRESS ) 
	{
		glfwSetWindowShouldClose(_windowPtr, GL_TRUE); 
		return ; 
	} 
	
	Eigen::Vector3f newPos = pos; 
	switch (key)
	{
	case 87:	// W
		newPos += sensitivity * front;
		break;
	case 83:	// A
		newPos -= sensitivity * front;        
		break;
	case 65:	// S
		newPos += sensitivity * (front.cross(up)).normalized();
		break;
	case 68:	// D
		newPos -= sensitivity * (front.cross(up)).normalized();
		break;
	case 32:	// space
		newPos += sensitivity * up;
		break;
	case 341:	// ctrl
		newPos -= sensitivity * up;
		break;
	// case 67: // c
	// 	// if(action == GLFW_PRESS)
	// 		// s_outContinue++; 
	// 	break; 

	default:
		std::cout << "key: " << key << std::endl; 
		break;
	}

	if((newPos- center).dot(pos-center) > 0.0f)
	{
		s_camViewer.SetExtrinsic(newPos, up, center); 
	}
}


// -------------------------------------------------------
Renderer::Renderer(const std::string &shaderFolder):m_shaderFolder(shaderFolder)
{
	InitShader();
}


Renderer::~Renderer()
{
	glDeleteTextures(1, &shadowTexture);
	glDeleteFramebuffers(1, &shadowFBO);

	glfwTerminate();
}


void Renderer::InitShader()
{
	colorShader = Shader(m_shaderFolder + "/basic_color_v.shader", m_shaderFolder + "/basic_color_f.shader", m_shaderFolder + "/basic_color_g.shader");
	textureShader = Shader(m_shaderFolder + "/basic_texture_v.shader", m_shaderFolder + "/basic_texture_f.shader", m_shaderFolder + "/basic_texture_g.shader");
	meshShader = Shader(m_shaderFolder + "/mesh_v.shader", m_shaderFolder + "/mesh_f.shader", m_shaderFolder + "/mesh_g.shader");
	depthShader = Shader(m_shaderFolder + "/depth_v.shader", m_shaderFolder + "/depth_f.shader", m_shaderFolder + "/depth_g.shader");

	glGenFramebuffers(1, &shadowFBO);	
	glGenTextures(1, &shadowTexture);
	glBindTexture(GL_TEXTURE_CUBE_MAP, shadowTexture);

	for (int i = 0; i < 6; i++)
	{
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, SHADOW_WINDOW_WIDTH, SHADOW_WINDOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	// attach depth texture as FBO's depth buffer
	glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadowTexture, 0);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	// glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// basicShader.Use();
	// basicShader.SetInt("object_texture", 0);
	// basicShader.SetInt("depth_cube", 1);
}

void Renderer::Draw()
{
	// set background
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// 1. render depth of scene to texture (from light's perspective)
	// --------------------------------------------------------------
	glViewport(0, 0, SHADOW_WINDOW_WIDTH, SHADOW_WINDOW_HEIGHT);
	glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
	glClear(GL_DEPTH_BUFFER_BIT);
	depthShader.Use();

	Eigen::Vector3f lightPos = -s_camViewer.GetPos();
	
	depthShader.SetVec3("light_pos", lightPos);
	depthShader.SetFloat("far_plane", RENDER_FAR_PLANE);
	{
		Eigen::Matrix<float, 4, 4, Eigen::ColMajor> perspective = EigenUtil::Perspective(0.5f*EIGEN_PI, 1.0f, RENDER_NEAR_PLANE, RENDER_FAR_PLANE);

		std::vector<Eigen::Vector3f> frontList = { 
			{1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
			{0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f} };

		std::vector<Eigen::Vector3f> upList = {
			{0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f },{0.0f, 0.0f, 1.0f},
			{0.0f, 0.0f, -1.0f },{ 0.0f, -1.0f, 0.0f },{ 0.0f, -1.0f, 0.0f} };

		for (int i = 0; i < 6; i++)
		{
			const Eigen::Matrix<float, 4, 4, Eigen::ColMajor> shadowTransform = perspective * EigenUtil::LookAt(lightPos, lightPos + frontList[i], upList[i]);
			depthShader.SetMat4("shadow_matrices[" + std::to_string(i) + "]", shadowTransform);
		}
	}

	for (auto iter = renderObjects.begin(); iter != renderObjects.end(); iter++)
	{
		iter->second->DrawDepth(depthShader);
	}
	for(int i = 0; i < skels.size(); i++)
	{
		skels[i]->Draw(depthShader); 
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	// 2. render scene as normal using the generated depth/shadow map  
	// --------------------------------------------------------------
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	for (auto iter = renderObjects.begin(); iter != renderObjects.end(); iter++)
	{
		const RenderObject* const objectPtr = iter->second;
		if (objectPtr->GetType() == RENDER_OBJECT_COLOR)
		{
			colorShader.Use();
			s_camViewer.ConfigShader(colorShader);
			colorShader.SetVec3("light_pos", lightPos);
			colorShader.SetFloat("far_plane", RENDER_FAR_PLANE);

			colorShader.SetInt("depth_cube", 1);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_CUBE_MAP, shadowTexture);

			objectPtr->DrawWhole(colorShader);
		}
		else if (objectPtr->GetType() == RENDER_OBJECT_TEXTURE)
		{
			textureShader.Use();
			s_camViewer.ConfigShader(textureShader);
			textureShader.SetVec3("light_pos", lightPos);
			textureShader.SetFloat("far_plane", RENDER_FAR_PLANE);

			textureShader.SetInt("depth_cube", 1);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_CUBE_MAP, shadowTexture);

			textureShader.SetInt("object_texture", 2);
			glActiveTexture(GL_TEXTURE2);

			objectPtr->DrawWhole(textureShader);
		}
		else if (objectPtr->GetType() == RENDER_OBJECT_MESH)
		{
			meshShader.Use();
			s_camViewer.ConfigShader(meshShader);
			objectPtr->DrawWhole(meshShader);
		}
		else
		{
			// throw std::string("unknow render object type");
			std::cout << "[Renderer] unknown render object type " << std::endl; 
			exit(-1); 
		}
	}

	for(int i = 0; i < skels.size(); i++)
	{
		colorShader.Use(); 
		s_camViewer.ConfigShader(colorShader); 
		colorShader.SetVec3("light_pos", lightPos); 
		colorShader.SetFloat("far_plane", RENDER_FAR_PLANE); 
		colorShader.SetInt("depth_cube", 1); 
		glActiveTexture(GL_TEXTURE1); 
		glBindTexture(GL_TEXTURE_CUBE_MAP, shadowTexture); 
		
		skels[i]->Draw(colorShader); 
	}
}