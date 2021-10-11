#include <iostream>
#include <string>

#include "renderer.h"
#include "GLFW/glfw3.h" 
#include <cfloat> 
#include <opencv2/opencv.hpp>

#include "cuda_utils_render.h"
#include "render_utils.h"

CamViewer               Renderer::s_camViewer;
GLFWwindow*             Renderer::s_windowPtr;
Renderer::MOUSE_ACTION  Renderer::s_mouseAction;
Eigen::Vector2f         Renderer::s_beforePos;
float                   Renderer::s_arcballRadius; 
double                  Renderer::s_leftClickTimeSeconds; 

#define SHOW_CAM_POSE

void Renderer::s_Init(bool isHideWindow)
{
	s_InitGLFW(isHideWindow);
	s_InitGLAD();
	s_InitMouse(); 
}


void Renderer::s_InitGLFW(bool isHideWindow)
{
	// glfw: initialize and configure
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// NOTE: 
	// if you want to render faceindex, please comment following 2 lines
	// to close anti-alise and smoothing. 
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 4);

	if(isHideWindow)
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	// glfw window creation
	s_windowPtr = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Renderer", NULL, NULL);
	if (s_windowPtr == NULL)
	{
		std::cout << "Failed to create GLFW window." << std::endl; 
		glfwTerminate();
		exit(-1); 
	}

	glfwMakeContextCurrent(s_windowPtr);
}

void Renderer::s_InitGLAD()
{
	// glad: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
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
	// bind callback
	glfwSetMouseButtonCallback(s_windowPtr, s_MouseButtonCallBack);
	glfwSetCursorPosCallback(s_windowPtr, s_CursorPoseCallBack);
	glfwSetScrollCallback(s_windowPtr, s_ScrollCallBack);
	glfwSetKeyCallback(s_windowPtr, s_KeyCallBack);
	s_mouseAction = MOUSE_NONE;
	s_beforePos = Eigen::Vector2f::Zero();
	s_arcballRadius = 1.0f;
	s_leftClickTimeSeconds = 0.0;
}

void Renderer::s_MouseButtonCallBack(GLFWwindow* _windowPtr, int button, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		switch (button)
		{
		case GLFW_MOUSE_BUTTON_LEFT:
		{
			s_mouseAction = MOUSE_LEFT;
			double seconds = glfwGetTime();   
			if(seconds - s_leftClickTimeSeconds < 0.2)
			{
				const Eigen::Vector3f camCenter = s_camViewer.GetCenter();
				const Eigen::Vector3f camPos = s_camViewer.GetPos();
				const Eigen::Vector3f camUp = s_camViewer.GetUp();
				Eigen::Vector3f newCamPos = camPos - camCenter; 
				Eigen::Vector3f newCenter = Eigen::Vector3f::Zero(); 
				s_camViewer.SetExtrinsic(newCamPos, camUp, newCenter);
#ifdef SHOW_CAM_POSE
				std::cout << "newCamPos button:" << newCamPos.transpose() << std::endl;
				std::cout << "camUp button    : " << camUp.transpose() << std::endl;
				std::cout << "newCenter button: " << newCenter.transpose() << std::endl; 
#endif 
			}
			s_leftClickTimeSeconds = seconds; 

			break;
		}
		case GLFW_MOUSE_BUTTON_MIDDLE:
		{
			s_mouseAction = MOUSE_MIDDLE;
			break;
		}
		case GLFW_MOUSE_BUTTON_RIGHT:
		{
			s_mouseAction = MOUSE_RIGHT;
			break;
		}
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
		float x = planeCoord.x() / s_arcballRadius;
		float y = planeCoord.y() / s_arcballRadius;
		float z = 0; 
		float r = x * x + y * y; 
		if(r > 1)
		{
			x = x / r; 
			y = y / r; 
			z = 0; 
		}
		else 
		{
			z = sqrtf(1 - powf(x, 2) - powf(y, 2));
		}
		
		return (right * x + up * y + front * z);
	};

	const Eigen::Vector3f camCenter = s_camViewer.GetCenter();
	const Eigen::Vector3f camPos = s_camViewer.GetPos();
	const Eigen::Vector3f camUp = s_camViewer.GetUp();
	const Eigen::Vector3f camRight = s_camViewer.GetRight();
	const Eigen::Vector3f camFront = s_camViewer.GetFront();

	Eigen::Vector2f _nowPos = nowPos; _nowPos(0) /= float(WINDOW_WIDTH); _nowPos(1) /= float(WINDOW_HEIGHT); // normalize to 0~1
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

		// std::cout << camCenter.transpose() << std::endl; 
		// std::cout << camPos.transpose() << std::endl; 
		const Eigen::Vector3f nowCamPos = Eigen::AngleAxisf(sensitivity * theta, rotationAxis) * (camPos - camCenter) + camCenter;
		s_camViewer.SetExtrinsic(nowCamPos, camUp, camCenter);

		
#ifdef SHOW_CAM_POSE
		std::cout << "nowCamPos:" << nowCamPos.transpose() << std::endl; 
		std::cout << "nowcamUp : " << camUp.transpose() << std::endl; 
		std::cout << "camCen   :" << camCenter.transpose() << std::endl; 
#endif 
		break;
	}

	case Renderer::MOUSE_MIDDLE:
		break;

	case Renderer::MOUSE_RIGHT:
	{
		const float distance = (camPos - camCenter).norm();
		Eigen::Vector3f nowCamcenter = camCenter + distance * (nowArcCoord - beforeArcCoord);
		s_camViewer.SetExtrinsic(camPos, camUp, nowCamcenter);
#ifdef SHOW_CAM_POSE
		std::cout << "camPOs:   " << camPos.transpose() << std::endl; 
		std::cout << "nowcamUp: " << camUp.transpose() << std::endl; 
		std::cout << "camCen   :" << nowCamcenter.transpose() << std::endl; 
#endif 
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
	const Eigen::Vector3f pos    = Renderer::s_camViewer.GetPos();
	const Eigen::Vector3f front  = Renderer::s_camViewer.GetFront();
	const Eigen::Vector3f up     = Renderer::s_camViewer.GetUp();
	const Eigen::Vector3f center = Renderer::s_camViewer.GetCenter();
	
	const Eigen::Vector3f newPos = pos - sensitivity * float(yOffset) * front;
	if ((newPos - center).dot(pos - center) > 0.0f)
	{
		s_camViewer.SetExtrinsic(newPos, up, center);
#ifdef SHOW_CAM_POSE
		std::cout << "camPOs:   " << newPos.transpose() << std::endl;
		std::cout << "nowcamUp: " << up.transpose() << std::endl;
		std::cout << "camCen   :" << center.transpose() << std::endl;
#endif 
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
	case 27:
		exit(-1); 
		break; 
	// case 67: // c
	// 	// if(action == GLFW_PRESS)
	// 		// s_outContinue++; 
	// 	break; 
	default:
		//std::cout << "key: " << key << std::endl; 
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

	is_useResource = false; 
	initResource(); 
}


Renderer::~Renderer()
{
	releaseResource(); 
	glfwTerminate();
}


void Renderer::InitShader()
{
	colorShader = SimpleShader(m_shaderFolder + "/color_v.shader", 
		m_shaderFolder + "/color_f.shader");
	textureShader = SimpleShader(m_shaderFolder + "/texture_v.shader", 
		m_shaderFolder + "/texture_f.shader");
	meshShader = SimpleShader(m_shaderFolder + "/mesh_v.shader",
		m_shaderFolder + "/mesh_f.shader"); 
	positionShader = SimpleShader(m_shaderFolder + "/position_v.shader",
		m_shaderFolder + "/position_f.shader");
	maskShader = SimpleShader(m_shaderFolder + "/mask_v.shader", 
		m_shaderFolder + "/mask_f.shader");

	textureShaderML = SimpleShader(m_shaderFolder + "/texture_v.shader",
		m_shaderFolder + "/texture_f_multilight.shader");
	colorShaderML = SimpleShader(m_shaderFolder + "/color_v.shader",
		m_shaderFolder + "/color_f_multilight.shader"); 

	xrayShader = SimpleShader(m_shaderFolder + "/xray_v.shader",
		m_shaderFolder + "/xray_f.shader"); 

	faceindexShader = SimpleShader(m_shaderFolder + "/faceindex_v.shader",
		m_shaderFolder + "/faceindex_f.shader"); 

	std::cout << "init depth shader" << std::endl; 

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::Draw(std::string type)
{
	// set background
	glClearColor(m_backgroundColor(0),
		m_backgroundColor(1),
		m_backgroundColor(2),
		m_backgroundColor(3));
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	Eigen::Vector3f lightPos = s_camViewer.GetPos(); 
	//Eigen::Vector3f lightPos = Eigen::Vector3f(0, 0, 3);

	// 2. render scene as normal using the generated depth/shadow map  
	// --------------------------------------------------------------
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glEnable(GL_BLEND);
	//glDisable(GL_DEPTH_TEST);
	//2.指定混合因子
	//注意:如果你修改了混合方程式,当你使用混合抗锯齿功能时,请一定要改为默认混合方程式
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//3.开启对点\线\多边形的抗锯齿功能
	//glEnable(GL_POINT_SMOOTH);
	//glEnable(GL_LINE_SMOOTH);
	//glEnable(GL_POLYGON_SMOOTH);
	//glEnable(GL_MULTISAMPLE);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	for(int i = 0; i < colorObjs.size(); i++)
	{
		if (type == "color")
		{
			if (colorObjs[i]->isMultiLight)
			{
				colorShaderML.Use();
				s_camViewer.ConfigShader(colorShaderML);
				colorShaderML.SetVec3("spotLight.position", s_camViewer.GetPos());
				colorShaderML.SetVec3("spotLight.direction", s_camViewer.GetFront());
				colorShaderML.configMultiLight();
				if (colorObjs[i]->isFill)
				{
					glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				}
				else 
					glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				colorObjs[i]->DrawWhole(colorShaderML);
			}
			else
			{
				colorShader.Use();
				s_camViewer.ConfigShader(colorShader);
				colorShader.SetVec3("light_pos", lightPos);
				colorShader.SetFloat("far_plane", RENDER_FAR_PLANE);
				if (colorObjs[i]->isFill)
				{
					glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				}
				else
					glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				colorObjs[i]->DrawWhole(colorShader);

				//xrayShader.Use(); 
				//s_camViewer.ConfigShader(xrayShader); 
				//xrayShader.SetVec3("light_pos", lightPos); 
				//xrayShader.SetFloat("far_plane", RENDER_FAR_PLANE);
				//colorObjs[i]->DrawWhole(xrayShader); 
			}
		}
		else if (type == "mask")
		{
			maskShader.Use(); 
			s_camViewer.ConfigShader(maskShader); 
			maskShader.SetVec3("light_pos", lightPos); 
			maskShader.SetFloat("far_plane", RENDER_FAR_PLANE); 
			if (colorObjs[i]->isFill)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
			else
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			colorObjs[i]->DrawWhole(maskShader); 
		}
		else if (type == "depth")
		{
			positionShader.Use();
			s_camViewer.ConfigShader(positionShader);
			positionShader.SetVec3("light_pos", lightPos);
			positionShader.SetFloat("far_plane", RENDER_FAR_PLANE);
			if (colorObjs[i]->isFill)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
			else
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			colorObjs[i]->DrawWhole(positionShader);
		}
	}
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	for(int i = 0; i < texObjs.size(); i++)
	{
		if (texObjs[i]->isFaceIndex)
		{
			faceindexShader.Use(); 
			s_camViewer.ConfigShader(faceindexShader); 
			faceindexShader.SetInt("object_texture", 0);
			glActiveTexture(GL_TEXTURE0);
			if (texObjs[i]->isFill)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
			else
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			texObjs[i]->DrawWhole(faceindexShader); 
		}
		else if (texObjs[i]->isMultiLight)
		{
			textureShaderML.Use();
			s_camViewer.ConfigShader(textureShaderML);
			textureShaderML.SetVec3("spotLight.position", s_camViewer.GetPos());
			textureShaderML.SetVec3("spotLight.direction", s_camViewer.GetFront());
			textureShaderML.configMultiLight();
			textureShaderML.SetInt("object_texture", 0);

			glActiveTexture(GL_TEXTURE0);
			if (texObjs[i]->isFill)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
			else
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			texObjs[i]->DrawWhole(textureShaderML);
		}
		else
		{
			textureShader.Use();
			s_camViewer.ConfigShader(textureShader);
			textureShader.SetVec3("light_pos", lightPos);
			textureShader.SetFloat("far_plane", RENDER_FAR_PLANE);
			textureShader.SetInt("object_texture", 0);
			glActiveTexture(GL_TEXTURE0);
			if (texObjs[i]->isFill)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
			else
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			texObjs[i]->DrawWhole(textureShader);
		}
	}

	for(int i = 0; i < meshObjs.size(); i++)
	{
		if (type == "color")
		{
			meshShader.Use();
			s_camViewer.ConfigShader(meshShader);
			meshShader.SetVec3("light_pos", lightPos);
			meshShader.SetFloat("far_plane", RENDER_FAR_PLANE);
			if (meshObjs[i]->isFill)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
			else
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			meshObjs[i]->DrawWhole(meshShader);
		}
	}

	for(int i = 0; i < skels.size(); i++)
	{
		if (type == "color")
		{
			if (skels[i]->isMultiLight)
			{
				colorShaderML.Use();
				s_camViewer.ConfigShader(colorShaderML);
				colorShaderML.SetVec3("spotLight.position", s_camViewer.GetPos());
				colorShaderML.SetVec3("spotLight.direction", s_camViewer.GetFront());
				colorShaderML.configMultiLight();
				if (skels[i]->isFill)
				{
					glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				}
				else
					glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				skels[i]->Draw(colorShaderML);
			}
			else
			{
				colorShader.Use();
				s_camViewer.ConfigShader(colorShader);
				colorShader.SetVec3("light_pos", lightPos);
				colorShader.SetFloat("far_plane", RENDER_FAR_PLANE);
				if (skels[i]->isFill)
				{
					glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				}
				else
					glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				skels[i]->Draw(colorShader);
			}
		}
	}
}

cv::Mat Renderer::GetImage()
{
	cv::Mat image(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_8UC3);
	glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, image.data);
	cv::flip(image, image, 0);
	return image;
}

cv::Mat Renderer::GetFloatImage()
{
	cv::Mat image(cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT), CV_32FC1);
	glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RED, GL_FLOAT, image.data);
	cv::flip(image, image, 0);
	return image; 
}

// gpu resource 
void Renderer::initResource()
{
	is_useResource = true;
	
	glGenRenderbuffers(2, m_renderbuffers); 

	glBindRenderbuffer(GL_RENDERBUFFER, m_renderbuffers[0]); 
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, WINDOW_WIDTH, WINDOW_HEIGHT); 
	glBindRenderbuffer(GL_RENDERBUFFER, m_renderbuffers[1]); 
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, WINDOW_WIDTH, WINDOW_HEIGHT);
	glBindRenderbuffer(GL_RENDERBUFFER, 0); 

	glGenFramebuffers(1, &m_framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer); 
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
		GL_RENDERBUFFER, m_renderbuffers[0]); 
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
		GL_RENDERBUFFER, m_renderbuffers[1]);
	glBindFramebuffer(GL_FRAMEBUFFER, 0); 
	
	m_cuda_gl_resources.resize(1); 

	cudaGraphicsGLRegisterImage(&m_cuda_gl_resources[0], m_renderbuffers[0],
		GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly);

	cudaMalloc((void**)&m_device_depth, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float));
	cudaMalloc((void**)&m_device_renderData, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float4));
}

void Renderer::releaseResource()
{
	if (m_device_renderData != nullptr) cudaFree(m_device_renderData); 
	if (m_device_depth != nullptr) cudaFree(m_device_depth); 
	glDeleteRenderbuffers(2, m_renderbuffers);
	glDeleteFramebuffers(1, &m_framebuffer); 
	is_useResource = false; 
}

void Renderer::beginOffscreenRender()
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer); 
}

void Renderer::endOffscreenRender()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0); 
}

void Renderer::mapRenderingResults()
{
	cudaGraphicsMapResources(m_cuda_gl_resources.size(), &m_cuda_gl_resources[0]); 
	cudaGraphicsSubResourceGetMappedArray(&m_colorArray, m_cuda_gl_resources[0], 0, 0);
}

void Renderer::unmapRenderingResults()
{
	cudaGraphicsUnmapResources(m_cuda_gl_resources.size(), &m_cuda_gl_resources[0]); 
}

float * Renderer::renderDepthDevice()
{
	beginOffscreenRender(); 
	Draw("depth"); 
	mapRenderingResults(); 
	cudaMemcpy2DFromArray(m_device_renderData, WINDOW_WIDTH * sizeof(float4),
		m_colorArray, 0, 0, WINDOW_WIDTH * sizeof(float4), WINDOW_HEIGHT, cudaMemcpyDeviceToDevice);
	extract_depth_channel(m_device_renderData, WINDOW_WIDTH, WINDOW_HEIGHT, m_device_depth); 
	unmapRenderingResults(); 
	endOffscreenRender(); 

	return m_device_depth;
}

cv::Mat Renderer::GetImageOffscreen()
{
	beginOffscreenRender();
	Draw();
	mapRenderingResults();
	cudaMemcpy2DFromArray(m_device_renderData, WINDOW_WIDTH * sizeof(float4),
		m_colorArray, 0, 0, WINDOW_WIDTH * sizeof(float4), WINDOW_HEIGHT, cudaMemcpyDeviceToDevice);
	cv::Mat img = extract_bgr_mat(m_device_renderData, WINDOW_WIDTH, WINDOW_HEIGHT); 
	unmapRenderingResults();
	endOffscreenRender();

	return img;
}

void Renderer::clearAllObjs()
{
	for (int i = 0; i < meshObjs.size(); i++) delete meshObjs[i];
	meshObjs.clear(); 
	for (int i = 0; i < colorObjs.size(); i++) delete colorObjs[i];
	colorObjs.clear(); 
	for (int i = 0; i < texObjs.size(); i++) delete texObjs[i];
	texObjs.clear();
	for (int i = 0; i < skels.size(); i++)skels[i]->deleteObjects();
	skels.clear(); 
}

void Renderer::createScene(std::string conf_projectFolder)
{
	//Mesh obj;
	//obj.Load(conf_projectFolder + "/data/calibdata/scene_model/manual_scene_part0.obj");
	////obj = ballMesh; 
	//RenderObjectTexture* p_scene = new RenderObjectTexture();
	//p_scene->SetTexture(conf_projectFolder + "/render/data/chessboard_black_large.png");
	//p_scene->SetFaces(obj.faces_v_vec);
	//p_scene->SetVertices(obj.vertices_vec);
	//p_scene->SetNormal(obj.normals_vec, 2);
	//p_scene->SetTexcoords(obj.textures_vec, 1);
	//p_scene->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	//p_scene->isMultiLight = false; 
	//texObjs.push_back(p_scene);
	
	createPlane(conf_projectFolder); 

	//Mesh obj2;
	//obj2.Load(conf_projectFolder + "/data/calibdata/scene_model/manual_scene_part1.obj");
	//RenderObjectTexture* p_scene2 = new RenderObjectTexture();
	//p_scene2->SetTexcoords(obj2.textures_vec, 1);
	//p_scene2->SetNormal(obj2.normals_vec, 2);
	//p_scene2->SetVertices(obj2.vertices_vec);
	//p_scene2->SetFaces(obj2.faces_v_vec);
	//p_scene2->SetTexture(conf_projectFolder + "/render/data/chessboard_bk.png");
	//p_scene2->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	//texObjs.push_back(p_scene2);

	Mesh obj2;
	obj2.Load(conf_projectFolder + "/data/calibdata/scene_model/manual_scene_part1.obj");
	RenderObjectColor * p_scene2 = new RenderObjectColor();
	p_scene2->SetVertices(obj2.vertices_vec);
	p_scene2->SetNormal(obj2.normals_vec);
	p_scene2->SetFaces(obj2.faces_v_vec);
	p_scene2->SetColor(Eigen::Vector3f(0.9, 0.9, 0.85));
	p_scene2->isMultiLight = true; 
	colorObjs.push_back(p_scene2);

	Mesh obj3;
	obj3.Load(conf_projectFolder + "/data/calibdata/scene_model/manual_scene_part2.obj");
	RenderObjectColor * p_scene3 = new RenderObjectColor();
	p_scene3->SetVertices(obj3.vertices_vec);
	p_scene3->SetNormal(obj3.normals_vec);
	p_scene3->SetFaces(obj3.faces_v_vec);
	p_scene3->SetColor(Eigen::Vector3f(0.753, 0.753, 0.753));
	p_scene3->isMultiLight = true; 
	colorObjs.push_back(p_scene3);
}

void Renderer::createPlane(std::string conf_projectFolder, float scale)
{
	//Mesh obj;
	//obj.Load(conf_projectFolder + "/data/calibdata/scene_model/manual_scene_part0.obj");
	////obj = ballMesh; 
	//RenderObjectTexture* p_scene = new RenderObjectTexture();
	//p_scene->SetTexture(conf_projectFolder + "/render/data/chessboard_black_large.png");
	//p_scene->SetFaces(obj.faces_v_vec);
	//p_scene->SetVertices(obj.vertices_vec);
	//p_scene->SetNormal(obj.normals_vec, 2);
	//p_scene->SetTexcoords(obj.textures_vec, 1);
	//p_scene->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	//p_scene->isMultiLight = false;
	//texObjs.push_back(p_scene);

	std::vector<Eigen::Vector3f> vertices, colors;
	std::vector<Eigen::Vector3u> faces;
	readObjectWithColor(conf_projectFolder + "/render/data/obj_model/floor_z+_gray.obj", vertices, colors, faces);
	Mesh obj;
	obj.faces_v_vec = faces;
	obj.vertices_vec = vertices;
	obj.vertex_num = vertices.size();
	obj.face_num = faces.size();
	if (scale != 1)
	{
		for (int i = 0; i < obj.vertices_vec.size(); i++)
		{
			obj.vertices_vec[i] = obj.vertices_vec[i] * scale; 
		}
	}
	obj.CalcNormal();
	RenderObjectMesh* p_floor = new RenderObjectMesh();
	p_floor->SetVertices(obj.vertices_vec);
	p_floor->SetColors(colors);
	p_floor->SetFaces(obj.faces_v_vec);
	p_floor->SetNormal(obj.normals_vec);
	meshObjs.push_back(p_floor);
}

void Renderer::createSceneDetailed(std::string conf_projectFolder, float scale, int flip_axis)
{
	createPlane(conf_projectFolder, scale); 
	for (int k = 2; k < 7; k++)
	{
		if (k == 4) continue; 
		std::stringstream ss;
		ss << conf_projectFolder << "/render/data/obj_model/zhujuan_new_part" << k << ".obj";
		Mesh obj(ss.str());
		if (flip_axis >= 0)
			obj.flip(flip_axis); 
		RenderObjectColor *p_model = new RenderObjectColor();
		
		if (scale != 1)
		{
			for (int i = 0; i < obj.vertices_vec.size(); i++)
			{
				obj.vertices_vec[i] = obj.vertices_vec[i] * scale;
			}
			obj.CalcNormal(); 
		}
		p_model->SetVertices(obj.vertices_vec);
		p_model->SetNormal(obj.normals_vec);
		p_model->SetColor(Eigen::Vector3f(0.9, 0.9, 0.9));
		p_model->isMultiLight = false;
		p_model->SetFaces(obj.faces_v_vec);
		colorObjs.push_back(p_model);
	}
}

void Renderer::createSceneHalf(std::string conf_projectFolder, float scale)
{
	createPlane(conf_projectFolder, scale);

	std::stringstream ss;
	ss << conf_projectFolder << "/render/data/obj_model/zhujuan_halfwall2.obj";
	Mesh obj(ss.str());
	RenderObjectColor *p_model = new RenderObjectColor();

	if (scale != 1)
	{
		for (int i = 0; i < obj.vertices_vec.size(); i++)
		{
			obj.vertices_vec[i] = obj.vertices_vec[i] * scale;
		}
		obj.CalcNormal();
	}
	p_model->SetVertices(obj.vertices_vec);
	p_model->SetNormal(obj.normals_vec);
	p_model->SetColor(Eigen::Vector3f(0.9, 0.9, 0.9));
	p_model->isMultiLight = false;
	p_model->SetFaces(obj.faces_v_vec);
	colorObjs.push_back(p_model);
}

void Renderer::createSceneHalf2(std::string conf_projectFolder, float scale)
{
	createPlane(conf_projectFolder, scale);
	{
		std::stringstream ss;
		ss << conf_projectFolder << "/render/data/obj_model/zhujuan_halfwall2.obj";
		Mesh obj(ss.str());
		RenderObjectColor *p_model = new RenderObjectColor();

		if (scale != 1)
		{
			for (int i = 0; i < obj.vertices_vec.size(); i++)
			{
				obj.vertices_vec[i] = obj.vertices_vec[i] * scale;
			}
			obj.CalcNormal();
		}
		p_model->SetVertices(obj.vertices_vec);
		p_model->SetNormal(obj.normals_vec);
		p_model->SetColor(Eigen::Vector3f(0.9, 0.9, 0.9));
		p_model->isMultiLight = false;
		p_model->SetFaces(obj.faces_v_vec);
		colorObjs.push_back(p_model);
	}
	{
		std::stringstream ss;
		ss << conf_projectFolder << "/render/data/obj_model/zhujuan_halfwall3.obj";
		Mesh obj(ss.str());
		RenderObjectColor *p_model = new RenderObjectColor();

		if (scale != 1)
		{
			for (int i = 0; i < obj.vertices_vec.size(); i++)
			{
				obj.vertices_vec[i] = obj.vertices_vec[i] * scale;
			}
			obj.CalcNormal();
		}
		p_model->SetVertices(obj.vertices_vec);
		p_model->SetNormal(obj.normals_vec);
		p_model->SetColor(Eigen::Vector3f(0.9, 0.9, 0.9));
		p_model->isMultiLight = false;
		p_model->SetFaces(obj.faces_v_vec);
		colorObjs.push_back(p_model);
	}
}

void Renderer::createHikonCam(std::string projectFolder, const std::vector<Camera>& cams)
{
	std::vector<Eigen::Vector3f> colors = {
		{0.71, 0.71, 0.71}, 
	{0.465, 0.50, 0.564}, 
	{0.07, 0.07, 0.07}, 
	{0.07, 0.07, 0.07}
	}; 
	std::vector<Mesh> meshes; 
	for (int k = 0; k < 4; k++)
	{
		std::stringstream ss; 
		ss << projectFolder << "/render/data/obj_model/camera_big_resize_part" << k + 1 << ".obj"; 
		Mesh model(ss.str()); 
		for (int i = 0; i < model.vertex_num; i++) model.vertices_vec[i] *= 0.01; 
		meshes.push_back(model); 
	}

	for (int camid = 0; camid < cams.size(); camid++)
	{
		Camera cam = cams[camid]; 
		Eigen::Matrix3f R = cam.inv_R; 
		Eigen::Vector3f T = -R * cam.T; 
		Eigen::Vector3f euler = Mat2Euler(R); 
		euler(0) = 0; euler(2) = 0; 
		Eigen::Matrix3f R_2 = EulerToRotRad(euler); 
		for (int part = 0; part < 4; part++)
		{
			Mesh local = meshes[part];
			for (int i = 0; i < local.vertices_vec.size(); i++)
			{
				local.vertices_vec[i] = R_2 * local.vertices_vec[i] + T;
			}
			RenderObjectColor * p_camera = new RenderObjectColor();
			p_camera->SetVertices(local.vertices_vec);
			p_camera->SetFaces(local.faces_v_vec);
			p_camera->SetNormal(local.normals_vec);
			p_camera->SetColor(colors[part]);
			colorObjs.push_back(p_camera);
		}
	}
}

void Renderer::createVirtualCam(std::string projectFolder, const std::vector<Camera>& cams)
{
	Mesh obj(projectFolder + "/render/data/obj_model/cam_watertight.obj"); 

	for (int camid = 0; camid < cams.size(); camid++)
	{
		Camera cam = cams[camid];
		Eigen::Matrix3f R = cam.inv_R;
		Eigen::Vector3f T = -R * cam.T;
		T.segment<2>(0) *= 1.2; 
		if (camid == 0 || camid == 3 || camid == 5 || camid == 6)
			T(2) -= 0.3; 
		else T(2) += 0.12; 
		Mesh local = obj;
		for (int i = 0; i < local.vertices_vec.size(); i++)
		{
			local.vertices_vec[i](2) *= -1; 
			local.vertices_vec[i](0) *= -1; 
			local.vertices_vec[i] = R * local.vertices_vec[i] + T;
		}
		RenderObjectColor * p_camera = new RenderObjectColor();
		p_camera->SetVertices(local.vertices_vec);
		p_camera->SetFaces(local.faces_v_vec);
		p_camera->SetNormal(local.normals_vec);
		p_camera->SetColor(Eigen::Vector3f(0,0,0));
		p_camera->isFill = false; 
		colorObjs.push_back(p_camera);
	}
}