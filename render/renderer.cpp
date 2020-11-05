#include <iostream>
#include <string>

#include "renderer.h"
#include "GLFW/glfw3.h" 
#include <cfloat> 
#include <opencv2/opencv.hpp>

#include "cuda_utils_render.h"

CamViewer               Renderer::s_camViewer;
GLFWwindow*             Renderer::s_windowPtr;
Renderer::MOUSE_ACTION  Renderer::s_mouseAction;
Eigen::Vector2f         Renderer::s_beforePos;
float                   Renderer::s_arcballRadius; 
double                  Renderer::s_leftClickTimeSeconds; 

 //#define SHOW_CAM_POSE

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
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
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

	lightingShader = SimpleShader(m_shaderFolder + "/texture_v.shader",
		m_shaderFolder + "/texture_f_multilight.shader");

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
	//Eigen::Vector3f lightPos = s_camViewer.GetPos(); 
	Eigen::Vector3f lightPos = Eigen::Vector3f(0, 0, 3);

	// 2. render scene as normal using the generated depth/shadow map  
	// --------------------------------------------------------------
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for(int i = 0; i < colorObjs.size(); i++)
	{
		if (type == "color")
		{
			colorShader.Use();
			s_camViewer.ConfigShader(colorShader);
			colorShader.SetVec3("light_pos", lightPos);
			colorShader.SetFloat("far_plane", RENDER_FAR_PLANE);
			colorObjs[i]->DrawWhole(colorShader);
		}
		else if (type == "mask")
		{
			maskShader.Use(); 
			s_camViewer.ConfigShader(maskShader); 
			maskShader.SetVec3("light_pos", lightPos); 
			maskShader.SetFloat("far_plane", RENDER_FAR_PLANE); 
			colorObjs[i]->DrawWhole(maskShader); 
		}
		else if (type == "depth")
		{
			positionShader.Use();
			s_camViewer.ConfigShader(positionShader);
			positionShader.SetVec3("light_pos", lightPos);
			positionShader.SetFloat("far_plane", RENDER_FAR_PLANE);
			colorObjs[i]->DrawWhole(positionShader);
		}
	}
	
	for(int i = 0; i < texObjs.size(); i++)
	{
#if 0
		textureShader.Use();
		s_camViewer.ConfigShader(textureShader);
		textureShader.SetVec3("light_pos", lightPos);
		textureShader.SetFloat("far_plane", RENDER_FAR_PLANE);
		textureShader.SetInt("object_texture", 0);
		glActiveTexture(GL_TEXTURE0);
		texObjs[i]->DrawWhole(textureShader);
		
#else
		lightingShader.Use(); 
		s_camViewer.ConfigShader(lightingShader); 
		lightingShader.SetInt("object_texture", 0);
		lightingShader.SetFloat("material.shininess", 1);
		lightingShader.SetFloat("material.diffuse", 0.6f);
		lightingShader.SetFloat("material.specular", 0.01f); 
		lightingShader.SetFloat("material.ambient", 0.5); 

		/*
		   Here we set all the uniforms for the 5/6 types of lights we have. We have to set them manually and index
		   the proper PointLight struct in the array to set each uniform variable. This can be done more code-friendly
		   by defining light types as classes and set their values in there, or by using a more efficient uniform approach
		   by using 'Uniform buffer objects', but that is something we'll discuss in the 'Advanced GLSL' tutorial.
		*/
		std::vector<Eigen::Vector3f> pointLightPositions = {
			Eigen::Vector3f(0,0,4),
			Eigen::Vector3f(3, 0, 1.2),
			Eigen::Vector3f(-3, 0, 1.2),
			Eigen::Vector3f(0,2, 1.2),
			Eigen::Vector3f(0, -2, 1.2),
			Eigen::Vector3f(0,0,-6)
		};
		// directional light
		lightingShader.SetVec3("dirLight.direction", 0, 0, 3.f);
		lightingShader.SetVec3("dirLight.ambient", 0.1f, 0.1f, 0.1f);
		lightingShader.SetVec3("dirLight.diffuse", 0.4f, 0.4f, 0.4f);
		lightingShader.SetVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);
		// point light 1
		lightingShader.SetVec3("pointLights[0].position", pointLightPositions[0]);
		lightingShader.SetVec3("pointLights[0].ambient", 0.05, 0.05, 0.05);
		lightingShader.SetVec3("pointLights[0].diffuse", 0.8f, 0.8f, 0.8f);
		lightingShader.SetVec3("pointLights[0].specular", 1.f, 1.f, 1.f);
		lightingShader.SetFloat("pointLights[0].constant", 1);
		lightingShader.SetFloat("pointLights[0].linear", 0.09);
		lightingShader.SetFloat("pointLights[0].quadratic", 0.032);
		// point light 2
		lightingShader.SetVec3("pointLights[1].position", pointLightPositions[1]);
		lightingShader.SetVec3("pointLights[1].ambient", 0.05f, 0.05f, 0.05f);
		lightingShader.SetVec3("pointLights[1].diffuse", 0.8f, 0.8f, 0.8f);
		lightingShader.SetVec3("pointLights[1].specular", 0.4f, 0.4f, 0.4f);
		lightingShader.SetFloat("pointLights[1].constant", 1.0f);
		lightingShader.SetFloat("pointLights[1].linear", 0.09);
		lightingShader.SetFloat("pointLights[1].quadratic", 0.032);
		// point light 3
		lightingShader.SetVec3("pointLights[2].position", pointLightPositions[2]);
		lightingShader.SetVec3("pointLights[2].ambient", 0.05f, 0.05f, 0.05f);
		lightingShader.SetVec3("pointLights[2].diffuse", 0.8f, 0.8f, 0.8f);
		lightingShader.SetVec3("pointLights[2].specular", 0.4f, 0.4f, 0.4f);
		lightingShader.SetFloat("pointLights[2].constant", 1.0f);
		lightingShader.SetFloat("pointLights[2].linear", 0.09);
		lightingShader.SetFloat("pointLights[2].quadratic", 0.032);
		// point light 4
		lightingShader.SetVec3("pointLights[3].position", pointLightPositions[3]);
		lightingShader.SetVec3("pointLights[3].ambient", 0.05f, 0.05f, 0.05f);
		lightingShader.SetVec3("pointLights[3].diffuse", 0.8f, 0.8f, 0.8f);
		lightingShader.SetVec3("pointLights[3].specular", 0.4f, 0.4f, 0.4f);
		lightingShader.SetFloat("pointLights[3].constant", 1.0f);
		lightingShader.SetFloat("pointLights[3].linear", 0.09);
		lightingShader.SetFloat("pointLights[3].quadratic", 0.032);
		// point light 5
		lightingShader.SetVec3("pointLights[4].position", pointLightPositions[4]);
		lightingShader.SetVec3("pointLights[4].ambient", 0.05f, 0.05f, 0.05f);
		lightingShader.SetVec3("pointLights[4].diffuse", 0.8f, 0.8f, 0.8f);
		lightingShader.SetVec3("pointLights[4].specular", 0.4f, 0.4f, 0.4f);
		lightingShader.SetFloat("pointLights[4].constant", 1.0f);
		lightingShader.SetFloat("pointLights[4].linear", 0.09);
		lightingShader.SetFloat("pointLights[4].quadratic", 0.032);
		// point light 6
		lightingShader.SetVec3("pointLights[5].position", pointLightPositions[5]);
		lightingShader.SetVec3("pointLights[5].ambient", 0.05f, 0.05f, 0.05f);
		lightingShader.SetVec3("pointLights[5].diffuse", 0.8f, 0.8f, 0.8f);
		lightingShader.SetVec3("pointLights[5].specular", 0.4f, 0.4f, 0.4f);
		lightingShader.SetFloat("pointLights[5].constant", 1.0f);
		lightingShader.SetFloat("pointLights[5].linear", 0.09);
		lightingShader.SetFloat("pointLights[5].quadratic", 0.032);

		// spotLight
		lightingShader.SetVec3("spotLight.position", s_camViewer.GetPos());
		lightingShader.SetVec3("spotLight.direction", s_camViewer.GetFront());
		lightingShader.SetVec3("spotLight.ambient", 0.0f, 0.0f, 0.0f);
		lightingShader.SetVec3("spotLight.diffuse", 1.0f, 1.0f, 1.0f);
		lightingShader.SetVec3("spotLight.specular", 1.0f, 1.0f, 1.0f);
		lightingShader.SetFloat("spotLight.constant", 1.0f);
		lightingShader.SetFloat("spotLight.linear", 0.09);
		lightingShader.SetFloat("spotLight.quadratic", 0.032);
		lightingShader.SetFloat("spotLight.cutOff", cosf(12.5f/180*M_PI));
		lightingShader.SetFloat("spotLight.outerCutOff", cosf(15.f / 180 * M_PI));
		
		glActiveTexture(GL_TEXTURE0);
		texObjs[i]->DrawWhole(lightingShader);
#endif
	}

	for(int i = 0; i < meshObjs.size(); i++)
	{
		if (type == "color")
		{
			meshShader.Use();
			s_camViewer.ConfigShader(meshShader);
			meshShader.SetVec3("light_pos", lightPos);
			meshShader.SetFloat("far_plane", RENDER_FAR_PLANE);
			meshObjs[i]->DrawWhole(meshShader);
		}
	}

	for(int i = 0; i < skels.size(); i++)
	{
		if (type == "color")
		{
			colorShader.Use();
			s_camViewer.ConfigShader(colorShader);

			colorShader.SetVec3("light_pos", lightPos);
			colorShader.SetFloat("far_plane", RENDER_FAR_PLANE);

			skels[i]->Draw(colorShader);
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
	Mesh obj;
	obj.Load(conf_projectFolder + "/data/calibdata/scene_model/manual_scene_part0.obj");
	//obj = ballMesh; 
	RenderObjectTexture* p_scene = new RenderObjectTexture();
	p_scene->SetTexture(conf_projectFolder + "/render/data/previous_board.png");
	p_scene->SetFaces(obj.faces_v_vec);
	p_scene->SetVertices(obj.vertices_vec);
	p_scene->SetNormal(obj.normals_vec, 2);
	p_scene->SetTexcoords(obj.textures_vec, 1);
	p_scene->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	texObjs.push_back(p_scene);

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
	p_scene2->SetColor(Eigen::Vector3f(0.9, 0.9, 0.95));
	colorObjs.push_back(p_scene2);

	Mesh obj3;
	obj3.Load(conf_projectFolder + "/data/calibdata/scene_model/manual_scene_part2.obj");
	RenderObjectColor * p_scene3 = new RenderObjectColor();
	p_scene3->SetVertices(obj3.vertices_vec);
	p_scene3->SetNormal(obj3.normals_vec);
	p_scene3->SetFaces(obj3.faces_v_vec);
	p_scene3->SetColor(Eigen::Vector3f(0.8, 0.8, 0.75));
	colorObjs.push_back(p_scene3);
}