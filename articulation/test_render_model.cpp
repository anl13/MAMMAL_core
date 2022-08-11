#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <filesystem>
#include <io.h> 
#include <process.h> 

#include "../render/renderer.h"
#include "../render/render_object.h" 
#include "../render/render_utils.h"
#include "../utils/camera.h"
#include "../utils/math_utils.h" 
#include "../utils/image_utils.h" 
#include "../utils/definitions.h" 
#include "../utils/mesh.h"
#include "../utils/timer_util.h"

#include "pigmodeldevice.h" 
#include "pigsolverdevice.h" 

#include "test_main.h"

// This function is used for generating demonstration of the PIG model (Paper Fig. 1b) 
// Also, you can get some basic usage of the PIG model, e.g., how to load it, how to render it, how to access vertices and joints, 
//     how to regress keypoints. 
// Rendered images are saved in "PROJECT_FOLDER/articulation/tmp/" folder. Here, "PROJECT_FOLDER" is where you put your project in.

int render_mean_pose()
{
	// render config 
	std::cout << "Render mean pose now!" << std::endl;
	std::string conf_projectFolder = PROJECT_FOLDER; 
	std::string tmp_folder = conf_projectFolder + "/articulation/tmp/"; 
	std::cout << "result folder: " << tmp_folder << std::endl; 
	std::cout << "current path : " << std::filesystem::current_path() << std::endl;
	if (!std::filesystem::is_directory(tmp_folder))
	{
		std::filesystem::create_directories(tmp_folder); 
	}
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");
	std::vector<Eigen::Vector3f> CM2 = getColorMapEigenF("anliang_rgb");
	std::vector<Eigen::Vector3f> CM3 = getColorMapEigenF("anliang_blend");

	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698f, 0.f, 0.502f,
		0.f, 1.243f, 0.483f,
		0.f, 0.f, 1.f;
	std::cout << K << std::endl;

	//// this parameter is used for NM paper. 
	Eigen::Vector3f up(-0.077325, 0.180478, 0.980535);
	Eigen::Vector3f pos(0.563461, -0.829368, 0.181309);
	Eigen::Vector3f center(0.131863, 0.0613784, -0.0129145);

	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(1, 1, 1, 1));

	// init element OBJ model for skeleton rendering. 
	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	// load the PIG model data 
	std::string pig_config = conf_projectFolder + "/articulation/PIG_model.json";
	PigModelDevice pig(pig_config);
	pig.UpdateVertices();
	pig.UpdateNormalFinal();

	// 1. render surface 
	{
		RenderObjectColor * animal_model = new RenderObjectColor();
		animal_model->SetFaces(pig.GetFacesVert());
		animal_model->SetVertices(pig.GetVertices());
		animal_model->SetNormal(pig.GetNormals());
		animal_model->SetColor(Eigen::Vector3f(1, 1, 1));
		animal_model->isMultiLight = true;
		animal_model->isFill = true;
		m_renderer.colorObjs.push_back(animal_model);

		// render and save 
		cv::Mat img = m_renderer.GetImageOffscreen();
		cv::imwrite(tmp_folder + "/surface.png", img);
		animal_model->isFill = false; 
	}

	// 2. render driven joints 
	if (1) {
		std::vector<Eigen::Vector3f> joints = pig.GetJoints();
		std::vector<int> parents = pig.GetParents();
		std::vector<Eigen::Vector3f> balls;
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
		GetBallsAndSticks(joints, parents, balls, sticks);
		int jointnum = pig.GetJointNum();
		std::vector<float> sizes;
		sizes.resize(jointnum, 0.01);
		std::vector<int> ids = { 0,1,2,3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 38, 39, 40, 41, 54, 55, 56, 57, 21, 22, 23 };
		std::vector<Eigen::Vector3f> colors;
		colors.resize(jointnum, Eigen::Vector3f(1.0, 0.95, 0.85));
		for (int k = 0; k < ids.size(); k++)
		{
			colors[ids[k]] = CM[2];
			sizes[ids[k]] = 0.012;
		}
		std::vector<float> bone_sizes(sticks.size(), 0.005);
		std::vector<Eigen::Vector3f> bone_colors(sticks.size());
		for (int k = 0; k < sticks.size(); k++)
		{
			bone_colors[k] = Eigen::Vector3f(1.0, 0.95, 0.85);
		}
		BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen, balls, sticks, sizes, bone_sizes, colors, bone_colors);
		p_skel->isMultiLight = false;
		m_renderer.skels.push_back(p_skel);

		// render and save 
		cv::Mat img = m_renderer.GetImageOffscreen();
		cv::imwrite(tmp_folder + "/surface+joints.png", img);
	}
	m_renderer.clearSkels(); 

    // 3. render skeleton
	if (0) {
		std::vector<Eigen::Vector2i> bones = {
			{0,1}, {0,2}, {1,2}, {1,3}, {2,4},
			 {5,7}, {7,9}, {6,8}, {8,10},
			{20,18},
			{18,11}, {18,12}, {11,13}, {13,15}, {12,14}, {14,16},
			{0,20},{5,20},{6,20}
		};
		std::vector<int> kpt_color_ids = {
			0,0,0,0,0,
			3,4,3,4,3,4,
			5,6,5,6,5,6,
			2,2,2,2,2,2
		};
		std::vector<int> bone_color_ids = {
			0,0,0,0,0,3,3,4,4,
			2,5,6,5,5,6,6,
			2,3,4
		};
		std::vector<Eigen::Vector3f> skels = pig.getRegressedSkel_host();
		std::vector<Eigen::Vector3f> balls;
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
		GetBallsAndSticks(skels, bones, balls, sticks);
		int jointnum = skels.size();
		std::vector<float> ball_sizes;
		ball_sizes.resize(jointnum, 0.015);
		std::vector<float> stick_sizes;
		stick_sizes.resize(sticks.size(), 0.008);
		std::vector<Eigen::Vector3f> ball_colors(jointnum);
		std::vector<Eigen::Vector3f> stick_colors(sticks.size());
		for (int i = 0; i < jointnum; i++)
		{
			ball_colors[i] = CM3[kpt_color_ids[i]];
		}
		for (int i = 0; i < sticks.size(); i++)
		{
			stick_colors[i] = CM3[bone_color_ids[i]];
		}


		BallStickObject* p_skel = new BallStickObject(ballMeshEigen, stickMeshEigen,
			balls, sticks, ball_sizes, stick_sizes, ball_colors, stick_colors);
		m_renderer.skels.push_back(p_skel);

		// render and save
		cv::Mat img = m_renderer.GetImageOffscreen();
		cv::imwrite(tmp_folder + "/surface+skel.png", img);
	}

	// open render panel for free-viewpoint visualization
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

	return 0;
}