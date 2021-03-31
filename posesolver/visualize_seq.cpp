#include "main.h"
#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>

#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "../articulation/pigmodel.h"
#include "../articulation/pigsolver.h"
#include "framesolver.h"
#include "../utils/mesh.h"
#include <vector_functions.hpp>
#include "main.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"


std::vector<Eigen::Vector3f> extract_pose_centers(int id, int start, int framenum, std::string dir)
{
	PigModelDevice pig("../articulation/artist_config.json");
	std::vector<Eigen::Vector3f> centers; 
	for (int i = start; i < start+framenum; i++)
	{
		std::cout << i << std::endl;
		std::stringstream ss;
		ss <<  dir << "/state/pig_" << id << "_frame_" << std::setw(6) << std::setfill('0') << i << ".txt"; 
		pig.readState(ss.str()); 
		pig.UpdateVertices(); 
		std::vector<Eigen::Vector3f> joints = pig.GetJoints(); 
		centers.push_back(joints[2]);
	}
	return centers; 
}

std::vector<Eigen::Vector3f> extract_pose_centers(std::string filename)
{
	std::ifstream file(filename); 
	std::vector<Eigen::Vector3f> points; 
	while (true)
	{
		float x, y, z;
		file >> x; 
		if (file.eof()) break; 
		file >> y; 
		if (file.eof())break; 
		file >> z; 
		points.emplace_back(Eigen::Vector3f(x, y, z)); 
	}
	file.close(); 
	return points; 
}

void save_points(std::string  outfile, const std::vector<Eigen::Vector3f>& points)
{
	std::ofstream file(outfile); 
	for (int i = 0; i < points.size(); i++)
	{
		file << points[i].transpose() << std::endl; 
	}
	file.close(); 
	return; 
}

void visualize_seq()
{
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_render");

	FrameSolver frame;
	frame.configByJson(conf_projectFolder + "/posesolver/config.json");
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(0);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];

	// init renderer
	Eigen::Matrix3f K = cam.K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(false);
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

	frame.mp_renderEngine = &m_renderer;

	frame.m_result_folder = "G:/pig_results_newtrack/";
	frame.is_smth = false;


	m_renderer.clearAllObjs();
	m_renderer.createScene(conf_projectFolder); 
	// add sequence 
	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");


	std::vector<Eigen::Vector3f> seq0 = extract_pose_centers(0, startid, framenum, frame.m_result_folder);
	save_points("../result_data/traj/seq0.txt", seq0); 

	std::vector<float> sizes(seq0.size(), 0.01);
	std::vector<Eigen::Vector3f> colors(seq0.size(), m_CM[0]);
	for (int i = 0; i < seq0.size(); i++)
	{
		float ratio = 1 - float(i) / framenum * 0.2 ;
		colors[i] = colors[i] * ratio;
	}
	BallStickObject* trajectory0 = new BallStickObject(ballMesh, seq0, sizes, colors); 

	m_renderer.skels.push_back(trajectory0); 
	
	while (!glfwWindowShouldClose(windowPtr))
	{
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}
