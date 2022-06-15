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
#include "framesolver.h"
#include "../utils/mesh.h"
#include <vector_functions.hpp>
#include "main.h"
#include "../utils/image_utils_gpu.h"
#include "../utils/show_gpu_param.h"


std::vector<Eigen::Vector3f> extract_pose_nose(int id, int start, int framenum, std::string dir)
{
	PigModelDevice pig("../articulation/artist_config_sym.json");
	std::vector<Eigen::Vector3f> centers; 
	for (int i = start; i < start+framenum; i++)
	{
		std::cout << i << std::endl;
		std::stringstream ss;
		ss <<  dir << "/state/pig_" << id << "_frame_" << std::setw(6) << std::setfill('0') << i << ".txt"; 
		pig.readState(ss.str()); 
		pig.UpdateVertices(); 
		std::vector<Eigen::Vector3f> joints = pig.getRegressedSkel_host();
		centers.push_back(joints[0]);
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

// visualize trajectory for animal paper
void visualize_seq()
{
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");
	std::vector<Eigen::Vector3f> m_CM = getColorMapEigenF("anliang_paper");

	FrameSolver frame;
	frame.configByJson(conf_projectFolder + "/configs/config_20191003_socialdemo.json");
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();

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
	frame.is_smth = true;

	m_renderer.clearAllObjs();
	m_renderer.createSceneDetailed(conf_projectFolder, 1.1); 
	// add sequence 
	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");


	std::vector<Eigen::Vector3f> seq2 = extract_pose_nose(2, startid, framenum, frame.m_result_folder);
	std::vector<Eigen::Vector3f> seq0 = extract_pose_nose(0, startid, framenum, frame.m_result_folder); 
	
	std::vector<float> sizes2(seq2.size(), 0.01); // WT2 20191003
	std::vector<Eigen::Vector3f> colors2(seq2.size(), m_CM[1]); // WT2 20191003
	BallStickObject* trajectory2 = new BallStickObject(ballMesh, seq2, sizes2, colors2); 

	std::vector<float> sizes0(seq0.size(), 0.01); // F0-1 20191003
	std::vector<Eigen::Vector3f> colors0(seq0.size(), m_CM[2]); // F0-1 20191003
	BallStickObject* trajectory0 = new BallStickObject(ballMesh, seq0, sizes0, colors0);


	m_renderer.skels.push_back(trajectory2);
	m_renderer.skels.push_back(trajectory0);

	
	while (!glfwWindowShouldClose(windowPtr))
	{
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
}
