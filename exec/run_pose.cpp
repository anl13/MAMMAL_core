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
#include "../associate/framedata.h"
#include "../utils/mesh.h"
#include "../nanorender/NanoRenderer.h"
#include <vector_functions.hpp>
#include "main.h"
#include "../utils/image_utils_gpu.h"

using std::vector;

//#define READ_SMOOTH
//#define RESUME

std::vector<float4> convertVertices(const Eigen::Matrix3Xf& v)
{
	int N = v.cols();
	std::vector<float4> vertices;
	vertices.resize(N);
#pragma omp parallel for 
	for (int i = 0; i < N; i++)
	{
		vertices[i].x = v(0, i);
		vertices[i].y = v(1, i);
		vertices[i].z = v(2, i);
		vertices[i].w = 1;
	}
	return vertices;
}

std::vector<unsigned int> convertIndices(const Eigen::MatrixXu& f)
{
	int N = f.cols();
	std::vector<unsigned int> indices(N * 3, 0);
	for (int i = 0; i < N; i++)
	{
		indices[3 * i + 0] = f(0, i);
		indices[3 * i + 1] = f(1, i);
		indices[3 * i + 2] = f(2, i);
	}
	return indices;
}

nanogui::Matrix4f eigen2nanoM4f(const Eigen::Matrix4f& mat)
{
	nanogui::Matrix4f M;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			M.m[j][i] = mat(i, j);
		}
	}
	return M;
}


std::vector<float4> getColorMapFloat4(std::string cm_type)
{
	std::vector<Eigen::Vector3i> CM;
	getColorMap(cm_type, CM);
	std::vector<float4> CM4;
	if (CM.size() > 0)
	{
		CM4.resize(CM.size());
		for (int i = 0; i < CM.size(); i++)
		{
			CM4[i] = make_float4(
				CM[i](0) / 255.f, CM[i](1) / 255.f, CM[i](2) / 255.f, 1.0f);
		}
	}
	return CM4;
}

std::vector<Camera> readCameras()
{
	std::vector<Camera> cams;
	std::vector<int> m_camids = {
		0,1,2,5,6,7,8,9,10,11
	};
	int m_camNum = m_camids.size();
	std::string m_camDir = "D:/Projects/animal_calib/data/calibdata/adjust/";
	for (int camid = 0; camid < m_camNum; camid++)
	{
		std::stringstream ss;
		ss << m_camDir << std::setw(2) << std::setfill('0') << m_camids[camid] << ".txt";
		std::ifstream camfile;
		camfile.open(ss.str());
		if (!camfile.is_open())
		{
			std::cout << "can not open file " << ss.str() << std::endl;
			exit(-1);
		}
		Eigen::Vector3f rvec, tvec;
		for (int i = 0; i < 3; i++) {
			float a;
			camfile >> a;
			rvec(i) = a;
		}
		for (int i = 0; i < 3; i++)
		{
			float a;
			camfile >> a;
			tvec(i) = a;
		}

		Camera camUndist = Camera::getDefaultCameraUndist();
		camUndist.SetRT(rvec, tvec);
		cams.push_back(camUndist);
		camfile.close();
	}
	return cams;
}


int run_pose()
{
	//std::string pig_config = "D:/Projects/animal_calib/articulation/pigmodel_config.json";
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");

	FrameData frame;
	frame.configByJson(conf_projectFolder + "/associate/config.json");
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();
	//PigSolver shapesolver(pig_config);

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

	frame.result_folder = "G:/pig_results/";
	frame.is_smth = false; 
	int start = frame.get_start_id();
	std::vector<cv::Mat> rawImgs = frame.get_imgs_undist(); 
	cv::Mat pack_raw; 
	packImgBlock(rawImgs, pack_raw); 
	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "processing frame " << frameid << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		//frame.view_dependent_clean();

#ifdef READ_SMOOTH 
		frame.load_clusters();
		frame.read_parametric_data(); 
		//std::stringstream state_file;
		//state_file << "G:/pig_results_level1_nosil/state_smth/pig_" << m_pid << "_frame_" <<
		//	std::setw(6) << std::setfill('0') << frameid << ".txt";
		//shapesolver.readState(state_file.str());
		//shapesolver.UpdateVertices();
#else 
		if(frameid == start) // load state 
		{
#ifndef RESUME
			frame.matching_by_tracking(); 
			frame.solve_parametric_model(); 
			//frame.solve_parametric_model_cpu();

			frame.save_clusters();
			//frame.save_parametric_data();
#else 
			frame.load_clusters();
			//frame.solve_parametric_model(); 
			frame.read_parametric_data();
			std::stringstream init_state_file;
			init_state_file << "E:/pig_results/state/pig_" << m_pid << "_frame_" <<
				std::setw(6) << std::setfill('0') << frameid << ".txt";
			shapesolver.readState(init_state_file.str());
			shapesolver.UpdateVertices(); 
			continue; 
#endif // RESUME
		}
		else {
			frame.matching_by_tracking();
			//frame.pureTracking();
			frame.solve_parametric_model();
			

			frame.save_clusters();
			frame.save_parametric_data();
		}

#endif // READ_SMOOTH
		auto solvers = frame.mp_bodysolverdevice;
		//auto solvers = frame.mp_bodysolver;

		m_renderer.clearAllObjs(); 
		RenderObjectColor* p_model = new RenderObjectColor();
		solvers[0]->UpdateNormalFinal();
		
		p_model->SetVertices(solvers[0]->GetVertices()); 
		p_model->SetNormal(solvers[0]->GetNormals()); 
		p_model->SetFaces(solvers[0]->GetFacesVert());
		p_model->SetColor(Eigen::Vector3f(0.8f, 0.6f, 0.4f));
		m_renderer.colorObjs.push_back(p_model); 

		std::vector<cv::Mat> all_renders(cams.size());
		for (int camid = 0; camid < cams.size(); camid++)
		{
			m_renderer.s_camViewer.SetExtrinsic(cams[camid].R, cams[camid].T);
			m_renderer.Draw(); 
			cv::Mat img = m_renderer.GetImage();
			
			all_renders[camid] = img;
		}
		cv::Mat packed_render; 
		packImgBlock(all_renders, packed_render);
		
		cv::Mat blend;
		overlay_render_on_raw_gpu(packed_render, pack_raw, blend); 

		std::stringstream all_render_file; 
#ifndef READ_SMOOTH
		all_render_file << "G:/pig_results/render_all/" << std::setw(6) << std::setfill('0')
			<< frameid << "_gpu.png";
#else 
		all_render_file << "G:/pig_results_level1_nosil/render_all_smth/" << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
#endif // READ_SMOOTH
		cv::imwrite(all_render_file.str(), blend); 
	}

	system("pause"); 
	return 0;
}
