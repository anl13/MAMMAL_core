#include "main.h"
#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>

#include "../utils/colorterminal.h" 
#include "../utils/obj_reader.h"
#include "../utils/timer_util.h"
#include "../smal/pigmodel.h"
#include "../smal/pigsolver.h"
#include "../associate/framedata.h"
#include "../utils/volume.h"
#include "../utils/model.h"
#include "../utils/dataconverter.h" 
#include "../nanorender/NanoRenderer.h"
#include <vector_functions.hpp>
#include "../utils/timer.hpp" 
#include "main.h"

using std::vector;

//#define READ_SMOOTH
//#define RESUME

int run_pose()
{
	std::string pig_config = "D:/Projects/animal_calib/smal/pigmodel_config.json";
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");

	FrameData frame;
	frame.configByJson(conf_projectFolder + "/associate/config.json");
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();
	PigSolver shapesolver(pig_config);

	int m_pid = 0; // pig identity to solve now. 
	frame.set_frame_id(0);
	frame.fetchData();
	auto cams = frame.get_cameras();
	auto cam = cams[0];

	NanoRenderer renderer;
	renderer.Init(1920, 1080, float(cam.K(0, 0)), float(cam.K(1, 1)), float(cam.K(0, 2)), float(cam.K(1, 2)), -3.f);
	auto animal_model = renderer.CreateOffscreenRenderObject(
		"animal", vs_vertex_position, fs_vertex_position, 1920, 1080, cam.K(0, 0), cam.K(1, 1), cam.K(0, 2), cam.K(1, 2), 1, true);
	shapesolver.animal_offscreen = animal_model; 
	Eigen::Matrix4f view_eigen = calcRenderExt(cam.R.cast<float>(), cam.T.cast<float>());
	nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
	renderer.UpdateCanvasView(view_eigen);

	auto animal_model_render = renderer.CreateOffscreenRenderObject(
		"animal", vs_phong_geometry, fs_phong_geometry, 1920, 1080, cam.K(0, 0), cam.K(1, 1), cam.K(0, 2), cam.K(1, 2), 1, false);
	animal_model_render->_SetViewByCameraRT(cam.R, cam.T);


	int start = frame.get_start_id();
	for (int frameid = start; frameid < start + frame.get_frame_num(); frameid++)
	{
		std::cout << "processing frame " << frameid << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		//frame.view_dependent_clean();

#ifdef READ_SMOOTH 
		frame.load_clusters();
		std::stringstream state_file;
		state_file << "E:/pig_results/state_smth/pig_" << m_pid << "_frame_" <<
			std::setw(6) << std::setfill('0') << frameid << ".txt";
		shapesolver.readState(state_file.str());
		shapesolver.UpdateVertices();
#else 
		if(frameid == start) // load state 
		{
#ifndef RESUME
			frame.matching_by_tracking(); 
			frame.solve_parametric_model(); 
			frame.save_clusters();
			frame.save_parametric_data();
#else 
			frame.load_clusters();
			//frame.solve_parametric_model(); 
			frame.load_parametric_data();
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
			frame.solve_parametric_model();
			frame.save_clusters();
			frame.save_parametric_data();
		}


		auto m_matched = frame.get_matched();
		cv::Mat det_img = frame.visualizeIdentity2D(-1);
		std::stringstream ss1;
		ss1 << "E:/pig_results/assoc/" << std::setw(6) << std::setfill('0')
			<< frameid << ".jpg";
		cv::imwrite(ss1.str(), det_img);
		frame.debug_fitting(0);
		
		auto m_rois = frame.getROI(m_pid);
		shapesolver.setFrameId(frameid - start);
		shapesolver.setCameras(frame.get_cameras());
		shapesolver.normalizeCamera();
		shapesolver.setId(m_pid);
		shapesolver.setSource(m_matched[m_pid]);
		shapesolver.normalizeSource();
		shapesolver.InitNodeAndWarpField();
		//shapesolver.LoadWarpField();
		shapesolver.UpdateVertices();
		shapesolver.globalAlign();
		shapesolver.optimizePose(10, 0.001);
		shapesolver.m_rois = m_rois;
		shapesolver.m_rawImgs = frame.get_imgs_undist();
		shapesolver.optimizePoseSilhouette(18);


#ifndef DEBUG_SIL
		std::stringstream state_file;
		state_file << "E:/pig_results/state/pig_" << m_pid << "_frame_" <<
			std::setw(6) << std::setfill('0') << frameid << ".txt";
		shapesolver.saveState(state_file.str());
#endif // DEBUG_SIL

#endif // READ_SMOOTH
		Model m3c;
		m3c.vertices = shapesolver.GetVertices();
		m3c.faces = shapesolver.GetFacesVert();
		m3c.CalcNormal();
		ObjModel m4c;
		convert3CTo4C(m3c, m4c);

		auto human_model = renderer.CreateRenderObject("human_model", vs_phong_geometry, fs_phong_geometry);
		human_model->SetIndices(m4c.indices);
		human_model->SetBuffer("positions", m4c.vertices);
		human_model->SetBuffer("normals", m4c.normals);
		renderer.ApplyCanvasView();

		animal_model_render->SetBuffer("positions", m4c.vertices);
		animal_model_render->SetBuffer("normals", m4c.normals);
		animal_model_render->SetIndices(m4c.indices); 

		std::vector<cv::Mat> all_renders(cams.size());
		for (int camid = 0; camid < cams.size(); camid++)
		{
			Eigen::Matrix4f camview = Eigen::Matrix4f::Identity();
			camview.block<3, 3>(0, 0) = cams[camid].R.cast<float>();
			camview.block<3, 1>(0, 3) = cams[camid].T.cast<float>();
			nanogui::Matrix4f camview_nano = eigen2nanoM4f(camview);
			animal_model_render->_SetViewByCameraRT(cams[camid].R, cams[camid].T);
			animal_model_render->SetUniform("view", camview_nano);
			animal_model_render->DrawOffscreen();
			std::vector<cv::Mat> rendered_imgs;
			cv::Mat rendered_img(1920, 1080, CV_8UC4);
			rendered_imgs.push_back(rendered_img);
			animal_model_render->DownloadRenderingResults(rendered_imgs);
			all_renders[camid] = rendered_imgs[0];
		}
		cv::Mat packed_render; 
		packImgBlock(all_renders, packed_render);
		std::stringstream all_render_file; 
#ifndef READ_SMOOTH
		all_render_file << "E:/pig_results/render_all/" << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
#else 
		all_render_file << "E:/pig_results/render_all_smth/" << std::setw(6) << std::setfill('0')
			<< frameid << ".png";
#endif // READ_SMOOTH
		cv::imwrite(all_render_file.str(), packed_render); 

#ifdef DEBUG_SIL
		int r_frameIdx = 0;
		while (!renderer.ShouldClose())
		{
			renderer.Draw();
			++r_frameIdx;
		}
		break; 
#endif 
	}
	return 0;
}