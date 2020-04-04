#include "main.h"
#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 

#include "../utils/colorterminal.h" 
#include "../utils/obj_reader.h"
#include "../utils/timer_util.h"
#include "../render/render_object.h"
#include "../render/render_utils.h"
#include "../render/renderer.h"

#include "../smal/pigmodel.h"
#include "../smal/pigsolver.h"

#include "../associate/framedata.h"


#define RUN_SEQ
#define VIS 
#define DEBUG_VIS
//#define LOAD_STATE

using std::vector;

int run_shape()
{
	std::string folder = "D:/Projects/animal_calib/data/pig_model_noeye/";
	std::string pig_config = "D:/Projects/animal_calib/smal/pigmodel_config.json";
	std::string conf_projectFolder = "D:/Projects/animal_calib/";

	SkelTopology topo = getSkelTopoByType("UNIV");
	FrameData frame;
	frame.configByJson(conf_projectFolder + "/associate/config.json");
	int startid = frame.get_start_id();

#ifdef VIS
	//// rendering pipeline. 
	auto CM = getColorMapEigen("anliang_rgb");
	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	
	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
#endif 
	int framenum = frame.get_frame_num();
	PigSolver shapesolver(pig_config); 
	int m_pid = 0; // pig identity to solve now. 

	for (int frameid = startid; frameid < startid + 1; frameid++)
	{
		std::cout << "processing frame " << frameid << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		frame.view_dependent_clean();
		frame.matching_by_tracking();

#ifdef LOAD_STATE
		shapesolver.readState("shapestate.txt");	
		shapesolver.RescaleOriginVertices(); 	
		shapesolver.UpdateNormalOrigin(); 
		shapesolver.UpdateNormalShaped();
		shapesolver.determineBodyParts(); 
		
		shapesolver.UpdateVertices();
		m_renderer.colorObjs.clear(); 
		m_renderer.texObjs.clear(); 

		Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces = shapesolver.GetFaces();
		Eigen::MatrixXf vs = shapesolver.GetVertices().cast<float>();
		Eigen::MatrixXf texcoords = shapesolver.GetTexcoords().cast<float>(); 
		
		//RenderObjectTexture* pig_tex_render = new RenderObjectTexture(); 
		//pig_tex_render->SetFaces(faces); 
		//pig_tex_render->SetVertices(vs); 
		//pig_tex_render->SetTexture(folder + "/piguv1.png");
		//pig_tex_render->SetTexcoords(texcoords); 
		//m_renderer.texObjs.push_back(pig_tex_render); 

		RenderObjectColor* pig_render = new RenderObjectColor();
		pig_render->SetFaces(faces);
		pig_render->SetVertices(vs);
		//Eigen::Vector3f color = rgb2bgr(CM[0]);
		Eigen::Vector3f color(1.0f, 1.0f, 1.0f);
		pig_render->SetColor(color);
		m_renderer.colorObjs.push_back(pig_render);

		m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));
		
		while (!glfwWindowShouldClose(windowPtr))
		{
			m_renderer.Draw(); 
			glfwSwapBuffers(windowPtr); 
			glfwPollEvents(); 
		}

		//auto cameras = frame.get_cameras(); 
		//auto rawimgs = frame.get_imgs_undist(); 
		//cv::Mat raw_pack;
		//packImgBlock(rawimgs, raw_pack); 
		//std::vector<cv::Mat> renders; 
		//for (int camid = 0; camid < cameras.size(); camid++)
		//{
		//	Eigen::Matrix3f R = cameras[camid].R.cast<float>();
		//	Eigen::Vector3f T = cameras[camid].T.cast<float>();
		//	m_renderer.s_camViewer.SetExtrinsic(R, T);
		//	m_renderer.Draw();
		//	cv::Mat cap = m_renderer.GetImage();
		//	renders.push_back(cap);
		//}
		//cv::Mat packRender; 
		//packImgBlock(renders, packRender); 
		//cv::Mat blended = blend_images(packRender, raw_pack, 0.5);
		//std::stringstream ss;
		//ss << "E:/debug_pig/render2/rend_" << frameid << ".png";
		//cv::imwrite(ss.str(), blended); 
#else 
		frame.solve_parametric_model();
		auto models = frame.get_models();
		auto m_matched = frame.get_matched();
		auto m_rois = frame.getROI();

		Eigen::VectorXd pose = models[m_pid]->GetPose(); 
		Eigen::Vector3d trans = models[m_pid]->GetTranslation(); 
		double scale = models[m_pid]->GetScale(); 
		shapesolver.SetPose(pose); 
		shapesolver.SetTranslation(trans); 
		shapesolver.SetScale(scale);
		shapesolver.RescaleOriginVertices(); 
		shapesolver.UpdateNormalOrigin(); 
		shapesolver.UpdateNormalShaped();
		shapesolver.determineBodyPartsByTex(); 
		//cv::Mat packMask;
		//vector<cv::Mat> masks;
		//for (int i = 0; i < m_rois.size(); i++)masks.push_back(m_rois[i].mask);
		//packImgBlock(masks, packMask);
		//cv::namedWindow("mask", cv::WINDOW_NORMAL);
		//cv::imshow("mask", packMask);
		//cv::waitKey();

		int iter = 0; 
		for (; iter < 10; iter++)
		{
			shapesolver.UpdateVertices();
			shapesolver.UpdateVerticesTex();
			m_renderer.colorObjs.clear(); 
			m_renderer.texObjs.clear(); 

			Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces = shapesolver.GetFacesTex();
			Eigen::MatrixXf vs = shapesolver.GetVerticesTex().cast<float>();
			
			Eigen::MatrixXf texcoords = shapesolver.GetTexcoords().cast<float>(); 
			
			//RenderObjectColor* pig_render = new RenderObjectColor();
			//pig_render->SetFaces(faces);
			//pig_render->SetVertices(vs);
			////Eigen::Vector3f color = CM[0];
			//Eigen::Vector3f color(1.0, 1.0, 1.0); 
			//pig_render->SetColor(color);
			//m_renderer.colorObjs.push_back(pig_render);

			RenderObjectTexture* pig_tex_render = new RenderObjectTexture(); 
			pig_tex_render->SetFaces(faces); 
			pig_tex_render->SetVertices(vs); 
			pig_tex_render->SetTexture(folder + "/piguv1.png");
			pig_tex_render->SetTexcoords(texcoords); 
			m_renderer.texObjs.push_back(pig_tex_render); 
			
			auto cameras = frame.get_cameras();
			std::vector<cv::Mat> renders;
			for (int view = 0; view < m_matched[m_pid].view_ids.size(); view++)
			{
				int camid = m_matched[m_pid].view_ids[view];
				Eigen::Matrix3f R = cameras[camid].R.cast<float>();
				Eigen::Vector3f T = cameras[camid].T.cast<float>();
				m_renderer.s_camViewer.SetExtrinsic(R, T);
				m_renderer.Draw();
				cv::Mat capture = m_renderer.GetImage();
				shapesolver.feedData(m_rois[view], models[m_pid]->getBodyState());
				shapesolver.feedRender(capture);
				renders.push_back(capture);
			}

			// debug
			cv::Mat pack_render;
			packImgBlock(renders, pack_render);
			//cv::namedWindow("test", cv::WINDOW_NORMAL);
			//cv::imshow("test", pack_render);
			//cv::waitKey();
			std::stringstream ss; 
			ss << "E:/debug_pig/iter/" << std::setw(6)
				<< iter << ".jpg"; 
			cv::imwrite(ss.str(), pack_render); 

			std::cout << RED_TEXT("iter:") << iter << std::endl;

			shapesolver.iterateStep(iter);
			shapesolver.clearData(); 

			glfwSwapBuffers(windowPtr);
			glfwPollEvents();
		}
		shapesolver.saveState("shapestate.txt"); 
#endif 
	}
	//system("pause"); 
	return 0;
}