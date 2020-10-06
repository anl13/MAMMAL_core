#include <vector>
#include <string> 

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "shapesolver.h"
#include "../posesolver/framedata.h" 
#include "../posesolver/framesolver.h"
#include "../utils/show_gpu_param.h"
#include "assist_functions.h" 
#include "../utils/node_graph.h"


// solve surface deformation to visual hull
// not work well 2020/10/06
int solve_shape()
{
	show_gpu_param(); 
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render"); 

	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config.json";
	ShapeSolver solver(pig_config); 

	// load data 
	FrameSolver framereader; 
	framereader.configByJson("D:/Projects/animal_calib/posesolver/config.json"); 
	framereader.set_frame_id(0); 
	framereader.fetchData(); 
	framereader.result_folder = "G:/pig_results_newtrack";
	framereader.load_clusters(); 

	// config shapesovler
	std::vector<Camera> cameras = framereader.get_cameras(); 
	solver.setCameras(cameras); 
	solver.normalizeCamera(); 

	// init renderer
	Eigen::Matrix3f K = cameras[0].K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(false);
	Renderer m_renderer("D:/Projects/animal_calib/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

	framereader.mp_renderEngine = &m_renderer; 
	solver.mp_renderEngine = &m_renderer; 

	// solve

	std::vector<MatchedInstance> matcheddata = framereader.get_matched(); 
	solver.setSource(matcheddata[0]); 
	solver.normalizeSource();
	solver.globalAlign(); 
	solver.optimizePose(); 

	RenderObjectColor* p_model0 = new RenderObjectColor();
	solver.UpdateNormalFinal();
	p_model0->SetVertices(solver.GetVertices());
	p_model0->SetNormal(solver.GetNormals());
	p_model0->SetFaces(solver.GetFacesVert());
	p_model0->SetColor(CM[1]);

	std::vector<ROIdescripter> rois; 
	framereader.getROI(rois, 0);
	solver.m_rois = rois; 
	//solver.optimizePoseSilhouette(10); 
	
	
	//for (int i = 0; i < solver.m_rois.size(); i++)
	//{
	//	cv::imshow("mask", solver.m_rois[i].mask * 40); 
	//	int key = cv::waitKey(); 
	//	if (key == 27)
	//		exit(-1); 
	//	cv::destroyAllWindows();

	//}
	solver.InitNodeAndWarpField(); 
	Mesh gthull_vec;
	gthull_vec.Load("H:/pig_results_shape/tmp.obj");
	std::shared_ptr<MeshEigen> p_gthull = std::make_shared<MeshEigen>(gthull_vec); 
	solver.setTargetModel(p_gthull); 
	solver.totalSolveProcedure(); 
	solver.SaveWarpField(); 

	solver.SaveObj("deformed.obj"); 
	solver.ResetPose(); 
	solver.UpdateVertices(); 
	solver.SaveObj("deformed_tpose.obj"); 

	//rendering

	RenderObjectColor* p_model = new RenderObjectColor();
	solver.UpdateNormalFinal(); 
	p_model->SetVertices(solver.GetVertices());
	p_model->SetNormal(solver.GetNormals());
	p_model->SetFaces(solver.GetFacesVert());
	p_model->SetColor(CM[0]);
	
	m_renderer.clearAllObjs(); 
	m_renderer.colorObjs.push_back(p_model0); 
	m_renderer.colorObjs.push_back(p_model);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

	return 0; 

}


void test_bone_var()
{
	show_gpu_param();
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render");

	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config.json";
	ShapeSolver solver(pig_config);

	// load data 
	FrameSolver framereader;
	framereader.configByJson("D:/Projects/animal_calib/posesolver/config.json");
	framereader.set_frame_id(0);
	framereader.fetchData();
	framereader.result_folder = "G:/pig_results_newtrack";
	framereader.load_clusters();

	// config shapesovler
	std::vector<Camera> cameras = framereader.get_cameras();
	solver.setCameras(cameras);
	solver.normalizeCamera();

	// init renderer
	Eigen::Matrix3f K = cameras[0].K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(false);
	Renderer m_renderer("D:/Projects/animal_calib/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

	framereader.mp_renderEngine = &m_renderer;
	solver.mp_renderEngine = &m_renderer;

	// solve

	//std::vector<MatchedInstance> matcheddata = framereader.get_matched();
	//solver.setSource(matcheddata[0]);
	//solver.normalizeSource();
	//solver.globalAlign();
	//solver.optimizePose();


	RenderObjectColor* p_model0 = new RenderObjectColor();
	solver.UpdateNormalFinal();
	p_model0->SetVertices(solver.GetVertices());
	p_model0->SetNormal(solver.GetNormals());
	p_model0->SetFaces(solver.GetFacesVert());
	p_model0->SetColor(CM[1]);

	//std::vector<ROIdescripter> rois;
	//framereader.getROI(rois, 0);
	//solver.m_rois = rois;
	////solver.optimizePoseSilhouette(10); 

	//rendering

	solver.m_bone_extend[1](2) = -0.02;
	solver.UpdateVertices(); 
	RenderObjectColor* p_model = new RenderObjectColor();
	solver.UpdateNormalFinal();
	p_model->SetVertices(solver.GetVertices());
	p_model->SetNormal(solver.GetNormals());
	p_model->SetFaces(solver.GetFacesVert());
	p_model->SetColor(CM[0]);

	m_renderer.clearAllObjs();
	m_renderer.colorObjs.push_back(p_model0);
	m_renderer.colorObjs.push_back(p_model);

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};

}

void main()
{
	solve_shape(); 
	//test_bone_var(); 
}