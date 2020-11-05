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
#include <json/json.h>
#include "../utils/image_utils_gpu.h" 

void readObs(std::vector<SingleObservation>& obs, std::string configfile, int pigid)
{
	obs.clear(); 

	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(configfile);
	if (!instream.is_open())
	{
		std::cout << "can not open " << configfile << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}

	FrameSolver framereader;
	framereader.configByJson("D:/Projects/animal_calib/posesolver/config.json");
	framereader.result_folder = "H:/pig_results_debug";

	for (const auto& data : root["frames"])
	{
		int frameid = data["frameid"].asInt();
		std::vector<int> usedviews;
		for (const auto& c : data["views"])
		{
			usedviews.emplace_back(c.asInt()); 
		}
		framereader.set_frame_id(frameid); 
		framereader.fetchData(); 
		framereader.load_clusters(); 
		framereader.read_parametric_data(); 
		SingleObservation ob;
		ob.source = framereader.m_matched[pigid];
		ob.usedViews = usedviews;
		std::vector<Eigen::Vector3f> pose = framereader.mp_bodysolverdevice[pigid]->GetPose(); 
		Eigen::VectorXf poseeigen;
		poseeigen.resize(3 * pose.size()); 
		for (int k = 0; k < pose.size(); k++)poseeigen.segment<3>(3 * k) = pose[k];
		ob.pose = poseeigen; 
		ob.translation = framereader.mp_bodysolverdevice[pigid]->GetTranslation(); 
		ob.scale = framereader.mp_bodysolverdevice[pigid]->GetScale(); 
		framereader.getROI(ob.rois, pigid); 
		obs.push_back(ob); 
	}

}

// solve surface deformation to visual hull
// not work well 2020/10/06
int solve_shape()
{
	show_gpu_param(); 
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_render"); 
	std::string conf_projectFolder = "D:/Projects/animal_calib/";

	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config.json";
	ShapeSolver solver(pig_config); 

	// load data 
	FrameSolver framereader; 
	framereader.configByJson("D:/Projects/animal_calib/posesolver/config.json"); 
	framereader.set_frame_id(0); 
	framereader.fetchData(); 
	framereader.result_folder = "H:/pig_results_debug/";
	framereader.load_clusters(); 

	// config shapesovler
	std::vector<Camera> cameras = framereader.get_cameras(); 
	solver.setCameras(cameras); 
	solver.normalizeCamera(); 

	// init renderer
	Eigen::Matrix3f K = cameras[0].K;
	K.row(0) = K.row(0) / 1920.f;
	K.row(1) = K.row(1) / 1080.f;
	Renderer::s_Init(true);
	Renderer m_renderer("D:/Projects/animal_calib/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f));

	framereader.mp_renderEngine = &m_renderer; 
	solver.mp_renderEngine = &m_renderer; 

	readObs(solver.obs, "pig0.json", 0); 

	solver.InitNodeAndWarpField(); 
	//Mesh gthull_vec;
	//gthull_vec.Load("H:/pig_results_shape/tmp.obj");
	//std::shared_ptr<MeshEigen> p_gthull = std::make_shared<MeshEigen>(gthull_vec); 
	//solver.setTargetModel(p_gthull); 
	solver.totalSolveProcedure(); 
	solver.SaveWarpField("wrapfield0.txt"); 

	solver.SaveObj("data/deformed.obj"); 


	RenderObjectColor* p_model = new RenderObjectColor();
	solver.UpdateNormalFinal(); 
	p_model->SetVertices(solver.GetVertices());
	p_model->SetNormal(solver.GetNormals());
	p_model->SetFaces(solver.GetFacesVert());
	p_model->SetColor(CM[0]);
	
	m_renderer.clearAllObjs(); 
	m_renderer.colorObjs.push_back(p_model);


	/// draw results 
	std::vector<cv::Mat> rawImgs = framereader.get_imgs_undist();
	cv::Mat pack_raw;
	packImgBlock(rawImgs, pack_raw);

	std::vector<cv::Mat> all_renders(cameras.size());
	for (int camid = 0; camid < cameras.size(); camid++)
	{
		m_renderer.s_camViewer.SetExtrinsic(cameras[camid].R, cameras[camid].T);
		m_renderer.Draw();
		cv::Mat img = m_renderer.GetImage();

		all_renders[camid] = img;
	}

	cv::Mat packed_render;
	packImgBlock(all_renders, packed_render);

	cv::Mat blend;
	overlay_render_on_raw_gpu(packed_render, pack_raw, blend);

	cv::imwrite("H:/pig_results_shape/rendertest.png", blend);

	solver.ResetPose();
	solver.ResetTranslation();
	solver.m_scale = 1;
	solver.UpdateVertices();
	solver.SaveObj("data/deformed_tpose.obj");

	//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	//while (!glfwWindowShouldClose(windowPtr))
	//{
	//	m_renderer.Draw();

	//	glfwSwapBuffers(windowPtr);
	//	glfwPollEvents();
	//};

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

	//solver.m_bone_extend[1](2) = -0.02;
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

void test_shapemodel()
{
	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config.json";
	PigModel solver(pig_config);

	solver.InitNodeAndWarpField(); 
	solver.LoadWarpField("warpfield0.txt"); 
	solver.UpdateVertices(); 
	solver.SaveObj("data/initialscale.obj"); 
}

// 2020 10 20 
// manual enlongate head 
void test_modify_head()
{
	std::string pig_config = "D:/Projects/animal_calib/shapesolver/artist_shape_config.json";
	PigSolver solver(pig_config);
	auto bodyparts = solver.GetBodyPart(); 
	auto sym = solver.m_symIdx;
	auto jreg = solver.GetJRegressor(); 
	auto vertices_raw = solver.GetVertices(); 

	Mesh manual;
	manual.Load("D:/Projects/animal_calib/shapesolver/data2/manual_artist2.obj"); 
	int N = manual.vertex_num; 

	Eigen::MatrixXf v_eigen = Eigen::MatrixXf::Zero(3, N); 
	for (int i = 0; i < N; i++)v_eigen.col(i) = manual.vertices_vec[i];

	std::cout << "jreg row: " << jreg.rows() << ", col: " << jreg.cols() << std::endl; 
	Eigen::MatrixXf joints = v_eigen * jreg; 
	Eigen::Vector3f center = joints.col(0);
	Eigen::MatrixXf v_eigen2 = v_eigen.colwise() - center; 
	//Eigen::Matrix3f R = EulerToRotRad(Eigen::Vector3f(M_PI / 2, 0, M_PI / 2));
	//v_eigen2 = R * v_eigen2; 

	std::vector<Eigen::Vector3f> vertices_mod(N, Eigen::Vector3f::Zero()); 
	for (int i = 0; i < N; i++)
	{
		int symid = sym[i][0];
		vertices_mod[i] = v_eigen2.col(i); 
		Eigen::Vector3f v_flip = v_eigen2.col(symid); 
		v_flip(1) *= -1; 
		vertices_mod[i] += v_flip;
		vertices_mod[i] /= 2; 
	}

	auto faces = solver.GetFacesVert(); 
	double sum0 = 0;
	double sum1 = 0; 
	for (int f = 0; f < faces.cols(); f++)
	{
		Eigen::Vector3u F = faces.col(f); 
		sum0 += (vertices_raw.col(F(0)) - vertices_raw.col(F(1))).norm();
		sum1 += (vertices_mod[F(0)] - vertices_mod[F(1)]).norm();
	}
	float scale = sum0 / sum1;
	for (int i = 0; i < N; i++) {
		vertices_mod[i] *= scale;
		v_eigen2.col(i) = vertices_mod[i];
	}

	manual.vertices_vec = vertices_mod; 

	manual.Save("D:/Projects/animal_calib/shapesolver/data2/manual_artist_sym.obj");

	joints = v_eigen2 * jreg; 
	Eigen::MatrixXf sym_joints = joints; 
	std::vector<int> flip_index = {
	0,1,2,3,4,13,14,15,16,17,18,19,20,5,6,7,8,9,10,11,12,21,22,23,25,24,31,32,33,34,35,
	26,27,28,29,30,36,37,54,55,56,57,58,59,60,61,46,47,48,49,50,51,52,53,
	38,39,40,41,42,43,44,45
	};
	for (int i = 0; i < sym_joints.cols(); i++)
	{
		Eigen::Vector3f flip = joints.col(flip_index[i]); 
		flip(1) *= -1;
		sym_joints.col(i) += flip;
	}
	sym_joints /= 2; 

	std::ofstream outjointfile("D:/Projects/animal_calib/data/artist_model_sym3/t_pose_joints.txt");
	outjointfile << sym_joints.transpose();
	outjointfile.close(); 

	std::ofstream outverticesfile("D:/Projects/animal_calib/data/artist_model_sym3/vertices.txt");
	outverticesfile << v_eigen2.transpose();
	outverticesfile.close(); 
}

int main()
{
	//solve_shape(); 
	//test_shapemodel(); 
	//test_bone_var();
	generate_nodegraph(); 

	return 0; 
}