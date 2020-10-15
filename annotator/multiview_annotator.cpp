#include "main.h"
#include <fstream>
#include <string>
#include <vector>
#include <iostream> 
#include <vector_functions.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nanogui/nanogui.h>
#include "../utils/mesh.h"
#include "../utils/image_utils.h"
#include "../utils/camera.h"
#include "../utils/geometry.h"
#include "../utils/math_utils.h"
#include "NanoRenderer.h"
#include "../articulation/pigmodeldevice.h"
#include "../posesolver/framesolver.h"
#include "../utils/image_utils_gpu.h"

int multiview_annotator()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/render";

	std::vector<float4> colormap = getColorMapFloat4("anliang_render");
	std::vector<Eigen::Vector3i> colormapeigen = getColorMapEigen("anliang_render");
    /// read frame data 
	FrameSolver data_loader; 
	std::string data_config = "D:/Projects/animal_calib/posesolver/config.json"; 
	data_loader.configByJson(data_config); 
	data_loader.set_frame_id(0); 
	data_loader.fetchData(); 
	data_loader.result_folder = "H:/pig_results_debug/"; 
	data_loader.load_clusters();
	data_loader.read_parametric_data(); 

	auto solvers = data_loader.mp_bodysolverdevice;

	/// read smal model 
	Mesh obj;
	obj.Load("D:/Projects/animal_calib/shapesolver/model.obj");
	MeshFloat4 objfloat4(obj);

	std::vector<Camera> cams = readCameras();
	Camera cam = cams[0];

	NanoRenderer renderer;
	renderer.Init(1920, 1080, cam.K(0, 0), cam.K(1, 1), cam.K(0, 2), cam.K(1, 2), 0);

	Eigen::Matrix4f view_eigen = calcRenderExt(cam.R, cam.T);
	nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
	renderer.UpdateCanvasView(view_eigen);

	std::vector<float4> colors_float4(obj.vertex_num, colormap[0]);
	renderer.ClearRenderObjects();
	auto smal_model = renderer.CreateRenderObject("smal_model", vs_phong_vertex_color, fs_phong_vertex_color);
	smal_model->SetIndices(objfloat4.indices);
	smal_model->SetBuffer("positions", objfloat4.vertices);
	smal_model->SetBuffer("normals", objfloat4.normals);
	smal_model->SetBuffer("incolor", colors_float4);

	std::string pig_conf = "D:/Projects/animal_calib/articulation/artist_config.json";
	PigModelDevice pigmodel(pig_conf);
	pigmodel.SetPose(solvers[0]->GetPose());
	pigmodel.SetScale(solvers[0]->GetScale());
	pigmodel.SetTranslation(solvers[0]->GetTranslation()); 
	renderer.set_joint_pose(pigmodel.GetPose()); 

	std::vector<cv::Mat> rawImgs = data_loader.get_imgs_undist(); 
	
	//auto ball_model = renderer.CreateRenderObject("ball_model", vs_phong_vertex_color, fs_phong_vertex_color);
	//Mesh ballobj;
	//ballobj.Load("D:/Projects/animal_calib/render/data/obj_model/ball.obj");
	//std::vector<float4> colors_ball(ballobj.vertex_num, colormap[1]);
	//MeshFloat4 ballfloat4(ballobj);
	//ball_model->SetIndices(ballfloat4.indices);
	//ball_model->SetBuffer("positions", ballfloat4.vertices);
	//ball_model->SetBuffer("normals", ballfloat4.normals);
	//ball_model->SetBuffer("incolor", colors_ball);

	auto box3_offscreen = renderer.CreateOffscreenRenderObject(
		"box3", vs_phong_vertex_color, fs_phong_vertex_color, 1920, 1080, cam.K(0, 0), cam.K(1, 1), cam.K(0, 2), cam.K(1, 2), 1, false, false);
	box3_offscreen->SetIndices(objfloat4.indices);
	box3_offscreen->SetBuffer("positions", objfloat4.vertices);
	box3_offscreen->SetBuffer("normals", objfloat4.normals);
	box3_offscreen->SetBuffer("incolor", colors_float4);

	cv::Mat rendered_img(1920, 1080, CV_8UC4);
	std::vector<cv::Mat> rendered_imgs;
	rendered_imgs.push_back(rendered_img);

	renderer.CreateRenderImage("Overlay", Vector2i(960, 540), Vector2i(0, 0));


	int frameIdx = 0;
	while (!renderer.ShouldClose())
	{
		pigmodel.SetPose(renderer.m_joint_pose);
		pigmodel.UpdateVertices();
		pigmodel.UpdateNormalFinal();
		obj.vertices_vec = pigmodel.GetVertices();
		obj.normals_vec = pigmodel.GetNormals();
		objfloat4.LoadFromMesh(obj);
		smal_model->SetBuffer("positions", objfloat4.vertices);
		smal_model->SetBuffer("normals", objfloat4.normals);

		box3_offscreen->_SetViewByCameraRT(cam.R, cam.T);
		box3_offscreen->SetBuffer("positions", objfloat4.vertices);
		box3_offscreen->SetBuffer("normals", objfloat4.normals);
		box3_offscreen->DrawOffscreen();
		box3_offscreen->DownloadRenderingResults(rendered_imgs);
		
		cv::Mat raw_img_small;
		cv::resize(rawImgs[0], raw_img_small, cv::Size(960, 540)); 
		cv::Mat out; 
		cv::Mat render; 
		cv::cvtColor(rendered_imgs[0], render, cv::COLOR_BGRA2BGR);
		cv::resize(render, render, cv::Size(960, 540));
		overlay_render_on_raw_gpu(render, raw_img_small, out);
		cv::cvtColor(out, out, cv::COLOR_BGR2BGRA);
		
		renderer.SetRenderImage("Overlay", out);

		
		renderer.Draw();
		++frameIdx;
	}
	cv::destroyAllWindows();

	return 0;
}
