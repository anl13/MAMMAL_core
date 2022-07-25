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
#include "anno_utils.h"

Eigen::Vector2f pack_backward(Eigen::Vector2f p, int& camid, int W = 256, int H = 256)
{
	int row = (int)p(1) / H; 
	int col = (int)p(0) / W; 
	Eigen::Vector2f out; 
	out(0) = p(0) - col * W; 
	out(1) = p(1) - row * H; 
	camid = row * 4 + col; 
	return out; 
}

Eigen::Vector2f padding_back(cv::Rect roi, Eigen::Vector2f p, int W = 256, int H = 256)
{
	int x = roi.x;
	int y = roi.y;
	int w = roi.width; 
	int h = roi.height; 
	float x_f = p(0) / W - 0.5; 
	float y_f = p(1) / H - 0.5;
	float centerx = x + (float)w / 2; 
	float centery = y + (float)h / 2; 
	float scale = my_max(w, h);
	Eigen::Vector2f out; 
	out(0) = centerx + scale * x_f; 
	out(1) = centery + scale * y_f; 
	return out; 
}

Eigen::Vector3f clicked_point; 

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	Eigen::Vector3f empty = { -1,-1 };
	Eigen::Vector3f* data = (Eigen::Vector3f*) userdata;
	int length = data->size();
	int cols = sqrt(length);
	if (cols * cols < length) cols += 1;
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		data->x() = x; 
		data->y() = y;
		data->z() = 1;  // left button 
		std::cout << "click on point (" << x << " , " << y << ")" << std::endl;
	}
	else if (event == cv::EVENT_RBUTTONDOWN)
	{
		data->x() = x; 
		data->y() = y; 
		data->z() = 2; // right button 
		std::cout << "cancel on point (" << x << " , " << y << ")" << std::endl;
	}
	else if (event == cv::EVENT_MBUTTONDOWN)
	{
		std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
	}
}

void readStateFile(std::string state_file, Eigen::Vector3f& translation, float& scale, std::vector<Eigen::Vector3f>& pose)
{
	pose.resize(62, Eigen::Vector3f::Zero());
	std::ifstream is(state_file);
	if (!is.is_open())
	{
		std::cout << "cant not open " << state_file << std::endl;
		return;
	}
	for (int i = 0; i < 3; i++) is >> translation(i);
	for (int i = 0; i < 62; i++)
		for (int k = 0; k < 3; k++)
			is >> pose[i](k);
	is >> scale;
	is.close();
}

int multiview_annotator()
{
	AnnoConfig config;
	std::string projectParentDir = config.project_dir;
	std::string projectDir = projectParentDir + "/MAMMAL_core/"; 
	std::vector<float4> colormap = getColorMapFloat4("anliang_render");
	std::vector<Eigen::Vector3i> colormapeigen = getColorMapEigen("anliang_render");
    /// read frame data 
	FrameSolver data_loader; 
	std::string data_config = projectDir + config.posesolver_config; 
	data_loader.configByJson(data_config); 
	data_loader.set_frame_id(config.current_frame_id); 
	data_loader.fetchData(); 
	//data_loader.fetchGtData(); 
	data_loader.is_smth = false;
	data_loader.load_clusters();
	data_loader.read_parametric_data(); 

	std::cout << "intrinsic type: " << data_loader.m_intrinsic_type << std::endl; 

	auto solvers = data_loader.mp_bodysolverdevice;

	/// read smal model 
	Mesh obj;
	obj.Load(projectDir + "data/artist_model_sym3/manual_artist_sym.obj");
	MeshFloat4 objfloat4(obj);

	std::vector<Camera> cams = data_loader.get_cameras();
	Camera cam = cams[0];
	int camNum = cams.size();

	NanoRenderer renderer;
	renderer.out_frameid = config.current_frame_id;
	renderer.m_pig_num = data_loader.m_pignum;
	renderer.Init(1920, 1080, cam.K(0, 0), cam.K(1, 1), cam.K(0, 2), cam.K(1, 2), 0, false, data_loader.m_annotation_folder);
	std::cout << "renderer init. " << std::endl; 

	//Eigen::Matrix4f view_eigen = calcRenderExt(cam.R, cam.T);
	//nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
	//renderer.UpdateCanvasView(view_eigen);
	Eigen::Vector3f pos(0, -2, 0);
	Eigen::Vector3f up(0, 0, 1);
	Eigen::Vector3f center(0, 0, 0);
	renderer.SetCanvasExtrinsic(pos, up, center); 

	std::vector<float4> colors_float4(obj.vertex_num, colormap[0]);
	renderer.ClearRenderObjects();
	auto smal_model = renderer.CreateRenderObject("smal_model", vs_phong_vertex_color, fs_phong_vertex_color);
	smal_model->SetIndices(objfloat4.indices);
	smal_model->SetBuffer("positions", objfloat4.vertices);
	smal_model->SetBuffer("normals", objfloat4.normals);
	smal_model->SetBuffer("incolor", colors_float4);

	std::string pig_conf = projectDir + "articulation/artist_config_sym.json";
	PigModelDevice pigmodel(pig_conf);
	pigmodel.SetPose(solvers[0]->GetPose());
	pigmodel.SetScale(solvers[0]->GetScale());
	pigmodel.SetTranslation(solvers[0]->GetTranslation()); 
	renderer.set_joint_pose(pigmodel.GetPose()); 
	renderer.set_pig_scale(pigmodel.GetScale()); 
	renderer.set_pig_translation(pigmodel.GetTranslation()); 

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

	// scene
	// -- part1
	Mesh planeobj;
	planeobj.Load( projectDir + "data/calibdata/scene_model/manual_scene_part0.obj");
	MeshFloat4 boxfloat4(planeobj);
	// create a renderObject, you need to specify which shader you want to use, and set corresponding buffers in the shader
	// by using the straightforward interfaces provided. 
	ref<RenderObject> box1 = renderer.CreateRenderObject("plane", vs_texture_object, fs_texture_object);
	box1->SetIndices(boxfloat4.indices);
	box1->SetBuffer("positions", boxfloat4.vertices);
	box1->SetTexCoords(boxfloat4.textures);
	cv::Mat tex0Image = cv::imread(projectDir + "render/data/chessboard_black.png", cv::IMREAD_UNCHANGED);
	cv::cvtColor(tex0Image, tex0Image, cv::COLOR_BGRA2RGBA);
	box1->SetTexture(
		"tex0",
		nanogui::Texture::PixelFormat::RGBA,
		nanogui::Texture::ComponentFormat::UInt8,
		nanogui::Vector2i(tex0Image.cols, tex0Image.rows), tex0Image.data);
	// -- part2
	Mesh obj2;
	obj2.Load(projectDir + "data/calibdata/scene_model/manual_scene_part1.obj");
	MeshFloat4 obj2float4(obj2);
	auto part1_model = renderer.CreateRenderObject("part1", vs_phong_vertex_color, fs_phong_vertex_color);
	part1_model->SetIndices(obj2float4.indices);
	part1_model->SetBuffer("positions", obj2float4.vertices);
	part1_model->SetBuffer("normals", obj2float4.normals);
	float4 part1_color = make_float4(0.9, 0.9, 0.95, 1);
	std::vector<float4> part1_colors(obj2.vertex_num,part1_color);
	part1_model->SetBuffer("incolor", part1_colors);
	// -- part3
	Mesh obj3;
	obj3.Load(projectDir + "data/calibdata/scene_model/manual_scene_part2.obj");
	MeshFloat4 obj3float4(obj3);
	auto part2_model = renderer.CreateRenderObject("part1", vs_phong_vertex_color, fs_phong_vertex_color);
	part2_model->SetIndices(obj3float4.indices);
	part2_model->SetBuffer("positions", obj3float4.vertices);
	part2_model->SetBuffer("normals", obj3float4.normals);
	float4 part2_color = make_float4(0.9, 0.9, 0.95, 1);
	std::vector<float4> part2_colors(obj3.vertex_num, part2_color);
	part2_model->SetBuffer("incolor", part2_colors);

	cv::Mat rendered_img(1920, 1080, CV_8UC4);
	std::vector<cv::Mat> rendered_imgs;
	rendered_imgs.push_back(rendered_img);

	//renderer.CreateRenderImage("Overlay", Vector2i(1024, 768), Vector2i(0, 0));

	int current_frame_id = config.current_frame_id;
	int current_pig_id = config.current_pig_id; 
	std::cout << "start annotator. " << std::endl; 
	cv::namedWindow("Overlay", cv::WINDOW_NORMAL);
	cv::setMouseCallback("Overlay", CallBackFunc, &clicked_point);

	while (!renderer.ShouldClose())
	{
		pigmodel.SetPose(renderer.m_joint_pose);
		pigmodel.SetScale(renderer.m_pig_scale); 
		pigmodel.SetTranslation(renderer.m_pig_translation); 

		// handling internal state 
		bool m_state_read = renderer.m_state_read; 
		bool m_state_save_obj = renderer.m_state_save_obj; 
		bool m_state_save_state = renderer.m_state_save_state; 
		bool m_state_load_last = renderer.m_state_load_last; 

		if (m_state_read)
		{
			current_pig_id = renderer.enumval;
			current_frame_id = renderer.out_frameid; 
			if (current_frame_id != data_loader.m_frameid)
			{
				data_loader.set_frame_id(current_frame_id);
				data_loader.fetchData();
				//data_loader.fetchGtData(); 
				data_loader.load_clusters();
				data_loader.read_parametric_data();
				rawImgs = data_loader.get_imgs_undist();

			}
			pigmodel.SetPose(solvers[current_pig_id]->GetPose());
			pigmodel.SetScale(solvers[current_pig_id]->GetScale());
			pigmodel.SetTranslation(solvers[current_pig_id]->GetTranslation());
			renderer.set_joint_pose(pigmodel.GetPose());
			renderer.set_pig_scale(pigmodel.GetScale());
			renderer.set_pig_translation(pigmodel.GetTranslation());
			renderer.m_state_read = false; 
		}

		if (m_state_load_last)
		{
			std::stringstream ss;
			ss << renderer.m_results_folder << "/pig_" << int(renderer.enumval) << "_frame_" << std::setw(6) << std::setfill('0') << renderer.out_frameid - 1 << ".txt";
			pigmodel.readState(ss.str());
			renderer.m_state_load_last = false; 
			renderer.set_joint_pose(pigmodel.GetPose());
			renderer.set_pig_scale(pigmodel.GetScale());
			renderer.set_pig_translation(pigmodel.GetTranslation());
		}

		if (renderer.m_state_load_this)
		{
			std::stringstream ss;
			ss << renderer.m_results_folder << "/pig_" << int(renderer.enumval) << "_frame_" << std::setw(6) << std::setfill('0') << renderer.out_frameid << ".txt";
			pigmodel.readState(ss.str());
			renderer.set_joint_pose(pigmodel.GetPose());
			renderer.set_pig_scale(pigmodel.GetScale());
			renderer.set_pig_translation(pigmodel.GetTranslation());
			renderer.m_state_load_this = false; 
		}

		if (renderer.m_state_load_next)
		{
			std::stringstream ss;
			ss << renderer.m_results_folder << "/pig_" << int(renderer.enumval) << "_frame_" << std::setw(6) << std::setfill('0') << renderer.out_frameid + 1 << ".txt";
			pigmodel.readState(ss.str());
			renderer.set_joint_pose(pigmodel.GetPose());
			renderer.set_pig_scale(pigmodel.GetScale());
			renderer.set_pig_translation(pigmodel.GetTranslation());
			renderer.m_state_load_next = false;
		}

		pigmodel.UpdateVertices();
		pigmodel.UpdateNormalFinal();

		if (m_state_save_obj)
		{
			std::stringstream ss; 
			ss << renderer.m_results_folder << "/pig_" << int(renderer.enumval) << "_frame_" << std::setw(6) << std::setfill('0') << renderer.out_frameid << ".obj"; 
			pigmodel.saveObj(ss.str()); 
			renderer.m_state_save_obj = false; 
		}

		if (m_state_save_state)
		{
			std::stringstream ss; 
			ss << renderer.m_results_folder << "/pig_" << int(renderer.enumval) << "_frame_" << std::setw(6) << std::setfill('0') << renderer.out_frameid << ".txt";
			pigmodel.saveState(ss.str());
			renderer.m_state_save_state = false;
		}

		obj.vertices_vec = pigmodel.GetVertices();
		obj.normals_vec = pigmodel.GetNormals();
		objfloat4.LoadFromMesh(obj);
		smal_model->SetBuffer("positions", objfloat4.vertices);
		smal_model->SetBuffer("normals", objfloat4.normals);
		
		std::vector<cv::Mat> render_views(12, cv::Mat(cv::Size(256,256),CV_8UC3)); 
		for (int i = 0; i < render_views.size(); i++)
		{
			render_views[i].setTo(cv::Scalar(228, 228, 240)); 
		}

		float alpha = renderer.m_overlay_transparency; 
		if (alpha < 0) alpha = 0; 
		if (alpha > 100) alpha = 100;
		alpha = alpha / 100;

		std::vector<cv::Rect> all_rois;
		all_rois.clear(); 
		all_rois.resize(camNum); 
		for (int i = 0; i < data_loader.m_matched[current_pig_id].view_ids.size(); i++)
		{
			
			int camid = data_loader.m_matched[current_pig_id].view_ids[i];
			cv::Rect roi = expand_box(data_loader.m_matched[current_pig_id].dets[i]);
			box3_offscreen->_SetViewByCameraRT(cams[camid].R, cams[camid].T);
			box3_offscreen->SetBuffer("positions", objfloat4.vertices);
			box3_offscreen->SetBuffer("normals", objfloat4.normals);
			box3_offscreen->DrawOffscreen();
			box3_offscreen->DownloadRenderingResults(rendered_imgs);

			all_rois[camid] = roi;

			//cv::Mat raw_img_small;
			//cv::resize(rawImgs[camid], raw_img_small, cv::Size(1920, 1080));
			cv::Mat render;
			cv::cvtColor(rendered_imgs[0], render, cv::COLOR_BGRA2BGR);
			//cv::resize(render, render, cv::Size(1920, 1080));
			cv::Mat render_roi = render(roi);
			cv::Mat raw_roi = rawImgs[camid](roi);

			cv::cvtColor(render_roi, render_roi, cv::COLOR_BGR2RGB); 
			cv::Mat out = overlay_renders(raw_roi, render_roi, alpha);
			out = resizeAndPadding(out, 256, 256);
			
			render_views[camid] = out; 
		}
		for (int camid = 0; camid < camNum; camid++)
		{
			if (in_list(camid, data_loader.m_matched[current_pig_id].view_ids))continue;
			Eigen::Vector3f center = pigmodel.getRegressedSkel_host()[20];
			Eigen::Vector3f center_2d;
			project(cams[camid], center, center_2d);
			cv::Rect roi;
			float left = std::fminf(std::fmaxf(center_2d[0] - 256, 0), 1079);
			float right = std::fmaxf(std::fminf(center_2d[0] + 256, 1919), 0); 
			float top = std::fminf(std::fmaxf(center_2d[1] - 256, 0), 1079); 
			float bottom = std::fmaxf(std::fminf(center_2d[1] + 256, 1079), 0);
			roi.x = int(left); 
			roi.y = int(top); 
			roi.width = int(right - left); 
			roi.height = int(bottom - top); 
			all_rois[camid] = roi;
			if (roi.width == 0 || roi.height == 0)continue;
			box3_offscreen->_SetViewByCameraRT(cams[camid].R, cams[camid].T);
			box3_offscreen->SetBuffer("positions", objfloat4.vertices);
			box3_offscreen->SetBuffer("normals", objfloat4.normals);
			box3_offscreen->DrawOffscreen();
			box3_offscreen->DownloadRenderingResults(rendered_imgs);

			cv::Mat render;
			cv::cvtColor(rendered_imgs[0], render, cv::COLOR_BGRA2BGR);
			cv::Mat render_roi = render(roi);
			cv::Mat raw_roi = rawImgs[camid](roi);
			cv::cvtColor(render_roi, render_roi, cv::COLOR_BGR2RGB);
			cv::Mat out = overlay_renders(raw_roi, render_roi, alpha);
			out = resizeAndPadding(out, 256, 256);

			render_views[camid] = out;
		}
		cv::Mat packedRender; 
		packImgBlock(render_views, packedRender); 
		cv::cvtColor(packedRender, packedRender, cv::COLOR_BGR2BGRA);
		//renderer.SetRenderImage("Overlay", packedRender);
		cv::imshow("Overlay", packedRender); 
		cv::waitKey(1); 
		
		renderer.Draw();
	}
	cv::destroyAllWindows();

	return 0;
}
