#include "main.h"

//#include "annotator.h"
// #include "state_annotator.h"
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

//void annotate()
//{
//	std::string folder = "D:/Projects/animal_calib/data/pig_model_noeye/";
//	std::string pig_config = "D:/Projects/animal_calib/smal/pigmodel_config.json";
//	std::string conf_projectFolder = "D:/Projects/animal_calib/";
//
//	SkelTopology topo = getSkelTopoByType("UNIV");
//	FrameData frame;
//	frame.configByJson(conf_projectFolder + "/associate/config.json");
//	int startid = frame.get_start_id();
//	int framenum = frame.get_frame_num();
//	int frameid = 0;
//
//
//	frame.set_frame_id(frameid);
//	frame.fetchData();
//	//frame.matching_by_tracking();
//	frame.load_labeled_data();
//
//	std::string result_folder = "E:/my_labels/";
//	Annotator A;
//	A.result_folder = result_folder;
//	A.frameid = frameid;
//	A.m_cams = frame.get_cameras();
//	A.m_camNum = A.m_cams.size();
//	A.setInitData(frame.get_matched());
//	A.m_imgs = frame.get_imgs_undist();
//	A.m_unmatched = frame.get_unmatched();
//
//	A.show_panel();
//}


// int main()
// {
// 	StateAnnotator A;
// 	A.show_panel(); 

// 	return 0; 
// }

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

nanogui::Matrix4f eigen2nanoM4f(Eigen::Matrix4f mat)
{
	nanogui::Matrix4f F; 
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			F.m[i][j] = mat(j, i); 
		}
	}
	return F; 
}

cv::Rect expand_box(const DetInstance& det )
{
	cv::Rect roi; 
	float minx = det.box(0);
	float maxx = det.box(2); 
	float miny = det.box(1); 
	float maxy = det.box(3); 
	for (int i = 0; i < det.keypoints.size(); i++)
	{
		if (det.keypoints[i](2) < 0.5)continue; 
		minx = std::fminf(det.keypoints[i](0), minx); 
		maxx = std::fmaxf(det.keypoints[i](0), maxx);
		miny = std::fminf(det.keypoints[i](1), miny);
		maxy = std::fmaxf(det.keypoints[i](1), maxy);
	}
	if (minx < 0) minx = 0; 
	if (miny < 0) miny = 0;
	if (maxx > 1919) maxx = 1919;
	if (maxy > 1079) maxy = 1079;
	float w = maxx - minx; 
	float h = maxy - miny;
	float margin = std::fmaxf(w, h) * 0.15;
	minx -= margin;
	miny -= margin;
	maxx += margin;
	maxy += margin;
	if (minx < 0) minx = 0;
	if (miny < 0) miny = 0;
	if (maxx > 1919) maxx = 1919;
	if (maxy > 1079) maxy = 1079;

	roi.x = int(minx / 2);
	roi.y = int(miny / 2); 
	roi.width = int(maxx / 2 - minx / 2); 
	roi.height = int(maxy / 2 - miny / 2); 
	return roi;
}


int test_datatype()
{
	// Init Renderer, you can set the center of the Arcball using the laster parameter
	// The default Arcball rotation center is set to 2.0m
	auto cameras = readCameras(); 
	NanoRenderer renderer;
	renderer.Init(1024, 768, 400, 400, 0.7);

	// create a renderImage using cv::Mat, then we are ready to render this image on the screen
	renderer.CreateRenderImage("Camera0", Vector2i(200, 200), Vector2i(5, 460));
	cv::Mat img = cv::imread("D:/Projects/animal_calib/render/data/awesomeface.png", cv::IMREAD_UNCHANGED);
	//cv::cvtColor(img, img, cv::COLOR_BGRA2RGBA);
	renderer.SetRenderImage("Camera0", img);

	Mesh boxobj;
	boxobj.Load("D:/Projects/animal_calib/render/data/obj_model/cube.obj");
	MeshFloat4 boxfloat4(boxobj); 
	// create a renderObject, you need to specify which shader you want to use, and set corresponding buffers in the shader
	// by using the straightforward interfaces provided. 
	ref<RenderObject> box1 = renderer.CreateRenderObject("box1", vs_texture_object, fs_texture_object);
	box1->SetIndices(boxfloat4.indices);
	box1->SetBuffer("positions", boxfloat4.vertices);
	box1->SetTexCoords(boxfloat4.textures);
	cv::Mat tex0Image = cv::imread("D:/Projects/animal_calib/render/data/football-icon.png", cv::IMREAD_UNCHANGED);
	cv::cvtColor(tex0Image, tex0Image, cv::COLOR_BGRA2RGBA);
	box1->SetTexture(
		"tex0",
		nanogui::Texture::PixelFormat::RGBA,
		nanogui::Texture::ComponentFormat::UInt8,
		nanogui::Vector2i(tex0Image.cols, tex0Image.rows), tex0Image.data);

	Mesh model;
	model.Load("D:/Projects/animal_calib/shapesolver/model.obj");
	MeshFloat4 model_float4(model);
	auto human_model = renderer.CreateRenderObject("human_model", vs_phong_geometry, fs_phong_geometry);
	human_model->SetIndices(model_float4.indices);
	human_model->SetBuffer("positions", model_float4.vertices);
	human_model->SetBuffer("normals", model_float4.normals);
	human_model->SetModelRT(nanogui::Matrix4f::translate(nanogui::Vector3f(0, -0, -0)));

	int frameIdx = 0;
	while (!renderer.ShouldClose())
	{
		// rotate box1 along z axis
		box1->SetModelRT(Matrix4f::translate(Vector3f(-0.1, 0, 0.1)) * Matrix4f::rotate(Vector3f(0, 0, 1), glfwGetTime()));

		// draw all the renderObjects defined above
		renderer.Draw();

		while (renderer.Pause())
		{
			renderer.Draw();
		}

		//// render box3_offscreen to a cv::Mat (offscreen rendering)
		//human_offscreen->DrawOffscreen();
		//human_offscreen->DownloadRenderingResults(rendered_imgs);
		//cv::imshow("rendered img", rendered_imgs[0]);
		//cv::waitKey(1);

		++frameIdx;
	}
	return 0;
}




int test_depth()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/render";

	std::vector<float4> colormap = getColorMapFloat4("anliang_render");
	std::vector<Eigen::Vector3i> colormapeigen = getColorMapEigen("anliang_render");
	/// read smal model 
	Mesh obj;
	obj.Load("D:/Projects/animal_calib/shapesolver/model.obj");
	MeshFloat4 objfloat4(obj);
	MeshEigen objeigen(obj);

	std::vector<Camera> cams = readCameras();
	Camera cam = cams[0];

	NanoRenderer renderer;
	renderer.Init(1920, 1080, float(cam.K(0, 0)), float(cam.K(1, 1)), float(cam.K(0, 2)), float(cam.K(1, 2)),0);

	Eigen::Matrix4f view_eigen = calcRenderExt(cam.R, cam.T);
	nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
	renderer.UpdateCanvasView(view_eigen);

	//std::vector<float4> colors_float4(obj.vertex_num, colormap[0]);
	//renderer.ClearRenderObjects();
	//auto smal_model = renderer.CreateRenderObject("smal_model", vs_phong_vertex_color, fs_phong_vertex_color);
	//smal_model->SetIndices(objfloat4.indices);
	//smal_model->SetBuffer("positions", objfloat4.vertices);
	//smal_model->SetBuffer("normals", objfloat4.normals);
	//smal_model->SetBuffer("incolor", colors_float4);

	auto ball_model = renderer.CreateRenderObject("ball_model", vs_phong_vertex_color, fs_phong_vertex_color);
	Mesh ballobj;
	ballobj.Load("D:/Projects/animal_calib/render/data/obj_model/ball.obj");
	std::vector<float4> colors_ball(ballobj.vertex_num, colormap[1]);
	MeshFloat4 ballfloat4(ballobj); 
	ball_model->SetIndices(ballfloat4.indices);
	ball_model->SetBuffer("positions", ballfloat4.vertices); 
	ball_model->SetBuffer("normals", ballfloat4.normals); 
	ball_model->SetBuffer("incolor", colors_ball); 

	int frameIdx = 0;
	while (!renderer.ShouldClose())
	{
		renderer.Draw();
		++frameIdx;
	}

	return 0;
}



