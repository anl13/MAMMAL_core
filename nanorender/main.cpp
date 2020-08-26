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

int test_datatype()
{
	
	std::vector<float4> vertices;
	vertices.push_back(make_float4(-1.f, 1.f, 6.f, 1.f));
	vertices.push_back(make_float4(-1.f, -1.f, 6.f, 1.f));
	vertices.push_back(make_float4(1.f, -1.f, 6.f, 1.f));
	vertices.push_back(make_float4(1.f, 1.f, 6.f, 1.f));
	vertices.push_back(make_float4(-1.f, 1.f, 4.f, 1.f));
	vertices.push_back(make_float4(-1.f, -1.f, 4.f, 1.f));
	vertices.push_back(make_float4(1.f, -1.f, 4.f, 1.f));
	vertices.push_back(make_float4(1.f, 1.f, 4.f, 1.f));
	for (int i = 0; i < vertices.size(); ++i)
	{
		vertices[i].x *= 0.05f;
		vertices[i].y *= 0.05f;
		vertices[i].z *= 0.05f;
	}

	pcl::gpu::DeviceArray<float4> vertices_device;
	vertices_device.upload(vertices);

	// init render objects
	std::vector<unsigned int> indices = {
		3, 2, 6, 6, 7, 3,
		4, 5, 1, 1, 0, 4,
		4, 0, 3, 3, 7, 4,
		1, 5, 6, 6, 2, 1,
		0, 1, 2, 2, 3, 0,
		7, 6, 5, 5, 4, 7
	};

	std::vector<float4> colors;
	colors.push_back(make_float4(0, 1, 1, 1));
	colors.push_back(make_float4(0, 0, 1, 1));
	colors.push_back(make_float4(1, 0, 1, 1));
	colors.push_back(make_float4(1, 1, 1, 1));
	colors.push_back(make_float4(0, 1, 0, 1));
	colors.push_back(make_float4(0, 0, 0, 1));
	colors.push_back(make_float4(1, 0, 0, 1));
	colors.push_back(make_float4(1, 1, 0, 1));

	std::vector<float2> texCoords;
	texCoords.push_back(make_float2(0, 0));
	texCoords.push_back(make_float2(1, 0));
	texCoords.push_back(make_float2(1, 1));
	texCoords.push_back(make_float2(0, 1));
	texCoords.push_back(make_float2(1, 0));
	texCoords.push_back(make_float2(0, 0));
	texCoords.push_back(make_float2(0, 1));
	texCoords.push_back(make_float2(1, 1));

	// Init Renderer, you can set the center of the Arcball using the laster parameter
	// The default Arcball rotation center is set to 2.0m
	NanoRenderer renderer;
	renderer.Init(1024, 768, 400, 400, 0.7);

	// create a renderImage using cv::Mat, then we are ready to render this image on the screen
	renderer.CreateRenderImage("Camera0", Vector2i(200, 200), Vector2i(5, 460));
	cv::Mat img = cv::imread("D:/Projects/animal_calib/nanorender/data/awesomeface.png", cv::IMREAD_UNCHANGED);
	//cv::cvtColor(img, img, cv::COLOR_BGRA2RGBA);
	renderer.SetRenderImage("Camera0", img);

	// create a renderObject, you need to specify which shader you want to use, and set corresponding buffers in the shader
	// by using the straightforward interfaces provided. 
	ref<RenderObject> box1 = renderer.CreateRenderObject("box1", vs_texture_object, fs_texture_object);
	box1->SetIndices(indices);
	// SetBuffer Option1: set vertices from host std::vector
	box1->SetBuffer("positions", vertices);
	//// SetBuffer Option2: set vertices from pcl::DeviceArray, will call map and unmap to eliminate host-device mem copies
	//box1->SetBuffer("positions", vertices_device); 
	// SetBuffer Option3: you can use the buffer of a previously defined renderObject
	// to init the corresponding buffer of a new renderObject
	// This will share the buffers between renderObjects without copying the same buffer several times. 
	// This is quite useful when you want to render a specific object using different types of shaders.
	box1->SetTexCoords(texCoords);
	cv::Mat tex0Image = cv::imread("D:/Projects/animal_calib/nanorender/data/football-icon.png", cv::IMREAD_UNCHANGED);
	cv::cvtColor(tex0Image, tex0Image, cv::COLOR_BGRA2RGBA);
	box1->SetTexture(
		"tex0",
		nanogui::Texture::PixelFormat::RGBA,
		nanogui::Texture::ComponentFormat::UInt8,
		nanogui::Vector2i(tex0Image.cols, tex0Image.rows), tex0Image.data);

	Mesh model; 
	model.Load("D:/Projects/animal_calib/nanorender/data/model.obj");
	MeshFloat4 model_float4; 
	model.GetMeshFloat4(model_float4); 
	auto human_model = renderer.CreateRenderObject("human_model", vs_phong_geometry, fs_phong_geometry);
	human_model->SetIndices(model_float4.indices);
	human_model->SetBuffer("positions", model_float4.vertices);
	human_model->SetBuffer("normals", model_float4.normals);
	human_model->SetModelRT(nanogui::Matrix4f::translate(nanogui::Vector3f(-0.5, -0.5, 0.2)));

	// create offscreen render object, you can render this object to a cuda texture or a cv::Mat
	// In this example code, I render this object to a cv::Mat and then use cv::imshow to demonstrate the rendering results
	// See interfaces of OffScreenRenderObject for more details
	auto human_offscreen = renderer.CreateOffscreenRenderObject(
		"box3", vs_vertex_position, fs_vertex_position, 400, 400, 512, 384, 1, true, true);
	human_offscreen->SetIndices(human_model);
	human_offscreen->SetBuffer("positions", human_model);
	human_offscreen->SetModelRT(nanogui::Matrix4f::translate(nanogui::Vector3f(-0.5, -0.5, 0.2)));

	cv::Mat rendered_img(800, 800, CV_32FC4);
	std::vector<cv::Mat> rendered_imgs;
	rendered_imgs.push_back(rendered_img);

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

		// render box3_offscreen to a cv::Mat (offscreen rendering)
		human_offscreen->DrawOffscreen();
		human_offscreen->DownloadRenderingResults(rendered_imgs);
		cv::imshow("rendered img", rendered_imgs[0]);
		cv::waitKey(1);

		++frameIdx;
	}
	return 0;
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
				CM[i](0)/255.f, CM[i](1)/255.f, CM[i](2)/255.f, 1.0f);
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


int test_depth()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/render";
	
	std::vector<float4> colormap = getColorMapFloat4("anliang_rgb");
	std::vector<Eigen::Vector3i> colormapeigen = getColorMapEigen("anliang_rgb");
	/// read smal model 
	Mesh obj; 
	obj.Load("F:/projects/model_preprocess/designed_pig/extracted/artist_model/model_triangle.obj");
	MeshFloat4 objfloat4; 
	obj.GetMeshFloat4(objfloat4); 
	MeshEigen objeigen; 
	obj.GetMeshEigen(objeigen); 

	std::vector<Camera> cams = readCameras(); 
	Camera cam = cams[0]; 

	NanoRenderer renderer;
	renderer.Init(1920, 1080, float(cam.K(0, 0)), float(cam.K(1, 1)), float(cam.K(0, 2)), float(cam.K(1, 2)), -2.0f);

	auto human_model = renderer.CreateRenderObject("human_model", vs_phong_geometry, fs_phong_geometry);
	human_model->SetIndices(objfloat4.indices);
	human_model->SetBuffer("positions", objfloat4.vertices);
	human_model->SetBuffer("normals", objfloat4.normals);
	//nanogui::Vector4f color;
	//color[0] = 1.0f; color[1] = 0.0f; color[2] = 0.0f; color[3] = 1.0f; 
	//human_model->SetUniform("incolor", color);

	Eigen::Matrix4f view_eigen = calcRenderExt(cam.R, cam.T);
	nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
	renderer.UpdateCanvasView(view_eigen);

	// create offscreen render object, you can render this object to a cuda texture or a cv::Mat
	// In this example code, I render this object to a cv::Mat and then use cv::imshow to demonstrate the rendering results
	// See interfaces of OffScreenRenderObject for more details
	auto human_offscreen = renderer.CreateOffscreenRenderObject(
		"box3", vs_vertex_position, fs_vertex_position, 1920, 1080, cam.K(0, 0), cam.K(1, 1), cam.K(0, 2), cam.K(1, 2), 1, true);
	std::cout << cam.K << std::endl;
	human_offscreen->SetIndices(objfloat4.indices);
	human_offscreen->SetBuffer("positions", objfloat4.vertices);
	human_offscreen->_SetViewByCameraRT(cam.R, cam.T);
	Eigen::Matrix4f camview = Eigen::Matrix4f::Identity();
	camview.block<3, 3>(0, 0) = cam.R;
	camview.block<3, 1>(0, 3) = cam.T;
	nanogui::Matrix4f camview_nano = eigen2nanoM4f(camview);
	human_offscreen->SetUniform("view", camview_nano);

	cv::Mat rendered_img(1920, 1080, CV_32FC4);
	std::vector<cv::Mat> rendered_imgs;
	rendered_imgs.push_back(rendered_img);
	// box3_offscreen to a cv::Mat (offscreen rendering)
	human_offscreen->DrawOffscreen();
	human_offscreen->DownloadRenderingResults(rendered_imgs);
	cv::imshow("rendered img", rendered_imgs[0]);
	std::cout << "img: " << rendered_imgs[0].cols << "," << rendered_imgs[0].rows << std::endl;
	std::vector<cv::Mat> channels(4);
	cv::split(rendered_imgs[0], channels);
	cv::Mat vis = pseudoColor(-channels[2]);

	cv::Mat mask = drawCVDepth(objeigen.vertices, objeigen.faces, cam);

	cv::Mat blend = blend_images(vis, mask, 0.5);

	cv::imshow("depth", vis);
	cv::imshow("mask", mask);
	cv::imshow("blend", blend);
	cv::imwrite("E:/render_test/nano_vis_align.png", vis);
	cv::imwrite("E:/render_test/nano_mask_align.png", mask);
	cv::imwrite("E:/render_test/nano_blend_align.png", blend);

	cv::waitKey();

	std::vector<float4> colors_float4(obj.vertex_num);

	for (int i = 0; i < obj.vertex_num; i++)
	{
		Eigen::Vector3f v = obj.vertices_vec[i];
		v = cam.K * (cam.R * v + cam.T);
		Eigen::Vector3f uv = v / v(2);
		float d = -queryDepth(channels[2], uv(0), uv(1));
		float w = queryDepth(channels[3], uv(0), uv(1));

		std::cout << "d: " << d << "  gt: " << v(2) << " diff: " << v(2) - d << "  w: " << w << std::endl;
		if (d > 0 && d != 1 && abs(d - v(2)) < 0.02)
		{
			colors_float4[i] = colormap[0];
		}
		else if (d < 0)
		{
			colors_float4[i] = colormap[1];
		}
		else
		{
			colors_float4[i] = colormap[2];
		}
	}

	renderer.ClearRenderObjects();
	auto smal_model = renderer.CreateRenderObject("smal_model", vs_phong_vertex_color, fs_phong_vertex_color);
	smal_model->SetIndices(objfloat4.indices);
	smal_model->SetBuffer("positions", objfloat4.vertices);
	smal_model->SetBuffer("normals", objfloat4.normals);
	smal_model->SetBuffer("incolor", colors_float4);

	int frameIdx = 0;
	while (!renderer.ShouldClose())
	{
		renderer.Draw();
		++frameIdx;
	}

	return 0;
}

void main()
{
	test_depth(); 
}