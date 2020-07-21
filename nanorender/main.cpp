#include <fstream>
#include <string>
#include <vector>
#include <iostream> 
#include <vector_functions.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nanogui/nanogui.h>
#include "../utils/timer.hpp"
#include "../utils/objloader.h"
#include "NanoRenderer.h"

int main()
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

	ObjModel model;
	loadOBJ("D:/Projects/animal_calib/nanorender/data/model.obj", model.vertices, model.normals, model.indices);
	auto human_model = renderer.CreateRenderObject("human_model", vs_phong_geometry, fs_phong_geometry);
	human_model->SetIndices(model.indices);
	human_model->SetBuffer("positions", model.vertices);
	human_model->SetBuffer("normals", model.normals);
	human_model->SetModelRT(nanogui::Matrix4f::translate(nanogui::Vector3f(-0.5, -0.5, 0.2)));

	// create offscreen render object, you can render this object to a cuda texture or a cv::Mat
	// In this example code, I render this object to a cv::Mat and then use cv::imshow to demonstrate the rendering results
	// See interfaces of OffScreenRenderObject for more details
	auto human_offscreen = renderer.CreateOffscreenRenderObject(
		"box3", vs_vertex_position, fs_vertex_position, 400, 400, 512, 384, 1, true);
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
