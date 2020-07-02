#include <iostream> 
#include <fstream> 
#include <sstream> 
#ifndef WIN32
#include <unistd.h> 
#else 
#ifndef _UNISTD_H
#define _UNISTD_H
#include <io.h>
#include <process.h>
#endif /* _UNISTD_H */
#endif 
#include "../utils/camera.h"
#include "../utils/image_utils.h"
#include "../associate/framedata.h" 
#include "../utils/math_utils.h"
#include "../utils/colorterminal.h" 
#include "../smal/pigmodel.h"
#include "../utils/node_graph.h"
#include "../utils/model.h"
#include "../utils/dataconverter.h" 
#include "../nanorender/NanoRenderer.h"
#include <vector_functions.hpp>
#include "../utils/timer.hpp" 
#include "main.h"

int render_smal_test()
{
    std::cout << "In smal render now!" << std::endl; 

    std::string conf_projectFolder = "D:/projects/animal_calib/render";
	std::string smal_config = "D:/Projects/animal_calib/smal/smal2_config.json";
	SkelTopology m_topo = getSkelTopoByType("UNIV");
    /// read smal model 
	PigSolver smal(smal_config);
	std::string folder = smal.GetFolder();
    smal.UpdateVertices();

	FrameData framedata; 
	framedata.configByJson("D:/Projects/animal_calib/associate/config.json");
	framedata.set_frame_id(0);
	framedata.fetchData();
	auto cams = framedata.get_cameras();

	Eigen::Vector3f up; up << 0, 0, -1;
	Eigen::Vector3f pos; pos << -1, 1.5, -0.8;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();

	Model m3c;
	m3c.vertices = smal.GetVertices(); 
	m3c.faces = smal.GetFacesVert();
	m3c.CalcNormal();
	ObjModel m4c;
	convert3CTo4C(m3c, m4c);

	Camera cam = cams[0];
	NanoRenderer renderer; 
	renderer.Init(1920, 1080, float(cam.K(0, 0)), float(cam.K(1, 1)), float(cam.K(0, 2)), float(cam.K(1, 2)));
	std::cout << "cam.K: " << cam.K << std::endl;

	auto human_model = renderer.CreateRenderObject("human_model", vs_phong_geometry, fs_phong_geometry);
	human_model->SetIndices(m4c.indices);
	human_model->SetBuffer("positions", m4c.vertices);
	human_model->SetBuffer("normals", m4c.normals);
	
	Eigen::Matrix4f view_eigen = calcRenderExt(cam.R.cast<float>(), cam.T.cast<float>());
	nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
	//human_model->SetView(view_nano);
	
	renderer.UpdateCanvasView(view_eigen);

	int frameIdx = 0;
	while (!renderer.ShouldClose())
	{
		// rotate box1 along z axis
		//human_model->SetModelRT(Matrix4f::translate(Vector3f(-0.1, 0, 0.1)) * Matrix4f::rotate(Vector3f(0, 0, 1), glfwGetTime()));
		renderer.Draw();
		++frameIdx;
	}

    return 0; 
}


int test_depth()
{
	std::string conf_projectFolder = "D:/projects/animal_calib/render";
	std::string smal_config = "D:/Projects/animal_calib/smal/smal2_config.json";
	SkelTopology m_topo = getSkelTopoByType("UNIV");
	std::vector<float4> colormap = getColorMapFloat4("anliang_rgb");
	std::vector<Eigen::Vector3i> colormapeigen = getColorMapEigen("anliang_rgb");
	/// read smal model 
	//PigModel smal(smal_config); 
	PigSolver smal(smal_config);
	std::string folder = smal.GetFolder();
	smal.UpdateVertices();

	FrameData frame;
	frame.configByJson("D:/Projects/animal_calib/associate/config.json");
	frame.set_frame_id(0); 
	frame.fetchData();
	auto cams = frame.get_cameras();
	Camera cam = cams[0]; 

	Model m3c;
	m3c.vertices = smal.GetVertices();
	m3c.faces = smal.GetFacesVert();
	m3c.CalcNormal();
	ObjModel m4c;
	convert3CTo4C(m3c, m4c);

	NanoRenderer renderer;
	renderer.Init(1920, 1080, float(cam.K(0, 0)), float(cam.K(1, 1)), float(cam.K(0, 2)), float(cam.K(1, 2)),-2.0f);

	auto human_model = renderer.CreateRenderObject("human_model", vs_phong_geometry, fs_phong_geometry);
	human_model->SetIndices(m4c.indices);
	human_model->SetBuffer("positions", m4c.vertices);
	human_model->SetBuffer("normals", m4c.normals);
	//nanogui::Vector4f color;
	//color[0] = 1.0f; color[1] = 0.0f; color[2] = 0.0f; color[3] = 1.0f; 
	//human_model->SetUniform("incolor", color);

	Eigen::Matrix4f view_eigen = calcRenderExt(cam.R.cast<float>(), cam.T.cast<float>());
	nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
	renderer.UpdateCanvasView(view_eigen);

	// create offscreen render object, you can render this object to a cuda texture or a cv::Mat
	// In this example code, I render this object to a cv::Mat and then use cv::imshow to demonstrate the rendering results
	// See interfaces of OffScreenRenderObject for more details
	auto human_offscreen = renderer.CreateOffscreenRenderObject(
		"box3", vs_vertex_position, fs_vertex_position, 1920, 1080, cam.K(0,0), cam.K(1,1), cam.K(0,2), cam.K(1,2), 1, true);
	std::cout << cam.K << std::endl; 
	human_offscreen->SetIndices(m4c.indices);
	human_offscreen->SetBuffer("positions", m4c.vertices);
	human_offscreen->_SetViewByCameraRT(cam.R, cam.T);
	Eigen::Matrix4f camview = Eigen::Matrix4f::Identity(); 
	camview.block<3, 3>(0, 0) = cam.R.cast<float>(); 
	camview.block<3, 1>(0, 3) = cam.T.cast<float>(); 
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

	cv::Mat mask = drawCVDepth(m3c.vertices, m3c.faces, cam);

	cv::Mat blend = blend_images(vis, mask, 0.5);

	cv::imshow("depth", vis);
	cv::imshow("mask", mask);
	cv::imshow("blend", blend); 
	cv::imwrite("E:/debug_pig3/vis_align.png", vis);
	cv::imwrite("E:/debug_pig3/mask_align.png", mask);
	cv::imwrite("E:/debug_pig3/blend_align.png", blend);

	cv::waitKey();

	Eigen::MatrixXf vsf = smal.GetVertices().cast<float>();
	std::vector<Eigen::Vector3f> points(vsf.cols());
	std::vector<float> sizes(vsf.cols());
	std::vector<Eigen::Vector3i> pointcolors(vsf.cols());
	std::vector<float4> colors_float4(vsf.cols());

	for (int i = 0; i < vsf.cols(); i++)
	{
		points[i] = vsf.col(i);
		sizes[i] = 0.006f;

		Eigen::Vector3d v = vsf.col(i).cast<double>();
		v = cam.K * ( cam.R * v + cam.T); 
		Eigen::Vector3d uv = v / v(2);
		double d = -queryDepth(channels[2], uv(0), uv(1));
		double w = queryDepth(channels[3], uv(0), uv(1));

		std::cout << "d: " << d << "  gt: " << v(2) << " diff: " << v(2) - d << "  w: " << w << std::endl;
		if (d > 0 && d != 1 && abs(d - v(2)) < 0.02)
		{
			pointcolors[i] = colormapeigen[0];
			colors_float4[i] = colormap[0];
		}
		else if (d < 0)
		{
			pointcolors[i] = colormapeigen[1]; 
			colors_float4[i] = colormap[1];
		}
		else 
		{
			pointcolors[i] = Eigen::Vector3i(255, 255, 255);
			colors_float4[i] = colormap[2]; 
		}
	}
	
	renderer.ClearRenderObjects();
	auto smal_model = renderer.CreateRenderObject("smal_model", vs_phong_vertex_color, fs_phong_vertex_color);
	smal_model->SetIndices(m4c.indices);
	smal_model->SetBuffer("positions", m4c.vertices);
	smal_model->SetBuffer("normals", m4c.normals);
	smal_model->SetBuffer("incolor", colors_float4);

	//renderer.CreatePointCloudObjects(points, sizes, pointcolors);

	int frameIdx = 0;
	while (!renderer.ShouldClose())
	{
		renderer.Draw();
		++frameIdx;
	}

	return 0;
}