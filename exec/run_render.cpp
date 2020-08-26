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
#include "../articulation/pigmodel.h"
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
	std::string smal_config = "D:/Projects/animal_calib/articulation/smal2_config.json";
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

