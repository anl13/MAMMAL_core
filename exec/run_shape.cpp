//#include "main.h"
//#include <json/json.h> 
//#include <sstream> 
//#include <vector>
//#include <iostream> 
//#include <fstream> 
//#include <Eigen/Eigen> 
//
//#include "../utils/colorterminal.h" 
//#include "../utils/obj_reader.h"
//#include "../utils/timer_util.h"
//
//#include "../articulation/pigmodel.h"
//#include "../articulation/pigsolver.h"
//
//#include "../associate/framedata.h"
//#include "../utils/volume.h"
//
//#include "../utils/model.h"
//#include "../utils/dataconverter.h" 
//#include "../nanorender/NanoRenderer.h" 
//#include <vector_functions.hpp> 
//#include "main.h"
//
//#define RUN_SEQ
//#define VIS 
//#define DEBUG_VIS
////#define LOAD_STATE
//#define SHAPE_SOLVER
////#define VOLUME
//
//using std::vector;
//
//int run_shape()
//{
//	//std::string pig_config = "D:/Projects/animal_calib/smal/pigmodel_config.json";
//	std::string pig_config = "D:/Projects/animal_calib/smal/smal2_config.json";
//	std::string type = "shapeiter_smal"; 
//
//	std::string conf_projectFolder = "D:/Projects/animal_calib/";
//
//	SkelTopology topo = getSkelTopoByType("UNIV");
//	FrameData frame;
//	frame.configByJson(conf_projectFolder + "/associate/config.json");
//	int startid = frame.get_start_id();
//
//
//	//// rendering pipeline. 
//	auto CM = getColorMapEigen("anliang_rgb");
//	auto CM_float4 = getColorMapFloat4("anliang_rgb"); 
//
//
//	int framenum = frame.get_frame_num();
//	PigSolver shapesolver(pig_config); 
//	int m_pid = 0; // pig identity to solve now. 
//
//	for (int frameid = startid; frameid < startid + 1; frameid++)
//	{
//		std::cout << "processing frame " << frameid << std::endl;
//		frame.set_frame_id(frameid);
//		frame.fetchData();
//		//frame.view_dependent_clean();
//		//frame.matching_by_tracking();
//		frame.load_labeled_data();
//		frame.solve_parametric_model();
//		auto m_matched = frame.get_matched();
//		auto cameras = frame.get_cameras();
//		auto cam = cameras[0];
//
//		cv::Mat check = frame.visualizeIdentity2D();
//		cv::imwrite("E:/debug_pig3/"+type+"/identity.png", check); 
//
//		auto m_rois = frame.getROI(m_pid);
//		shapesolver.setCameras(frame.get_cameras());
//		shapesolver.normalizeCamera();
//		shapesolver.setId(m_pid);
//		shapesolver.setSource(m_matched[m_pid]);
//		shapesolver.normalizeSource();
//		shapesolver.InitNodeAndWarpField();
//		//shapesolver.LoadWarpField();
//		shapesolver.UpdateVertices();
//		
//		shapesolver.globalAlign();
//
//		shapesolver.SaveObj("E:/debug_pig3/"+type+"/global_align.obj");
//		shapesolver.SetTranslation(Eigen::Vector3d::Zero());
//		auto pose0 = shapesolver.GetPose();
//		pose0.setZero();
//		shapesolver.SetPose(pose0);
//		shapesolver.UpdateVertices();
//		shapesolver.SaveObj("E:/debug_pig3/"+type+"/scaled.obj");
//		std::cout << "scale: " << shapesolver.GetScale() << std::endl; 
//		shapesolver.optimizePose(5, 0.001);
//
//		shapesolver.SaveObj("E:/debug_pig3/"+type+"/init.obj");
//		
//		system("pause");
//		return 0;
//		shapesolver.m_rois = m_rois;
//
//		//Volume V; 
//		//Eigen::MatrixXd joints = shapesolver.getZ();
//		//std::cout << joints.transpose() << std::endl;
//		//V.center = joints.col(20).cast<float>();
//		//V.computeVolumeFromRoi(m_rois);
//		//std::cout << "compute volume now. " << std::endl;
//		//V.getSurface();
//		//std::stringstream ss; 
//		//ss << "E:/debug_pig3/gthull.xyz";
//		//V.saveXYZFileWithNormal(ss.str());
//		//std::stringstream cmd;
//		//cmd << "D:/Projects/animal_calib/PoissonRecon.x64.exe --in " << ss.str() << " --out " << ss.str() << ".ply";
//		//const std::string cmd_str = cmd.str();
//		//const char* cmd_cstr = cmd_str.c_str();
//		//system(cmd_cstr);
//		//return 0; 
//
//		std::shared_ptr<Model> targetModel = std::make_shared<Model>();
//		targetModel->Load("E:/debug_pig3/gthull.obj");
//		shapesolver.setTargetModel(targetModel);
//
//		shapesolver.totalSolveProcedure();
//		shapesolver.SaveWarpField();
//		shapesolver.SaveObj("E:/debug_pig3/"+type+"/warped.obj");
//
//		Eigen::VectorXd pose = shapesolver.GetPose();
//		pose.setConstant(0);
//		shapesolver.SetPose(pose);
//		shapesolver.SetTranslation(Eigen::Vector3d::Zero());
//		shapesolver.UpdateVertices();
//		shapesolver.SaveObj("E:/debug_pig3/"+type+"/meanpose.obj");
//
//		// visualize correspondances 
//		shapesolver.findCorr();
//		Eigen::VectorXi corr = shapesolver.m_corr;
//		
//		NanoRenderer renderer; 
//		renderer.Init(1920, 1080, float(cam.K(0, 0)), float(cam.K(1, 1)), float(cam.K(0, 2)), float(cam.K(1, 2)), -2.f, false);
//		
//		ObjModel m4c; 
//		convert3CTo4C(*targetModel, m4c);
//
//		std::vector<float4> colors_float4; 
//		colors_float4.resize(m4c.vertices.size(), make_float4(1, 1, 1, 0));
//		auto parts = shapesolver.GetBodyPart();
//		for (int i = 0; i < corr.size(); i++)
//		{
//			if (parts[i] != L_F_LEG)continue;
//			if (corr(i) >= 0)
//			{
//				int id = corr(i);
//				colors_float4[id] = CM_float4[0];
//			}
//		}
//		
//		renderer.ClearRenderObjects();
//		auto smal_model = renderer.CreateRenderObject("smal_model", vs_phong_vertex_color, fs_phong_vertex_color);
//		smal_model->SetIndices(m4c.indices);
//		smal_model->SetBuffer("positions", m4c.vertices);
//		smal_model->SetBuffer("normals", m4c.normals);
//		smal_model->SetBuffer("incolor", colors_float4);
//
//		Eigen::Matrix4f view_eigen = calcRenderExt(cam.R.cast<float>(), cam.T.cast<float>());
//		nanogui::Matrix4f view_nano = eigen2nanoM4f(view_eigen);
//		renderer.UpdateCanvasView(view_eigen);
//
//		while (!renderer.ShouldClose())
//		{
//			renderer.Draw();
//		}
//	}
//	return 0;
//}