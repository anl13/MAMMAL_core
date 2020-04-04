#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include "../utils/colorterminal.h" 
#include "../utils/obj_reader.h"
#include "../render/render_object.h"
#include "../render/render_utils.h"
#include "../render/renderer.h"

#include "../associate/framedata.h"
#include "../smal/pigmodel.h"
#include "../smal/pigsolver.h"
#include "../utils/math_utils.h"
#include "main.h"
#include <fstream> 

using std::vector; 

int TestVerticesAlign()
{   
    std::string target_file = "D:/Projects/animal_calib/data/smalr/pig_B_tpose.obj"; 
    OBJReader objreader; 
    objreader.read(target_file); 
	std::cout << "tex to vert size: " << objreader.tex_to_vert.size() << std::endl; 

	//std::ofstream os_tex("D:/Projects/animal_calib/data/smalr/textures.txt");
	//os_tex << objreader.textures_eigen.transpose();
	//os_tex.close();
	//std::ofstream os_map("D:/Projects/animal_calib/data/smalr/tex_to_vert.txt");
	//for (int i = 0; i < objreader.textures.size(); i++) os_map << objreader.tex_to_vert[i] << std::endl;
	//os_map.close();
	//std::ofstream os_fv("D:/Projects/animal_calib/data/smalr/faces_vert.txt");
	//for (int i = 0; i < objreader.faces_v.size(); i++) {
	//	os_fv << objreader.faces_v[i](0) << " "
	//		<< objreader.faces_v[i](1) << " "
	//		<< objreader.faces_v[i](2) << "\n";
	//}
	//os_fv.close();
	//std::ofstream os_ft("D:/Projects/animal_calib/data/smalr/faces_tex.txt");
	//for (int i = 0; i < objreader.faces_t.size(); i++)
	//{
	//	os_ft << objreader.faces_t[i](0) << " "
	//		<< objreader.faces_t[i](1) << " "
	//		<< objreader.faces_t[i](2) << "\n";
	//}
	//os_ft.close(); 
	//system("pause");
	//return 0;

    /// read smal model 
    std::string smal_config = "D:/Projects/animal_calib/smal/smal_config.json";
    PigModel smal(smal_config); 
	smal.readShapeParam("D:/Projects/animal_calib/data/smalr/pigshape.txt");
    smal.UpdateVertices();
	smal.UpdateVerticesTex(); 

    //PigSolver solver(smal_config); 
    //solver.setTargetVSameTopo(objreader.vertices_eigen.cast<double>()); 
    //solver.globalAlignToVerticesSameTopo(); 
    //solver.FitShapeToVerticesSameTopo(50, 0.00000001); 
    //solver.saveShapeParam("D:/Projects/animal_calib/data/smalr/pigshape.txt"); 

    //// rendering pipeline. 
    std::string renderFolder = "D:/Projects/animal_calib/render";
    auto CM = getColorMapEigen("anliang_rgb"); 

    // init a camera 
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
    // std::cout << K << std::endl; 

    Eigen::Vector3f up; up << 0,0, -1; 
    Eigen::Vector3f pos; pos << -1, 1.5, -0.8; 
    Eigen::Vector3f center = Eigen::Vector3f::Zero(); 

    // init renderer 
    Renderer::s_Init(); 
    Renderer m_renderer(renderFolder + "/shader/"); 
    m_renderer.s_camViewer.SetIntrinsic(K, 1, 1); 
    m_renderer.s_camViewer.SetExtrinsic(pos, up, center); 

    // init element obj
    const ObjData ballObj(renderFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(renderFolder + "/data/obj_model/cylinder.obj");
    const ObjData squareObj(renderFolder + "/data/obj_model/square.obj"); 

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 

    RenderObjectColor* smal_render = new RenderObjectColor(); 
    Eigen::Matrix<unsigned int,-1,-1,Eigen::ColMajor> faces = smal.GetFacesTex(); 
	Eigen::MatrixXf vs = smal.GetVerticesTex().cast<float>(); 
	//Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces = solver.GetFacesVert();
	//Eigen::MatrixXf vs = solver.GetVertices().cast<float>(); 
    smal_render->SetFaces(faces); 
    smal_render->SetVertices(vs);
    Eigen::Vector3f color; 
    color << 0.8f, 0.8f, 0.7f; 
    smal_render->SetColor(color); 
    m_renderer.colorObjs.push_back(smal_render); 

 //   RenderObjectColor* smal_target = new RenderObjectColor(); 
 //   Eigen::MatrixXf vs_gt = objreader.vertices_eigen; 
	//Eigen::Matrix<unsigned int, -1, -1> faces_gt = objreader.faces_v_eigen.cast<unsigned int>();
 //   smal_target->SetFaces(faces_gt); 
 //   smal_target->SetVertices(vs_gt); 
 //   color << 0.0f, 0.6f, 0.0f;
 //   smal_target->SetColor(color); 
 //   m_renderer.colorObjs.push_back(smal_target);  

	//int vertexNum = vs.cols();
	//std::vector<Eigen::Vector3f> colors(vertexNum, Eigen::Vector3f(0, 0.8, 0));
	//std::vector<float> sizes(vertexNum, 0.006);
	//std::vector<Eigen::Vector3f> balls;
	//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	//Eigen::VectorXi parents;
	//GetBallsAndSticks(vs, parents, balls, sticks);

	//auto body_parts = smal.GetBodyPart();
	//for (int i = 0; i < vertexNum; i++)
	//{
	//	int b = (int)body_parts[i];
	//	std::cout << b << " " << std::endl;
	//	colors[i] = CM[(int)body_parts[i]];
	//}
	//BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors);
	//m_renderer.skels.push_back(skelObject);

    while(!glfwWindowShouldClose(windowPtr))
    {
        m_renderer.Draw(); 

        glfwSwapBuffers(windowPtr); 
        glfwPollEvents(); 
    };
    return 0; 
}

int removeSmalTail()
{
	std::string target_file = "D:/Projects/animal_calib/data/smalr/pig_B_tpose.obj";
	OBJReader objreader;
	objreader.read(target_file);
	std::cout << "tex to vert size: " << objreader.tex_to_vert.size() << std::endl;

	//std::ofstream os_tex("D:/Projects/animal_calib/data/smalr/textures.txt");
	//os_tex << objreader.textures_eigen.transpose();
	//os_tex.close();
	//std::ofstream os_map("D:/Projects/animal_calib/data/smalr/tex_to_vert.txt");
	//for (int i = 0; i < objreader.textures.size(); i++) os_map << objreader.tex_to_vert[i] << std::endl;
	//os_map.close();
	//std::ofstream os_fv("D:/Projects/animal_calib/data/smalr/faces_vert.txt");
	//for (int i = 0; i < objreader.faces_v.size(); i++) {
	//	os_fv << objreader.faces_v[i](0) << " "
	//		<< objreader.faces_v[i](1) << " "
	//		<< objreader.faces_v[i](2) << "\n";
	//}
	//os_fv.close();
	//std::ofstream os_ft("D:/Projects/animal_calib/data/smalr/faces_tex.txt");
	//for (int i = 0; i < objreader.faces_t.size(); i++)
	//{
	//	os_ft << objreader.faces_t[i](0) << " "
	//		<< objreader.faces_t[i](1) << " "
	//		<< objreader.faces_t[i](2) << "\n";
	//}
	//os_ft.close(); 
	//system("pause");
	//return 0;

	/// read smal model 
	std::string smal_config = "D:/Projects/animal_calib/smal/smal_config.json";
	PigModel smal(smal_config);
	//smal.readShapeParam("D:/Projects/animal_calib/data/smalr/pigshape.txt");
	smal.UpdateVertices();
	//smal.SaveObj("D:/Projects/animal_calib/data/smalr_notail/with_tail.obj");

	//PigSolver solver(smal_config); 
	//solver.setTargetVSameTopo(objreader.vertices_eigen.cast<double>()); 
	//solver.globalAlignToVerticesSameTopo(); 
	//solver.FitShapeToVerticesSameTopo(50, 0.00000001); 
	//solver.saveShapeParam("D:/Projects/animal_calib/data/smalr/pigshape.txt"); 

	//// rendering pipeline. 
	std::string renderFolder = "D:/Projects/animal_calib/render";
	auto CM = getColorMapEigen("anliang_rgb");

	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
	// std::cout << K << std::endl; 

	Eigen::Vector3f up; up << 0, 0, -1;
	Eigen::Vector3f pos; pos << -1, 1.5, -0.8;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();

	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(renderFolder + "/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	// init element obj
	const ObjData ballObj(renderFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(renderFolder + "/data/obj_model/cylinder.obj");
	const ObjData squareObj(renderFolder + "/data/obj_model/square.obj");

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;

	//   RenderObjectColor* smal_target = new RenderObjectColor(); 
	//   Eigen::MatrixXf vs_gt = objreader.vertices_eigen; 
	//Eigen::Matrix<unsigned int, -1, -1> faces_gt = objreader.faces_v_eigen.cast<unsigned int>();
	//   smal_target->SetFaces(faces_gt); 
	//   smal_target->SetVertices(vs_gt); 
	//   color << 0.0f, 0.6f, 0.0f;
	//   smal_target->SetColor(color); 
	//   m_renderer.colorObjs.push_back(smal_target);  


	Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces = smal.GetFacesVert();
	Eigen::MatrixXf vs = smal.GetVertices().cast<float>();

	int vertexNum = vs.cols();
	std::vector<Eigen::Vector3f> colors(vertexNum, Eigen::Vector3f(0, 0.8, 0));
	std::vector<float> sizes(vertexNum, 0.006);


	auto body_parts = smal.GetBodyPart();
	//for (int i = 0; i < vertexNum; i++)
	//{
	//	int b = (int)body_parts[i];
	//	colors[i] = CM[(int)body_parts[i]];
	//}

	std::vector<int> tail_rings = {
		459,1820,2424,2423,2419,1813,454,458
	};
	std::vector<int> tail_inner = {
		1811,2418,2030,2420,1812,455,52,453
	};
	//for (int i = 0; i < tail_inner.size(); i++)
	//{
	//	int a = tail_inner[i];
	//	colors[a] = CM[0];
	//}


	// remove tail 
	int count_tail = 0; 
	std::vector<int> tails;
	for (int i = 0; i < body_parts.size(); i++)
	{
		if (body_parts[i] == TAIL) {
			count_tail++;
			tails.push_back(i);
		}
	}
	int N = vertexNum - count_tail;
	std::vector<int> old_to_new(vertexNum,-1); 
	std::vector<int> new_to_old(N, -1);
	Eigen::MatrixXd vsd = smal.GetVertices();
	Eigen::MatrixXd vs_new; 
	Eigen::Matrix<unsigned int, 3, -1, Eigen::ColMajor> face_new; 
	vs_new.resize(3, N);
	int n = 0; 
	
	for (int i = 0; i < vertexNum; i++)
	{
		if (in_list(i, tails)) {
			continue;
		}
		vs_new.col(n) = vsd.col(i);
		old_to_new[i] = n;
		new_to_old[n] = i;
		n++;
	}
	std::vector<Eigen::Vector3u> faces_list; 
	std::vector<Eigen::Vector3u> faces_tex_list;
	int faceNum = faces.cols();
	for (int i = 0; i < faceNum; i++)
	{
		Eigen::Vector3u f = faces.col(i);
		if (body_parts[f(0)] == TAIL ||
			body_parts[f(1)] == TAIL ||
			body_parts[f(2)] == TAIL)
		{
			continue;
		}
		else
		{
			Eigen::Vector3u fnew;
			fnew(0) = old_to_new[f(0)];
			fnew(1) = old_to_new[f(1)];
			fnew(2) = old_to_new[f(2)];
			faces_list.push_back(fnew);
			faces_tex_list.push_back(objreader.faces_t[i].cast<unsigned int>());
		}
	}
	std::vector<Eigen::Vector3u> add_faces = {
		{455,1812,52},
	{52,1812,453},
	{453,1812,1811},
	{1811,1812,2418},
	{2418,1812,2030},
	{2030,1812,2420}
	};
	for (int i = 0; i < add_faces.size(); i++)
	{
		Eigen::Vector3u fnew;
		fnew(0) = old_to_new[add_faces[i](0)];
		fnew(1) = old_to_new[add_faces[i](1)];
		fnew(2) = old_to_new[add_faces[i](2)];
		faces_list.push_back(fnew);
	}
	int F = faces_list.size();
	face_new.resize(3, F);
	for (int i = 0; i < F; i++)
	{
		face_new.col(i) = faces_list[i];
	}

	// save results (note: pig shaped smal)
	std::string folder = "D:/Projects/animal_calib/data/smalr_notail/";
	OBJReader saver; 
	saver.vertices_eigen = vs_new;
	saver.faces_v_eigen = face_new;
	saver.write(folder + "smal_notail.obj");
	// -- v and parts
	std::ofstream os_v(folder+"vertices.txt");
	//os_v << vs_new.transpose(); 
	for (int i = 0; i < N; i++)
	{
		os_v << vs_new(0, i) << " " << vs_new(1, i) << " " << vs_new(2, i) << "\n";
	}
	os_v.close(); 
	std::ofstream os_parts(folder + "body_parts.txt");
	for (int i = 0; i < N; i++)
	{
		int oldvindex = new_to_old[i];
		os_parts << body_parts[oldvindex] << "\n";
	}
	os_parts.close();
	// -- f 
	std::ofstream os_f(folder + "faces_vert.txt");
	for (int i = 0; i < F; i++)
	{
		os_f << faces_list[i](0) << " "
			<< faces_list[i](1) << " "
			<< faces_list[i](2) << "\n";
	}
	os_f.close();
	// -- joints 
	int jointnum = 27;
	std::vector<int> mapj(27, -1);
	for (int i = 0; i < 26; i++)mapj[i] = i;
	mapj[26] = 32;
	auto joints = smal.GetJoints();
	Eigen::MatrixXd jointsnew(3, jointnum);
	for (int i = 0; i < jointnum; i++) jointsnew.col(i) = joints.col(mapj[i]);
	std::ofstream os_joint(folder + "t_pose_joints.txt");
	os_joint << jointsnew.transpose();
	os_joint.close();
	// -- skinning weights
	Eigen::MatrixXd weights = smal.GetLBSWeights(); // joint num * vertices num
	Eigen::MatrixXd W;
	W.resize(jointnum, N);
	for (int i = 0; i < jointnum; i++)
	{
		for (int j = 0; j < N; j++)
		{
			int a = mapj[i];
			int b = new_to_old[j];
			W(i, j) = weights(a, b);
		}
	}
	std::ofstream os_w(folder + "skinning_weights.txt");
	for (int i = 0; i < jointnum; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (W(i, j) > 0) os_w << i << " " << j << " " << W(i, j) << std::endl;
		}
	}
	os_w.close();
	// -- regressor 
	auto JR = smal.GetJRegressor();//vnum*jnum
	std::ofstream os_r(folder + "J_regressor.txt");
	for (int i = 0; i < jointnum; i++)
	{
		for (int j = 0; j < N; j++)
		{
			int oi = mapj[i];
			int oj = new_to_old[j];
			double w = JR(oj, oi);
			if (w > 0) os_r << i << " " << j << " " << w << std::endl;
		}
	}
	os_r.close();
	// -- shape dir 
	Eigen::MatrixXd shapedir(3 * N, 41);
	auto sd = smal.GetShapeBlendV();
	for (int i = 0; i < N; i++)
	{
		int old = new_to_old[i];
		shapedir.middleRows(i * 3, 3) = sd.middleRows(old * 3, 3);
	}
	std::ofstream os_shape(folder + "shapedirs.txt");
	os_shape << shapedir;
	os_shape.close();
	// -- parents 
	std::vector<int> parents = {
		-1,0,1,2,3,4,5,6,7,8,9,6,11,12,13,6,15,0,17,18,19,0,21,22,23,0,16
	};
	std::ofstream os_p(folder + "parents.txt");
	for (int i = 0; i < parents.size(); i++)os_p << parents[i] << "\n";
	os_p.close();

	// render
	RenderObjectColor* smal_render = new RenderObjectColor();
	smal_render->SetFaces(face_new);
	smal_render->SetVertices(vs_new.cast<float>());
	Eigen::Vector3f color;
	color << 0.8f, 0.8f, 0.7f;
	smal_render->SetColor(color);
	m_renderer.colorObjs.push_back(smal_render);

	//std::vector<Eigen::Vector3f> balls;
	//std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	//Eigen::VectorXi parents;
	//GetBallsAndSticks(vs, parents, balls, sticks);
	//BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors);
	//m_renderer.skels.push_back(skelObject);

	m_renderer.SetBackgroundColor(Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f));
	while (!glfwWindowShouldClose(windowPtr))
	{
		m_renderer.Draw();

		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
	};
	return 0;
}


int ComputeSymmetry()
{
	// The smal notail model is symmetric to y axis
	std::string target_file = "D:/Projects/animal_calib/data/smalr_notail/smal_notail.obj";
	OBJReader mesh; 
	mesh.read(target_file);
	
	int vertexNum = mesh.vertices.size();

	Eigen::MatrixXd flipped = mesh.vertices_eigen;
	flipped.row(1) = flipped.row(1) * -1;
	
	std::vector<std::pair<int, int> > S;
	for (int i = 0; i < vertexNum; i++)
	{
		for (int j = i+1; j < vertexNum; j++)
		{
			double dist = (mesh.vertices[i] - flipped.col(j)).norm();
			if (dist < 0.0005) S.push_back({ i,j });
		}
	}
	std::cout << S.size() << std::endl;

	system("pause");
	return 0; 
}