#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <io.h> 
#include <process.h> 

#include "../render/renderer.h"
#include "../render/render_object.h" 
#include "../render/render_utils.h"
#include "../utils/camera.h"
#include "../utils/math_utils.h" 
#include "../utils/image_utils.h" 

#include "pigmodel.h"
#include "pigsolver.h"
#include "../utils/mesh.h"

#include "test_main.h"


std::vector<Camera> readCameras()
{
	std::vector<Camera> cams;
	std::vector<int> m_camids = {
		0,1,2,5,6,7,8,9,10,11
	};
	int m_camNum = m_camids.size();
	std::string m_camDir = "D:/Projects/animal_calib/data/calibdata/extrinsic/";
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
			double a;
			camfile >> a;
			rvec(i) = a;
		}
		for (int i = 0; i < 3; i++)
		{
			double a;
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

std::vector<cv::Mat> readImgs()
{
	std::string folder = "D:/Projects/animal_calib/";
	std::string m_imgDir = folder + "/data/backgrounds/bg";
	std::vector<int> m_camids = {
		0,1,2,5,6,7,8,9,10,11
	};
	int m_camNum = m_camids.size();
	std::vector<cv::Mat> m_imgs;
	for (int camid = 0; camid < m_camNum; camid++)
	{
		std::stringstream ss;
		ss << m_imgDir << m_camids[camid] << "_undist.png";
		cv::Mat img = cv::imread(ss.str());
		if (img.empty())
		{
			std::cout << "img is empty! " << ss.str() << std::endl;
			exit(-1);
		}
		m_imgs.push_back(img);
	}
	return m_imgs;
}


void read_obj(std::string filename, Eigen::MatrixXf& vertices, Eigen::MatrixXu& faces)
{
	std::vector<Eigen::Vector3f> vs;
	std::vector<Eigen::Vector3u> fs;
	std::fstream reader;
	reader.open(filename.c_str(), std::ios::in);

	if (!reader.is_open())
	{
		std::cout << "[ObjData] file not exist!" << std::endl;
		exit(-1);
	}
	while (!reader.eof())
	{
		std::string dataType;
		reader >> dataType;

		if (reader.eof()) break;

		if (dataType == "v")
		{
			Eigen::Vector3f temp;
			reader >> temp.x() >> temp.y() >> temp.z();
			vs.push_back(temp);
		}
		else if (dataType == "f")
		{
			Eigen::Vector3u temp;
			reader >> temp.x() >> temp.y() >> temp.z();
			temp[0] = temp[0] - 1;
			temp[1] = temp[1] - 1;

			temp[2] = temp[2] - 1;

			fs.push_back(temp);
		}
		else
		{
			continue;
		}
	}
	int vertexnum = vs.size();
	int facenum = fs.size();
	vertices.resize(3, vertexnum);
	faces.resize(3, facenum);
	for (int i = 0; i < vertexnum; i++)vertices.col(i) = vs[i];
	for (int i = 0; i < facenum; i++)faces.col(i) = fs[i];
}


int test_write_video()
{
	// render config 
	std::string conf_projectFolder = "D:/projects/animal_calib/";
	std::vector<Eigen::Vector3f> CM = getColorMapEigenF("anliang_rgb");
	std::vector<Camera> cams = readCameras();

	// init a camera 
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
	std::cout << K << std::endl;

	Eigen::Vector3f pos = Eigen::Vector3f(0.873302, -0.961363, 0.444287);
	Eigen::Vector3f up = Eigen::Vector3f(-0.220837, 0.236654, 0.946164);
	Eigen::Vector3f center = Eigen::Vector3f::Zero();
	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/render/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);
	//m_renderer.s_camViewer.SetExtrinsic(cams[0].R.cast<float>(), cams[1].T.cast<float>());

	// init element obj
	// init element obj
	Mesh ballMesh(conf_projectFolder + "/render/data/obj_model/ball.obj");
	Mesh stickMesh(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	Mesh squareMesh(conf_projectFolder + "/render/data/obj_model/square.obj");
	Mesh cameraMesh(conf_projectFolder + "/render/data/obj_model/camera.obj");
	MeshEigen ballMeshEigen(ballMesh);
	MeshEigen stickMeshEigen(stickMesh);

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareMesh.faces_v_vec);
	chess_floor->SetVertices(squareMesh.vertices_vec);
	chess_floor->SetNormal(squareMesh.normals_vec, 2);
	chess_floor->SetTexcoords(squareMesh.textures_vec, 1);
	chess_floor->SetTransform({ 0.f, 0.f, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor);

	std::string tmp_folder = "F:/projects/model_preprocess/designed_pig/pig_prior/tmp/";

	//std::vector<cv::Mat> images;
	for (int k = 0; k < 100; k++)
	{
		m_renderer.colorObjs.clear();
		std::stringstream ss;
		ss << tmp_folder << "demo2_lie/demo" << k << ".obj";
		Eigen::MatrixXu faces_u;
		Eigen::MatrixXf vertices_f;
		read_obj(ss.str(), vertices_f, faces_u);
		RenderObjectColor* animal_model = new RenderObjectColor();

		vertices_f = vertices_f.colwise() + Eigen::Vector3f(0, 0, 0.21f);

		animal_model->SetFaces(faces_u);
		animal_model->SetVertices(vertices_f);
		animal_model->SetColor(Eigen::Vector3f(0.5, 0.5, 0.1));

		m_renderer.colorObjs.push_back(animal_model);

		m_renderer.Draw();
		cv::Mat img = m_renderer.GetImage();
		//images.push_back(img); 
		std::stringstream ss_out;
		ss_out << tmp_folder << "demo2_lie/img" << k << ".png";
		cv::imwrite(ss_out.str(), img);
	}
	//cv::Mat packed; 
	//packImgBlock(images, packed);
	//cv::Mat small;
	//cv::resize(packed, small, cv::Size(1920, 1080));
	//cv::imwrite(tmp_folder + "range/all.png", small);

	//GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	//while (!glfwWindowShouldClose(windowPtr))
	//{
	//	m_renderer.Draw();
	//	glfwSwapBuffers(windowPtr);
	//	glfwPollEvents();
	//};

	return 0;
}