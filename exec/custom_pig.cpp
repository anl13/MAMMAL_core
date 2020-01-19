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

#include "../smal/pigmodel.h"
#include "../smal/pigsolver.h"

#include "../associate/framedata.h"

using std::vector; 
const float kFloorDx = 0.28; 
const float kFloorDy = 0.2; 
std::vector<Eigen::Vector2i> bones = {
{ 1 , 0 },
{ 2 , 1 },
{ 3 , 2 },
{ 4 , 3 },
{ 5 , 4 },
{ 6 , 5 },
{ 7 , 6 },
{ 8 , 7 },
{ 9 , 8 },
{ 10 , 9 },
{ 11 , 4 },
{ 12 , 11 },
{ 13 , 12 },
{ 14 , 13 },
{ 15 , 14 },
{ 16 , 14 },
{ 17 , 16 },
{ 18 , 14 },
{ 19 , 18 },
{ 20 , 4 },
{ 21 , 20 },
{ 22 , 21 },
{ 23 , 22 },
{ 24 , 23 },
{ 25 , 24 },
{ 26 , 0 },
{ 27 , 26 },
{ 28 , 27 },
{ 29 , 28 },
{ 30 , 29 },
{ 31 , 0 },
{ 32 , 31 },
{ 33 , 32 },
{ 34 , 33 },
{ 35 , 34 },
{ 36 , 35 },
{ 37 , 36 },
{ 38 , 0 },
{ 39 , 38 },
{ 40 , 39 },
{ 41 , 40 },
{ 42 , 41 }
}; 

std::vector<std::pair<int, int> > Mapper = {
	{1, 239}, // nose
{1, 50}, // left eye
{1, 353}, // right eye
{1, 1551}, // left ear 
{1, 1571}, // right ear 
{0, 21}, 
{0, 6}, 
{0, 22 },
{0, 7},
{0, 23},
{0,8},
{0, 39},
{0, 27},
{0,40},
{0,28},
{0,41},
{0,29},
{-1, -1}, 
{0, 31},
{-1, -1},
{0, 2},
{-1, -1},
{-1,-1}
};

int main()
{
#ifdef _WIN32
    std::string folder = "D:/Projects/animal_calib/data/pig_model/"; 
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
#else 
	std::string folder = "/home/al17/animal/animal_calib/data/pig_model/"; 
	std::string conf_projectFolder = "/home/al17/animal/animal_calib/render";
#endif 
	SkelTopology topo = getSkelTopoByType("UNIV"); 
	FrameData frame; 
	frame.configByJson(conf_projectFolder + "/associate/config.json");
	int startid = frame.get_start_id(); 

    //// rendering pipeline. 
    auto CM = getColorMapEigen("anliang_rgb"); 

    // init a camera 
    Eigen::Matrix3f K; 
    K << 0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0, 1;
    // std::cout << K << std::endl; 

    Eigen::Vector3f up; up << 0,0, -1; 
    Eigen::Vector3f pos; pos << -1, 1.5, -0.8; 
    Eigen::Vector3f center = Eigen::Vector3f::Zero(); 
    // init renderer 
    Renderer::s_Init(); 
    Renderer m_renderer(conf_projectFolder + "/render/shader/"); 
    m_renderer.s_camViewer.SetIntrinsic(K, 1, 1); 
    m_renderer.s_camViewer.SetExtrinsic(pos, up, center); 

    // init element obj
    const ObjData ballObj(conf_projectFolder + "/render/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/render/data/obj_model/cylinder.obj");
	const ObjData cubeObj(conf_projectFolder + "/render/data/obj_model/cube.obj");
    const ObjData squareObj(conf_projectFolder + "/render/data/obj_model/square.obj"); 

	RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/render/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, true);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ kFloorDx, kFloorDy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor); 

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 

	int framenum = frame.get_frame_num();
	std::string videoname_render = conf_projectFolder + "/result_data/render.avi";
	cv::VideoWriter writer_render(videoname_render, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 25.0, cv::Size(1024, 1024));
	if (!writer_render.isOpened())
	{
		std::cout << "can not open video file " << videoname_render << std::endl;
		return -1;
	}

	PigSolver model(folder);
	model.setMapper(Mapper); 

	for (int frameid = startid; frameid < startid + framenum; frameid++)
	{
		std::cout << "processing frame " << frameid << std::endl; 
		frame.set_frame_id(frameid); 
		frame.fetchData();
		frame.matching(); 
		frame.tracking(); 
		
		std::vector<MatchedInstance> matched_source = frame.get_matched(); 
		m_renderer.colorObjs.clear(); 
		m_renderer.skels.clear(); 
		std::vector<Camera> cams = frame.get_cameras(); 
		model.setCameras(cams); 
		model.normalizeCamera(); 
		for (int pid = 0; pid < 4; pid++)
		{
			model.setSource(matched_source[pid]); 
			model.normalizeSource(); 
			model.globalAlign();
			model.optimizePose(100, 0.001); 

			RenderObjectColor* pig_render = new RenderObjectColor();
			Eigen::Matrix<unsigned int, -1, -1, Eigen::ColMajor> faces = model.GetFaces();
			Eigen::MatrixXf vs = model.GetVertices().cast<float>();
			pig_render->SetFaces(faces);
			pig_render->SetVertices(vs);
			Eigen::Vector3f color;
			color << 1.0f, 0.0f, 0.0f;
			pig_render->SetColor(color);
			m_renderer.colorObjs.push_back(pig_render);

			std::vector<Eigen::Vector3f> balls;
			std::vector< std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
			Eigen::Matrix3Xf joints = model.GetJoints().cast<float>();
			Eigen::VectorXi parents = model.GetParents().cast<int>();
			GetBallsAndSticks(joints, parents, balls, sticks);
			BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks,
				0.02f, 0.01f, 0.5 * CM[1]);
			m_renderer.skels.push_back(skelObject);
		}

		m_renderer.Draw();
		cv::Mat capture = m_renderer.GetImage();
		//writer_render.write(capture);
		glfwSwapBuffers(windowPtr);
		glfwPollEvents();
		while (!glfwWindowShouldClose(windowPtr))
		{

		}

	}
    return 0; 
}