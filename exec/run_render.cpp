#include "../render/renderer.h" 
#include "../render/render_object.h" 
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
#include "../render/eigen_util.h"
#include "../render/render_utils.h" 
#include "../utils/colorterminal.h" 
#include "../smal/pigmodel.h"

int render_smal_test()
{
    std::cout << "In smal render now!" << std::endl; 

    std::string conf_projectFolder = "D:/projects/animal_calib/render";
	std::string smal_config = "D:/Projects/animal_calib/smal/smal_config.json";
	std::string pig_config = "D:/Projects/animal_calib/smal/pigmodel_config.json";
	auto CM = getColorMapEigen("jet"); 
    /// read smal model 
    //Eigen::VectorXd pose = Eigen::VectorXd::Random(43*3); 
	PigModel smal(pig_config); 
    //smal.SetPose(pose); 
    smal.UpdateVertices();
	//smal.debugRemoveEye();
	//exit(-1);

	// init render
	Eigen::Matrix3f K;
	K << 0.698, 0, 0.502,
		0, 1.243, 0.483,
		0, 0, 1;
	std::cout << K << std::endl;

	Eigen::Vector3f up; up << 0, 0, -1;
	Eigen::Vector3f pos; pos << -1, 1.5, -0.8;
	Eigen::Vector3f center = Eigen::Vector3f::Zero();

	// init renderer 
	Renderer::s_Init();
	Renderer m_renderer(conf_projectFolder + "/shader/");
	m_renderer.s_camViewer.SetIntrinsic(K, 1, 1);
	m_renderer.s_camViewer.SetExtrinsic(pos, up, center);

	// init element obj
	const ObjData ballObj(conf_projectFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/data/obj_model/cylinder.obj");
	const ObjData squareObj(conf_projectFolder + "/data/obj_model/square.obj");

	GLFWwindow* windowPtr = m_renderer.s_windowPtr;
	m_renderer.SetBackgroundColor(Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f));

#if 1
    Eigen::Matrix<unsigned int,-1,-1, Eigen::ColMajor> faces = smal.GetFacesVert(); 
    Eigen::MatrixXf vs    = smal.GetVertices().cast<float>(); 

	RenderObjectColor* smal_render = new RenderObjectColor();
    smal_render->SetFaces(faces); 
    smal_render->SetVertices(vs);
    Eigen::Vector3f color; 
    color << 0.8f, 0.4f, 0.4f; 
    smal_render->SetColor(color); 
    m_renderer.colorObjs.push_back(smal_render); 

	//Eigen::MatrixXf texcoords = smal.GetTexcoords().cast<float>();
	//RenderObjectTexture* pig_tex_render = new RenderObjectTexture(); 
	//pig_tex_render->SetFaces(faces); 
	//pig_tex_render->SetVertices(vs); 
	//pig_tex_render->SetTexture(smal.GetFolder() + "/chessboard_blue.png");
	//pig_tex_render->SetTexcoords(texcoords); 
	//m_renderer.texObjs.push_back(pig_tex_render); 

	int vertexNum = smal.GetVertexNum();
	//vector<vector<int> > nonzero = smal.GetRegressorNoneZero(); 
	vector<vector<int> > nonzero = smal.GetWeightsNoneZero(); 
    std::vector<Eigen::Vector3f> colors(vertexNum,Eigen::Vector3f(0.8,0.8,0.8)); 
	std::vector<float> sizes(vertexNum, 0.006);

	std::vector<int> partIds = {
	    7,8,9,10
	};
	for (int k = 0; k < partIds.size(); k++)
	{
		int partId = partIds[k];
		for (int i = 0; i < nonzero[partId].size(); i++)
		{
			int col = nonzero[partId][i];
			colors[col] = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
		}
	}

	//std::vector<int> vids = {
	//	193, 108, 295, 1318,1329
	//};
	//for (int i = 0; i < vids.size(); i++)
	//{
	//	colors[vids[i]] = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
	//	sizes[vids[i]] = 0.005;
	//}
    
    std::vector<Eigen::Vector3f> balls; 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks; 
	Eigen::VectorXi parents;
    GetBallsAndSticks(vs, parents, balls, sticks); 
        
    // BallStickObject* skelObject = new BallStickObject(ballObj, stickObj, balls, sticks, 0.02f, 0.01f, color); 
    BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors); 
    m_renderer.skels.push_back(skelObject); 
#else 
	Eigen::MatrixXf joints = smal.GetJoints().cast<float>(); 
	int jointNum = smal.GetJointNum(); 
	Eigen::VectorXi parents = smal.GetParents(); 
	std::vector<float> sizes(jointNum, 0.03);
	std::vector<Eigen::Vector3f> balls;
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > sticks;
	std::vector<Eigen::Vector3f> colors(jointNum, Eigen::Vector3f(0.8, 0.8, 0.8));
	for (int i = 0; i < jointNum; i++)
	{
		colors[i] = CM[i * 5];
	}
	colors[24] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
	GetBallsAndSticks(joints, parents, balls, sticks);
	BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors);
	m_renderer.skels.push_back(skelObject); 

#endif
	

    while(!glfwWindowShouldClose(windowPtr))
    {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        m_renderer.Draw(); 
        glfwSwapBuffers(windowPtr); 
        glfwPollEvents(); 
    };
	
	smal.SaveObj("D:/Projects/cpp_nonrigid_icp/data/pig_stitched.obj");
    return 0; 
}

void renderScene()
{
	const float kFloorDx = 0.28;
	const float kFloorDy = 0.2;


    std::cout << "In render scene now!" << std::endl; 

    std::string conf_projectFolder = "D:/projects/animal_calib/render";
    auto CM = getColorMapEigen("anliang_rgb"); 

    // init a camera 
    Eigen::Matrix3f K; 
    K << 0.5, 0, 0.5, 0, 0.5, 0.5, 0, 0, 1;
    std::cout << K << std::endl; 

    Eigen::Vector3f up; up << 0,0, -1; 
    Eigen::Vector3f pos; pos << -1, 1.5, -0.8; 
    Eigen::Vector3f center = Eigen::Vector3f::Zero(); 

    // init renderer 
    Renderer::s_Init(); 
    Renderer m_renderer(conf_projectFolder + "/shader/"); 
    m_renderer.s_camViewer.SetIntrinsic(K, 1, 1); 
    m_renderer.s_camViewer.SetExtrinsic(pos, up, center); 

    // init element obj
    const ObjData ballObj(conf_projectFolder + "/data/obj_model/ball.obj");
	const ObjData stickObj(conf_projectFolder + "/data/obj_model/cylinder.obj");
    const ObjData squareObj(conf_projectFolder + "/data/obj_model/square.obj"); 
    const ObjData cameraObj(conf_projectFolder + "/data/obj_model/camera.obj"); 

    RenderObjectTexture* chess_floor = new RenderObjectTexture();
	chess_floor->SetTexture(conf_projectFolder + "/data/chessboard.png");
	chess_floor->SetFaces(squareObj.faces, true);
	chess_floor->SetVertices(squareObj.vertices);
	chess_floor->SetTexcoords(squareObj.texcoords);
	chess_floor->SetTransform({ kFloorDx, kFloorDy, 0.0f }, { 0.0f, 0.0f, 0.0f }, 1.0f);
	m_renderer.texObjs.push_back(chess_floor); 

    std::string point_file = conf_projectFolder + "/../build/results/points3d.txt";
    std::vector<Eigen::Vector3d> points = read_points(point_file);
    std::vector<float> sizes(points.size(), 0.05f); 
    std::vector<Eigen::Vector3f> balls, colors; 
    balls.resize(points.size()); 
    colors.resize(points.size());
    for(int i = 0; i < points.size(); i++)
    {
        balls[i] = points[i].cast<float>(); 
        colors[i] = CM[0]; 
    }
    BallStickObject* skelObject = new BallStickObject(ballObj, balls, sizes, colors); 
    m_renderer.skels.push_back(skelObject); 

    GLFWwindow* windowPtr = m_renderer.s_windowPtr; 

    while(!glfwWindowShouldClose(windowPtr))
    {
        m_renderer.Draw(); 
        glfwSwapBuffers(windowPtr); 
        glfwPollEvents(); 
    };
}

