#include "test_main.h" 

#include <iostream> 
#include <fstream> 
#include <sstream> 
#include <io.h> 
#include <process.h> 

#include "../utils/camera.h"
#include "../utils/math_utils.h" 
#include "../utils/image_utils.h" 

#include "../utils/mesh.h"

#include "test_main.h"
#include "../utils/timer_util.h"

#include "pigsolverdevice.h" 
#include "pigsolver.h"
#include "gpuutils.h"

void test_compare_cpugpu()
{
	std::vector<Camera> cams = readCameras();

	// model data 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	PigSolverDevice smalgpu(smal_config);
	PigSolver smalcpu(smal_config); 
	PigSolver smal_target(smal_config); 

	// smal random pose 
	Eigen::VectorXf pose = Eigen::VectorXf::Random(smalgpu.GetJointNum() * 3) * 0.3;
	smal_target.SetPose(pose); 
	smal_target.UpdateVertices(); 
	Eigen::MatrixXf V = smal_target.GetVertices(); 
	std::vector<Eigen::Vector3f> V_vec;
	V_vec.resize(V.cols()); 
	for (int i = 0; i < V.cols(); i++) V_vec[i] = V.col(i); 

	smalcpu.m_targetVSameTopo = V;
	
	TimerUtil::Timer<std::chrono::milliseconds> tt;
	tt.Start(); 
	smalcpu.FitPoseToVerticesSameTopo(100, 0.0001);
	std::cout << "cpu fitting takes: " << tt.Elapsed() << std::endl; 

	tt.Start();
	smalgpu.fitPoseToVSameTopo(V_vec); 
	std::cout << "gpu fitting taks: " << tt.Elapsed() << std::endl; 

	Eigen::VectorXf pose_est_cpu = smalcpu.GetPose(); 
	std::vector<Eigen::Vector3f> pose_est_gpu_vec = smalgpu.GetPose(); 
	Eigen::VectorXf pose_est_gpu;
	pose_est_gpu.resize(pose_est_cpu.rows()); 
	for (int i = 0; i < pose_est_gpu_vec.size(); i++)pose_est_gpu.segment<3>(3 * i) = pose_est_gpu_vec[i];
	Eigen::VectorXf diff = pose_est_gpu - pose_est_cpu; 
	std::cout << "diff: " << diff.norm() << std::endl; 
	std::cout << "cpu: " << pose_est_cpu.transpose() << std::endl; 
	std::cout << "gpu: " << pose_est_gpu.transpose() << std::endl; 
}