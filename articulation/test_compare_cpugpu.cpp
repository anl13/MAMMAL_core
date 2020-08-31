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

	// smal random pose 
	Eigen::VectorXf pose = Eigen::VectorXf::Random(smalgpu.GetJointNum() * 3) * 0.3;
	smalgpu.SetPose(pose);
	smalgpu.UpdateVertices();

	PigSolver smalcpu(smal_config); 
	smalcpu.SetPose(pose); 
	smalcpu.UpdateVertices(); 

	std::cout << "...... TEST Full Theta Jacobi ......" << std::endl; 

	//Eigen::MatrixXf J_joint_cpu, J_vert_cpu, J_joint_gpu, J_vert_gpu; 
	//pcl::gpu::DeviceArray2D<float> J_joint_device, J_vert_device; 

	//TimerUtil::Timer<std::chrono::microseconds> tt; 
	//tt.Start();
	//smalcpu.CalcPoseJacobiFullTheta(J_joint_cpu, J_vert_cpu, true); 
	//std::cout << "cpu: " << tt.Elapsed() << " mcs" << std::endl; 
	//tt.Start(); 
	//smalgpu.calcPoseJacobiFullTheta_device(J_joint_device, J_vert_device);
	//std::cout << "gpu: " << tt.Elapsed() << " mcs" << std::endl; 
	//J_joint_gpu = J_joint_cpu;
	//J_vert_gpu = J_vert_cpu; 
	//J_joint_gpu.setZero(); 
	//J_vert_gpu.setZero(); 
	int jointnum = smalgpu.GetJointNum();
	int vertexnum = smalgpu.GetVertexNum(); 
	//J_joint_device.download(J_joint_gpu.data(), 3 * jointnum * sizeof(float)); 
	//J_vert_device.download(J_vert_gpu.data(), 3 * vertexnum * sizeof(float)); 

	////Eigen::MatrixXf J_joint_diff = J_joint_cpu - J_joint_gpu; 
	////std::cout << J_joint_diff.norm() << std::endl; 
	////std::cout << "J_joint_cpu: " << std::endl << J_joint_cpu.block<10, 10>(0, 0) << std::endl; 
	////std::cout << "J_Joint_gpu: " << std::endl << J_joint_gpu.block<10, 10>(0, 0) << std::endl; 

	//Eigen::MatrixXf J_vert_diff = J_vert_cpu - J_vert_gpu; 
	//std::cout << "J_vert diff F-norm: " << J_vert_diff.norm() << std::endl; 
	//std::cout << "J_vert_cpu: " << std::endl << J_vert_cpu.block<10, 10>(0, 0) << std::endl; 
	//std::cout << "J_vert_gpu: " << std::endl << J_vert_gpu.block<10, 10>(0, 0) << std::endl; 
	//std::cout << "no error. " << std::endl; 


	std::cout << "...... TEST Part Theta Jacobi ......" << std::endl; 
	TimerUtil::Timer<std::chrono::microseconds> tt2; 
	
	Eigen::MatrixXf J_joint_part_cpu, J_vert_part_cpu, J_joint_part_gpu, J_vert_part_gpu; 
	pcl::gpu::DeviceArray2D<float> J_joint_part_device, J_vert_part_device; 

	tt2.Start(); 
	smalcpu.CalcPoseJacobiPartTheta(J_joint_part_cpu, J_vert_part_cpu); 
	std::cout << "cpu: " << tt2.Elapsed() << std::endl; 
	tt2.Start(); 
	smalgpu.calcPoseJacobiPartTheta_device(J_joint_part_device, J_vert_part_device); 
	//std::cout << "gpu: " << tt2.Elapsed() << std::endl; 
	//J_joint_part_gpu = J_joint_part_cpu; J_joint_part_gpu.setZero(); 
	//J_vert_part_gpu = J_vert_part_cpu; J_joint_part_gpu.setZero(); 
	//J_joint_part_device.download(J_joint_part_gpu.data(), 3 * jointnum * sizeof(float));
	//J_vert_part_device.download(J_vert_part_gpu.data(), 3 * vertexnum * sizeof(float)); 
	//Eigen::MatrixXf J_vert_part_diff = J_vert_part_cpu - J_vert_part_gpu; 
	//std::cout << "J_vert part diff F-norm: " << J_vert_part_diff.norm() << std::endl; 

	TimerUtil::Timer<std::chrono::microseconds> tt3; 
	tt3.Start();
	Eigen::MatrixXf ATA = J_vert_part_cpu.transpose() * J_vert_part_cpu; 
	std::cout << "compute ATA: " << tt3.Elapsed() << " mcs" << std::endl;

	
	pcl::gpu::DeviceArray2D<float> ATA_device; 
	Eigen::MatrixXf ATA_gpu = ATA; ATA_gpu.setZero(); 
	int H = ATA.cols(); std::cout << " H : " << H << std::endl << std::endl; 
	//ATA_device.create(H, H); 
	tt3.Start();
	computeATA_device(J_vert_part_device, J_vert_part_device.cols(), J_vert_part_device.rows(), ATA_device); 
	ATA_device.download(ATA_gpu.data(), ATA_gpu.cols()*sizeof(float)); 
	std::cout << "compute ATA gpu: " << tt3.Elapsed() << " mcs" << std::endl; 

	Eigen::MatrixXf ATA_diff = ATA - ATA_gpu; 
	std::cout << "f-norm: " << ATA_diff.norm() << std::endl; 

	std::cout << "ATA cpu: " << std::endl <<
		ATA.block<9, 9>(0, 0) << std::endl; 
	std::cout << "ATA gpu: " << std::endl <<
		ATA_gpu.block<9, 9>(0, 0) << std::endl; 
}