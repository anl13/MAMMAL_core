#include "test_main.h"

#include "pigsolverdevice.h"

#include "../GMM/gmm.h"

void test_pointer()
{

	std::cout << " in test pointer. " << std::endl; 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	std::shared_ptr<PigModelDevice> p_smal = std::make_shared<PigModelDevice>(smal_config); 

	GMM gmm;
	gmm.Load(); 

	//Eigen::VectorXf pose = Eigen::VectorXf::Zero(62 * 3); 
	//std::cout << "gmm mu size: " << gmm.mu[0].rows() << std::endl; 
	//std::vector<int> pose_to_optimize = p_smal->getPoseToOptimize(); 
	//for (int i = 1; i < pose_to_optimize.size(); i++)
	//{
	//	int jid = pose_to_optimize[i];
	//	std::cout << "jid: " << jid << std::endl;
	//	pose.segment<3>(3 * jid) = gmm.mu[0].segment<3>(3 * i-3);
	//}
	//p_smal->SetPose(pose); 
	p_smal->UpdateVertices(); 

	std::cout << gmm.mu[0].tail(6) << std::endl;

	p_smal->saveObj("G:/pig_results_newtrack/tmp/mu.obj");

	system("pause"); 
}