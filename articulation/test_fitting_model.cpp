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

#include "test_main.h"
#include "../utils/timer_util.h"

#include "pigsolverdevice.h" 
#include "pigmodeldevice.h"

// 20200808: reduce data dimension 
std::vector<Eigen::VectorXf> loadData()
{
	std::string filename = "F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\data\\samples.txt";
	std::ifstream is(filename);
	if (!is.is_open())
	{
		std::cout << "Not ok. " << std::endl;
		exit(-1);
	}
	int sample_num = 5452;
	int dim = 183;
	std::vector<Eigen::VectorXf> data(sample_num, Eigen::VectorXf::Zero(dim));
	for (int i = 0; i < sample_num; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			is >> data[i](j);
		}
	}
	return data;
}

void test_fitting()
{
	std::string pig_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	PigModelDevice gtpig(pig_config);
	PigSolverDevice pig(pig_config);

	std::vector<Eigen::VectorXf> data = loadData();
	//std::vector<Eigen::VectorXf> newdata;

	std::ofstream log_stream("F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\tmp\\samples\\log.txt");

	std::vector<int> elim = { 46, 47, 48, 49, 50, 51, 52, 53 };
	for (int i = 0; i < data.size(); i++)
	{
		Eigen::VectorXf pose = Eigen::VectorXf::Zero(62 * 3);
		pose.segment<61 * 3>(3) = data[i];
		for (int k = 0; k < elim.size(); k++)
		{
			int elim_num = elim[k]; 
			pose.segment<3>(3 * elim_num) = Eigen::Vector3f::Zero(); 
		}
		gtpig.SetPose(pose);
		gtpig.UpdateVertices();
		std::stringstream ss;
		ss << "F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\tmp\\samples\\gt" << i << ".obj";
		gtpig.saveObj(ss.str());

		TimerUtil::Timer<std::chrono::milliseconds> tt; 
		tt.Start(); 
		pig.fitPoseToVSameTopo(gtpig.GetVertices()); 

		std::cout << tt.Elapsed() << std::endl; 
		std::vector<Eigen::Vector3f> newpose = pig.GetPose();
		Eigen::VectorXf newposeeigen = convertStdVecToEigenVec(newpose); 
		//newdata.push_back(newposeeigen);
		std::stringstream ss1;
		ss1 << "F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\tmp\\samples\\est" << i << ".obj";
		pig.saveObj(ss1.str());

		std::cout << "finish " << i << std::endl;

		std::stringstream outfile;
		outfile << "F:/projects/model_preprocess/designed_pig/pig_prior/data/samples_new_pose/" << std::setw(4) << std::setfill('0') << i << ".txt"; 
		std::ofstream stream(outfile.str());
		stream << newposeeigen.transpose(); 
		stream.close(); 
	}
	log_stream.close();

}