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
#include "../utils/obj_reader.h"

#include "test_main.h"

// 20200808: reduce data dimension 
std::vector<Eigen::VectorXd> loadData()
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
	std::vector<Eigen::VectorXd> data(sample_num, Eigen::VectorXd::Zero(dim));
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
	PigModel gtpig(pig_config);
	PigSolver pig(pig_config);

	std::vector<Eigen::VectorXd> data = loadData();
	std::vector<Eigen::VectorXd> newdata;
	Eigen::VectorXd lastpose = Eigen::VectorXd::Zero(62 * 3);

	std::ofstream log_stream("F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\tmp\\samples\\log.txt");

	for (int i = 0; i < data.size() / 2; i += 20)
	{
		Eigen::VectorXd pose = Eigen::VectorXd::Zero(62 * 3);
		pose.segment<61 * 3>(3) = data[i];
		gtpig.SetPose(pose);
		gtpig.UpdateVertices();
		std::stringstream ss;
		ss << "F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\tmp\\samples\\gt" << i << ".obj";
		gtpig.SaveObj(ss.str());

		pig.SetPose(lastpose);
		pig.m_targetVSameTopo = gtpig.GetVertices();
		double loss = pig.FitPoseToVerticesSameTopo(40, 0.0001);
		Eigen::VectorXd newpose = pig.GetPose();
		newdata.push_back(newpose);
		std::stringstream ss1;
		ss1 << "F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\tmp\\samples\\est" << i << ".obj";
		pig.SaveObj(ss1.str());

		lastpose = newpose;
		std::cout << "finish " << i << std::endl;

		log_stream << "pose " << i << "  loss: " << std::setw(4) << std::setprecision(6) << loss << std::endl << std::endl;

		//if ((i + 1) % 1000 == 0)
		//{
		//	std::stringstream ss; 
		//	ss << "F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\data\\newsamples" << i << ".txt";
		//	std::ofstream stream(ss.str());
		//	for (int i = 0; i < newdata.size(); i++)
		//	{
		//		stream << newdata[i].transpose() << std::endl;
		//	}
		//	newdata.clear(); 
		//}

	}
	log_stream.close();

	std::ofstream stream("F:\\projects\\model_preprocess\\designed_pig\\pig_prior\\data\\newsamples.txt");
	for (int i = 0; i < newdata.size(); i++)
	{
		stream << newdata[i].transpose() << std::endl;
	}
}