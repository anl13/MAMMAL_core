#include "../utils/definitions.h"
#include "anno_utils.h"
#include <fstream> 
#include <iostream> 

// util function 
void drawLabel(cv::Mat& img, int x, int y, int w, int h, std::string name, bool clicked)
{
	cv::Scalar color(255, 255, 255);
	if (clicked)
	{
		color = cv::Scalar(0, 255, 0);
	}
	cv::rectangle(img, cv::Rect(x + 10, y + 10, w - 20, h - 20), color, -1);
	cv::putText(img, name, cv::Point(x + 10, y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1.5);
}

AnnoConfig::AnnoConfig()
{
	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::string jsonfile = PROJECT_FOLDER; 
	jsonfile += "/annotator/anno_config.json";
	std::ifstream instream(jsonfile);
	if (!instream.is_open())
	{
		std::cout << "can not open " << jsonfile << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}

	posesolver_config = root["posesolver_config"].asString();
	pig_config        = root["pig_config"].asString();
	current_frame_id  = root["current_frame_id"].asInt();
	current_pig_id    = root["current_pig_id"].asInt(); 
	instream.close(); 
}