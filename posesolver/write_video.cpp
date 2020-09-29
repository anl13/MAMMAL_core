#include "main.h"
#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 
#include <opencv2/opencv.hpp>

#include "../utils/colorterminal.h" 
#include "../utils/timer_util.h"
#include "../articulation/pigmodel.h"
#include "../articulation/pigsolver.h"
#include "../associate/framedata.h"
#include "../utils/volume.h"
#include <vector_functions.hpp>


int write_video()
{
	std::string video_folder = "E:/task_animation/Videos7/";

	std::string conf_projectFolder = "D:/Projects/animal_calib/";

	FrameData frame;
	frame.configByJson(conf_projectFolder + "/associate/config.json");

	
	std::vector<cv::VideoWriter> writers; 
	writers.resize(10);
	for (int i = 0; i < 10; i++)
	{
		std::string filename = video_folder + std::to_string(i) + ".avi";
		writers[i] = cv::VideoWriter(filename, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 25.0, cv::Size(1920, 1080));
	}


	int m_pid =0;
	for (int frameid = 9300; frameid < 9300 + 500;  frameid++)
	{
		std::cout << "processing frame " << frameid << std::endl;
		frame.set_frame_id(frameid);
		frame.fetchData();
		frame.load_clusters();
		
		auto raws = frame.get_imgs_undist(); 
		
		auto matched = frame.get_matched(); 
		auto data = matched[m_pid];
		for (int k = 0; k < data.view_ids.size(); k++)
		{
			int view = data.view_ids[k];
			my_draw_box(raws[view], data.dets[k].box, Eigen::Vector3i(0, 255, 0));
			//cv::imshow("box", raws[view]);
			//cv::waitKey();
			//return 0; 
		}
		for (int i = 0; i < 10; i++)
		{
			writers[i].write(raws[i]);
		}
	}
	for (int i = 0; i < 10; i++)
	{
		writers[i].release(); 
	}
	exit(-1);
	return 0; 
}