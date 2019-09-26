#include "extract_utils.h" 

#include <iostream>
#include <iomanip>
#include <sstream> 
#include <vector>
#include <fstream> 
#include <opencv2/opencv.hpp>
//#include <boost/filesystem.hpp> 

using std::vector; 
//namespace fs = boost::filesystem; 

vector<vector<std::string>> get_video_lists(std::string folder, std::string date/*=20190704_THU*/)
{
	vector<vector<std::string>> video_paths; 
	video_paths.resize(12); 

	for (int camid = 0; camid < 12; camid++)
	{
		vector<int> video_ids;

		std::stringstream ss; 
		ss << folder << "/lists/cam" << camid << "/cam" << camid
			<< "_" << date << ".txt"; 
		std::string filename = ss.str(); 
		std::ifstream listfile(filename); 
		if (!listfile.is_open())
		{
			std::cout << "Can not open file " << filename << std::endl; 
			continue; 
		}
		while (!listfile.eof())
		{
			int vid; 
			listfile >> vid;
			video_ids.push_back(vid); 
		};
		if (video_ids.size() > 0)
		{
			vector<std::string> names; 
			for (int j = 0; j < video_ids.size(); j++)
			{
				std::stringstream  ss1; 
				ss1 << folder << "/cam" << camid << "/hiv" << std::setw(5) << std::setfill('0') << video_ids[j] << ".mp4"; 
				names.push_back(ss1.str()); 
			}
			video_paths[camid] = names; 
		}
	}

	return video_paths; 
}

double get_all_frame_num(const std::vector<std::vector<std::string>> &paths)
{
	double all_num = 0; 
	for (int camid = 0; camid < 12; camid++)
	{
		int video_num = paths[camid].size(); 
		for (int i = 0; i < video_num; i++)
		{
			cv::VideoCapture cap(paths[camid][i]); 
			if (!cap.isOpened())
			{
				std::cout << "Could not open video " << paths[camid][i] << std::endl; 
				exit(-1); 
			}
			double frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT); 
			cap.release(); 
			all_num += frame_num; 
		}
	}
	return all_num; 
}


double extract_frames(const std::vector<std::vector<std::string>> &paths, std::string save_folder)
{
	int total_frame_num = 0; 
	int extracted_frame_num = 0; 
	for (int camid = 0; camid < 12; camid++)
	{
		if (camid != 5) continue; 
		extracted_frame_num = 0; 
		int video_num = paths[camid].size(); 
		for (int i = 0; i < video_num; i++)
		{
			cv::VideoCapture cap(paths[camid][i]);
			if (!cap.isOpened())
			{
				std::cout << "Could not open video " << paths[camid][i] << std::endl;
				exit(-1);
			}

			double frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT); 
			double fps = cap.get(cv::CAP_PROP_FPS); 
			
			int frameid = 0; 
			int interval = 1; 
			for (;frameid < frame_count / interval;)
			{
				//cap.set(cv::CAP_PROP_POS_FRAMES, frameid * interval); 
				//cap.set(cv::CAP_PROP_POS_MSEC, frameid * 90 * 1000); 
				
				std::cout << "frameid " << frameid << std::endl; 
				cv::Mat frame; 
				cap.read(frame); 
				int den = 4500 / interval; 
				frameid++;
				if (frameid % den != 0) continue; 
				
				if (!frame.empty())
				{
					std::stringstream ss; 
					ss << save_folder << "/cam_" << camid << "_" << std::setw(8) << std::setfill('0') << extracted_frame_num << ".png";
					cv::imwrite(ss.str(), frame); 
					extracted_frame_num++;
					std::cout << "cam " << camid << " video " << i << "  frame " << frameid << "/" << frame_count / 6000 << std::endl; 
				}
				else {
					break; 
				}
			}
			cap.release();
			total_frame_num += extracted_frame_num; 
		}
	}

	return total_frame_num; 
}