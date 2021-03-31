#include "../posesolver/framedata.h"
#include "../posesolver/scenedata.h" 
#include "sift_matcher.h"
#include <sstream>
#include <json/json.h>
#include <iostream>
#include <fstream> 

using std::vector;

int detect_sift()
{
	FrameData frame;
	frame.configByJson("D:/Projects/animal_calib/tracking/track.json");
	vector<vector<cv::KeyPoint> > keys; 
	vector<cv::Mat> dess; 
	keys.resize(10); 
	dess.resize(10); 
	for (int index = 0; index < 4000; index++)
	{
		frame.set_frame_id(index);
		frame.fetchData();

		for (int camid = 0; camid < 10; camid++)
		{
			cv::Mat mask(cv::Size(1920, 1080), CV_8UC1);
			for (int i = 0; i < frame.m_detUndist[camid].size(); i++)
				my_draw_mask_gray(mask, frame.m_detUndist[camid][i].mask, 255);
			cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
			cv::dilate(mask, mask, cv::Mat());
			cv::dilate(mask, mask, cv::Mat());
			cv::dilate(mask, mask, cv::Mat());


			cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
			std::vector<cv::KeyPoint> key1;
			cv::Mat des1;
			cv::Mat img1 = frame.m_imgsUndist[0];
			siftPtr->detectAndCompute(img1, mask, key1, des1);
			keys[camid] = key1; 
			dess[camid] = des1;
		}

		std::stringstream ss; 
		ss << "Z:/sequences/20190704_noon/sift/sift" << std::setw(10) << std::setfill('0')
			<< index << ".txt"; 
		saveSIFTKeypoints(ss.str(), keys, dess);
		std::cout << "detect " << index << std::endl; 
	}

	return 0; 
}

int track_sift()
{
	vector<vector<cv::KeyPoint> > keys;
	vector<cv::Mat> dess;
	vector<vector<cv::KeyPoint> > keys_last; 
	vector<cv::Mat> dess_last;
	keys.resize(10);
	dess.resize(10);
	cv::FlannBasedMatcher matcher;

	for (int frameid = 0; frameid < 4000; frameid++)
	{
		std::stringstream ss;
		ss << "Z:/sequences/20190704_noon/sift/sift" << std::setw(10) << std::setfill('0')
			<< frameid << ".txt";
		readSIFTKeypoints(ss.str(), keys, dess); 
		if (frameid == 0)
		{
			keys_last = keys;
			dess_last = dess; 
			continue; 
		}
		vector<vector<cv::DMatch> > all_matches;
		all_matches.resize(10);
		for (int camid = 0; camid < 10; camid++)
		{
			//cv::BFMatcher matcher;
			vector<cv::DMatch> matches;
			matcher.match(dess_last[camid], dess[camid], matches);
			std::vector<cv::DMatch> cleaned;
			clean_bfmatches(keys_last[camid], keys[camid], matches, cleaned);
			all_matches[camid] = matches; 
		}
		std::stringstream ss1;
		ss1 << "Z:/sequences/20190704_noon/sift/match" << std::setw(10) << std::setfill('0')
			<< frameid << ".txt";
		saveSIFTMatches(ss1.str(), all_matches); 
		keys_last = keys; 
		dess_last = dess; 
		std::cout << "track " << frameid << std::endl; 
	}
	return 0; 
}

int main()
{
	detect_sift(); 
	track_sift(); 
	return 0; 

#if 0
	FrameData frame; 
	frame.configByJson("track.json");
	FrameData frame2;
	frame2.configByJson("track.json");
	cv::FlannBasedMatcher matcher;

	for (int index = 0; index < 1; index++)
	{
		std::cout << "index: " << index << std::endl; 
		frame.set_frame_id(index);
		frame.fetchData();

		frame2.set_frame_id(index+1);
		frame2.fetchData();

		vector<vector<cv::KeyPoint> > keys1; 
		vector<cv::Mat> des1;
		vector<vector<cv::KeyPoint> > keys2;
		vector<cv::Mat> des2;
		readSIFTKeypoints("Z:/sequences/20190704_noon/sift/sift0000000000.txt", keys1, des1);
		readSIFTKeypoints("Z:/sequences/20190704_noon/sift/sift0000000001.txt", keys2, des2);
		//readSIFTMatches("Z:/sequences/20190704_noon/sift/match0000000001.txt", matches);

		vector<vector<cv::DMatch> > all_matches;
		all_matches.resize(10);
		for (int camid = 0; camid < 10; camid++)
		{
			//cv::BFMatcher matcher;
			vector<cv::DMatch> matches;
			matcher.match(keys1[camid], des1[camid], matches);
	/*		std::vector<cv::DMatch> cleaned;
			clean_bfmatches(keys1[camid], keys2[camid], matches, cleaned);*/
			all_matches[camid] = matches;
		}

		//cv::Mat img1 = frame.get_imgs_undist()[0];
		//cv::Mat img2 = frame2.get_imgs_undist()[0];
		//cv::Mat img_match;
		//draw_sift_matches_same_color(img1, img2, keys1[0], keys2[0], all_matches[0], img_match); 
		//cv::Mat img_match_overlay;
		//draw_sift_matches_overlay(img1, img2, keys1[0], keys2[0], all_matches[0], img_match_overlay); 

		//std::stringstream ss; 
		//ss << "D:/results/seq_noon/flow/sift" << std::setw(6) << std::setfill('0') << index << ".jpg"; 
		//cv::imwrite(ss.str(), img_match_overlay); 
	}
#endif 
	return 0; 
}