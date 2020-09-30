#include "../posesolver/framedata.h"
#include "../posesolver/scenedata.h" 
#include "sift_matcher.h"

int main()
{
	FrameData frame; 
	frame.configByJson("track.json");
	FrameData frame2;
	frame2.configByJson("track.json");
	
	for (int index = 2600; index < 9000; index++)
	{
		frame.set_frame_id(index);
		frame.fetchData();

		cv::Mat mask(cv::Size(1920, 1080), CV_8UC1);
		for (int i = 0; i < frame.m_detUndist[0].size(); i++)
			my_draw_mask_gray(mask, frame.m_detUndist[0][i].mask, 255);



		cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
		std::vector<cv::KeyPoint> key1;
		cv::Mat des1;
		cv::Mat img1 = frame.m_imgsUndist[0];
		siftPtr->detectAndCompute(img1, mask, key1, des1);


		frame2.set_frame_id(index+1);
		frame2.fetchData();
		cv::Mat mask2(cv::Size(1920, 1080), CV_8UC1);
		for (int i = 0; i < frame2.m_detUndist[0].size(); i++)
			my_draw_mask_gray(mask2, frame2.m_detUndist[0][i].mask, 255);
		std::vector<cv::KeyPoint> key2;
		cv::Mat des2;
		cv::Mat img2 = frame2.m_imgsUndist[0];
		cv::Ptr<cv::SIFT> siftPtr2 = cv::SIFT::create();
		siftPtr2->detectAndCompute(img2, mask2, key2, des2);
		std::cout << "des2.rows: " << des2.rows << " des2.cols: " << des2.cols << std::endl;


		//std::vector<std::pair<int,int> > pairs = sift_match(key1, key2, des1, des2); 
		//cv::Mat output; 
		//draw_sift_matches(img1, img2, key1, key2, pairs, output); 
		

		std::vector<cv::DMatch> matches;
		
		cv::FlannBasedMatcher matcher; 
		//cv::BFMatcher matcher;
		matcher.match(des1, des2, matches);
		std::vector<cv::DMatch> cleaned; 
		clean_bfmatches(key1, key2, matches, cleaned); 
		cv::Mat img_match;
		//cv::drawMatches(img1, key1, img2, key2, cleaned, img_match);
		draw_sift_matches_same_color(img1, img2, key1, key2, cleaned, img_match); 
		cv::Mat img_match_overlay;
		draw_sift_matches_overlay(img1, img2, key1, key2, cleaned, img_match_overlay); 
		//cv::Mat output;
		//cv::drawKeypoints(img1, key1, output);

		cv::namedWindow("sift", cv::WINDOW_NORMAL);
		cv::imshow("sift", img_match);
		cv::namedWindow("sift_overlay", cv::WINDOW_NORMAL); 
		cv::imshow("sift_overlay", img_match_overlay); 
		int key = cv::waitKey();
		if (key == 27) break; 
	}
	cv::destroyAllWindows();


	return 0; 
}