#include "sift_matcher.h"
#include <opencv2/core/eigen.hpp> 
#include <vector>
#include "../utils/Hungarian.h"
#include "../utils/image_utils.h"

std::vector<std::pair<int, int> > sift_match(
	const std::vector<cv::KeyPoint> &key1, const std::vector<cv::KeyPoint>& key2,
	const cv::Mat& des1, const cv::Mat& des2
)
{
	Eigen::MatrixXf feature1; 
	cv::cv2eigen(des1, feature1); 
	Eigen::MatrixXf feature2;
	cv::cv2eigen(des2, feature2); 

	std::vector<Eigen::Vector2f> keypoints1; 
	keypoints1.resize(key1.size()); 
	for (int i = 0; i < key1.size(); i++)
	{
		keypoints1[i](0) = key1[i].pt.x; 
		keypoints1[i](1) = key1[i].pt.y;
	}
	std::vector<Eigen::Vector2f> keypoints2;
	keypoints2.resize(key2.size());
	for (int i = 0; i < key2.size(); i++)
	{
		keypoints1[i](0) = key2[i].pt.x;
		keypoints1[i](1) = key2[i].pt.y;
	}

	int N1 = key1.size(); 
	int N2 = key2.size(); 
	Eigen::MatrixXf sim = Eigen::MatrixXf::Zero(N1, N2); 
	for (int i = 0; i < N1; i++)
	{
		for (int j = 0; j < N2; j++)
		{
			float geo_dist = (keypoints1[i] - keypoints2[j]).norm(); 
			float semantic_dist = (feature1.row(i) - feature2.row(j)).norm(); 
			if (geo_dist > 50) sim(i, j) = 100000;
			else sim(i, j) = semantic_dist;
		}
	}

	std::vector<int> pairs = solveHungarian(sim);

	std::vector<std::pair<int, int> > output; 
	for (int i = 0; i < pairs.size(); i++)
	{
		if (pairs[i] > 0 && sim(i,pairs[i] < 100000)) output.push_back({ i,pairs[i] });
	}
	return output; 

}

void draw_sift_matches(
	const cv::Mat& img1, const cv::Mat& img2,
	const std::vector<cv::KeyPoint>& key1, const std::vector<cv::KeyPoint>& key2,
	const std::vector<std::pair<int, int> > &pairs, cv::Mat& output
)
{
	std::vector<Eigen::Vector3i> CM = getColorMapEigen("rapidtable");
	int H = img1.rows;
	int W = img1.cols;
	output.create(cv::Size(W * 2, H), img1.type()); 
	img1.copyTo(output(cv::Rect(0, 0, W, H)));
	img2.copyTo(output(cv::Rect(W, 0, W, H)));
	for (int i = 0; i < key1.size(); i++)
	{
		int colorindex = i % CM.size(); 
		cv::Scalar color(CM[colorindex](2), CM[colorindex](1), CM[colorindex](0));
		cv::circle(output, key1[i].pt, 5, color,2);
	}

	for (int j = 0; j < key2.size(); j++)
	{
		int colorindex = j % CM.size(); 
		cv::Scalar color(CM[colorindex](2), CM[colorindex](1), CM[colorindex](0));
		cv::Point2f p;
		p.x = key2[j].pt.x + W; 
		p.y = key2[j].pt.y;
		cv::circle(output, p, 5, color, 2);
	}

	for (int i = 0; i < pairs.size(); i++)
	{
		int a = pairs[i].first; 
		int b = pairs[i].second; 
		cv::Point2f p1 = key1[a].pt;
		cv::Point2f p2 = key2[b].pt;
		p2.x += W; 
		int colorid = a % CM.size(); 
		cv::Scalar color(CM[colorid](2), CM[colorid](1), CM[colorid](1)); 
		cv::line(output, p1, p2, color, 2);
	}
}

void clean_bfmatches(
	const std::vector<cv::KeyPoint>& key1, const std::vector<cv::KeyPoint>& key2,
	const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& output,
	float thresh
)
{
	output.clear(); 
	for (int i = 0; i < matches.size(); i++)
	{
		int a = matches[i].queryIdx;
		int b = matches[i].trainIdx; 
		float dist = (key1[a].pt.x - key2[b].pt.x) * (key1[a].pt.x - key2[b].pt.x) +
			(key1[a].pt.y - key2[b].pt.y)*(key1[a].pt.y - key2[b].pt.y);
		dist = sqrtf(dist); 
		if (dist < thresh) output.push_back(matches[i]); 
	}
}

void draw_sift_matches_same_color(
	const cv::Mat& img1, const cv::Mat& img2,
	const std::vector<cv::KeyPoint>& key1, const std::vector<cv::KeyPoint>& key2,
	const std::vector<cv::DMatch>& matches, cv::Mat& output
)
{
	std::vector<Eigen::Vector3i> CM = getColorMapEigen("anliang_rgb");
	int H = img1.rows;
	int W = img1.cols;
	output.create(cv::Size(W * 2, H), img1.type());
	img1.copyTo(output(cv::Rect(0, 0, W, H)));
	img2.copyTo(output(cv::Rect(W, 0, W, H)));
	for (int i = 0; i < key1.size(); i++)
	{
		int colorindex = i % CM.size();
		cv::Scalar color(CM[colorindex](2), CM[colorindex](1), CM[colorindex](0));
		cv::circle(output, key1[i].pt, 4, color, 1);
	}

	for (int j = 0; j < key2.size(); j++)
	{
		int colorindex = j % CM.size();
		cv::Scalar color(CM[colorindex](2), CM[colorindex](1), CM[colorindex](0));
		cv::Point2f p;
		p.x = key2[j].pt.x + W;
		p.y = key2[j].pt.y;
		cv::circle(output, p, 4, color, 1);
	}

	for (int i = 0; i < matches.size(); i++)
	{
		int a = matches[i].queryIdx;
		int b = matches[i].trainIdx;
		cv::Point2f p1 = key1[a].pt;
		cv::Point2f p2 = key2[b].pt;
		p2.x += W;
		int colorid = i % CM.size();
		cv::Scalar color(CM[colorid](2), CM[colorid](1), CM[colorid](0));
		cv::circle(output, p1, 4, color, 1.5);
		cv::circle(output, p2, 4, color, 1.5); 
		cv::line(output, p1, p2, color, 1.5);
	}
}


void draw_sift_matches_overlay(
	const cv::Mat& img1, const cv::Mat& img2,
	const std::vector<cv::KeyPoint>& key1, const std::vector<cv::KeyPoint>& key2,
	const std::vector<cv::DMatch>& matches, cv::Mat& output
)
{
	std::vector<Eigen::Vector3i> CM = getColorMapEigen("anliang_rgb");
	int H = img1.rows;
	int W = img1.cols;
	output = (img1 + img2) / 2; 
	//for (int i = 0; i < key1.size(); i++)
	//{
	//	int colorindex = i % CM.size();
	//	cv::Scalar color(CM[colorindex](2), CM[colorindex](1), CM[colorindex](0));
	//	cv::circle(output, key1[i].pt, 4, color, 1);
	//}

	//for (int j = 0; j < key2.size(); j++)
	//{
	//	int colorindex = j % CM.size();
	//	cv::Scalar color(CM[colorindex](2), CM[colorindex](1), CM[colorindex](0));
	//	cv::Point2f p;
	//	p.x = key2[j].pt.x + W;
	//	p.y = key2[j].pt.y;
	//	cv::circle(output, p, 4, color, 1);
	//}

	for (int i = 0; i < matches.size(); i++)
	{
		int a = matches[i].queryIdx;
		int b = matches[i].trainIdx;
		cv::Point2f p1 = key1[a].pt;
		cv::Point2f p2 = key2[b].pt;
		int colorid = i % CM.size();
		cv::Scalar color(CM[colorid](2), CM[colorid](1), CM[colorid](0));
		cv::circle(output, p1, 4, color, -1);
		cv::circle(output, p2, 4, color, 1.5);
		cv::line(output, p1, p2, color, 1.5);
	}
}

