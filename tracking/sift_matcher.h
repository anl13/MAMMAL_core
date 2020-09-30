#pragma once
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

std::vector<std::pair<int, int> > sift_match(
	const std::vector<cv::KeyPoint> &key1, const std::vector<cv::KeyPoint>& key2,
	const cv::Mat& des1, const cv::Mat& des2
);

void draw_sift_matches(
	const cv::Mat& img1, const cv::Mat& img2,
	const std::vector<cv::KeyPoint>& key1, const std::vector<cv::KeyPoint>& key2,
	const std::vector<std::pair<int, int> > &pairs, cv::Mat& output
);

void clean_bfmatches(
	const std::vector<cv::KeyPoint>& key1, const std::vector<cv::KeyPoint>& key2,
	const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& output
);

void draw_sift_matches_same_color(
	const cv::Mat& img1, const cv::Mat& img2,
	const std::vector<cv::KeyPoint>& key1, const std::vector<cv::KeyPoint>& key2,
	const std::vector<cv::DMatch>& matches, cv::Mat& output
);

void draw_sift_matches_overlay(
	const cv::Mat& img1, const cv::Mat& img2,
	const std::vector<cv::KeyPoint>& key1, const std::vector<cv::KeyPoint>& key2,
	const std::vector<cv::DMatch>& matches, cv::Mat& output
);