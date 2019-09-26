#pragma once

#include <string>
#include <vector>

std::vector<std::vector<std::string>> get_video_lists(std::string folder, std::string date/*=20190704_THU*/);

double get_all_frame_num(const std::vector<std::vector<std::string>> &paths); 

double extract_frames(const std::vector<std::vector<std::string>> &paths, std::string save_folder); 