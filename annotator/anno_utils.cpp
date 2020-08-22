#include "anno_utils.h"

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