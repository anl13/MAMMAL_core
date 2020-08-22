#include "state_annotator.h"
#include <Eigen/Eigen>
#include <vector> 
#include <json/json.h>

std::vector<std::pair<std::string, Eigen::Vector2i> > state_labels = {
	{ "lie_side",{ 0,0 } },
{ "lie_down",{ 0,1 } },
{ "stand_four_feet",{ 0,2 } },
{ "stand_two_feet",{ 0,3 } },
{ "walk", {0,4}}, 

{"next", {1,0}}
};


void StateAnnotator::construct_panel_attr()
{
	getColorMap("anliang_rgb", m_CM);
	int cols = 5;
	int rows = 5;
	int cellw = 250;
	int cellh = 50;
	m_panel_attr.create(cv::Size(cellw*cols, cellh*rows), CV_8UC3);

	for (int i = 0; i < state_labels.size(); i++)
	{
		int row = state_labels[i].second(0);
		int col = state_labels[i].second(1);
		int x = col * cellw;
		int y = row * cellh;

		drawLabel(m_panel_attr, x, y, cellw, cellh, state_labels[i].first, false);
	}
}



void StateAnnotator::show_panel()
{
	cv::namedWindow("panel", cv::WINDOW_AUTOSIZE);
	//cv::setMouseCallback("panel", CallBackFuncPanel, &status);

	cv::namedWindow("image", cv::WINDOW_KEEPRATIO);
	//cv::setMouseCallback("image", CallBackImage, &labeldata);

	construct_panel_attr(); 
	cv::imshow("panel", m_panel_attr); 
	while (true)
	{
		char c = cv::waitKey(50);
		if (c == 27)
		{
			break;
		}
		else if (c == 's' || c == 'S')
		{
			//save_label_result();
		}
		else if (c == 'r' || c == 'R')
		{
			//read_label_result();
		}
	}
}