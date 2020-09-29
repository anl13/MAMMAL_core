#include "scenedata.h"
#include <json/json.h>

SceneData::SceneData()
{
	m_camDir = "D:/Projects/animal_calib/data/calibdata/adjust/";
	m_camids = { 0,1,2,5,6,7,8,9,10,11 };
	m_camNum = m_camids.size(); 
	readBackground(); 
	readSceneMask(); 
}

void SceneData::readBackground()
{
	m_backgrounds.clear();
	for (int camid = 0; camid < m_camNum; camid++)
	{
		std::stringstream ss_bg;
		ss_bg << m_camDir << "../backgrounds/bg" << m_camids[camid] << "_undist.png";
		cv::Mat img2 = cv::imread(ss_bg.str());
		if (img2.empty())
		{
			std::cout << "can not open background image" << ss_bg.str() << std::endl;
			exit(-1);
		}
		m_backgrounds.push_back(img2);
	}

	std::stringstream ss;
	ss << m_camDir << "../backgrounds/undist_mask.png";
	m_undist_mask = cv::imread(ss.str());
	if (m_undist_mask.empty())
	{
		std::cout << "can not open undist mask " << ss.str() << std::endl;
		exit(-1);
	}
	cv::cvtColor(m_undist_mask, m_undist_mask, cv::COLOR_BGR2GRAY);
	m_undist_mask_chamfer = get_dist_trans(m_undist_mask);

}


void SceneData::readSceneMask()
{
	m_scene_masks.resize(m_camNum);
	std::string scenejson = "D:/Projects/animal_calib/data/scenemask.json";

	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(scenejson);
	if (!instream.is_open())
	{
		std::cout << "can not open " << scenejson << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}

	for (int i = 0; i < m_camNum; i++)
	{
		int camid = m_camids[i];
		std::string key = std::to_string(camid);
		Json::Value c = root[key];
		int id_num = c.size();
		vector<vector<Eigen::Vector2f> > masks;
		for (int bid = 0; bid < id_num; bid++)
		{
			//if(bid >= 4) break; // remain only 4 top boxes 
			Json::Value mask_parts = c[bid]["points"];
			std::vector<Eigen::Vector2f> mask;
			for (auto m : mask_parts)
			{
				double x = m[0].asDouble();
				double y = m[1].asDouble();
				mask.emplace_back(Eigen::Vector2f((float)x, (float)y));
			}
			masks.push_back(mask);
		}
		cv::Mat sceneimage(cv::Size(1920, 1080), CV_8UC1);
		sceneimage.setTo(0);
		my_draw_mask_gray(sceneimage, masks, 255);
		m_scene_masks[i] = sceneimage;
	}

	m_scene_mask_chamfer.resize(m_camNum);
	for (int i = 0; i < m_camNum; i++)
	{
		m_scene_mask_chamfer[i] = get_dist_trans(255 - m_scene_masks[i]);
	}
}