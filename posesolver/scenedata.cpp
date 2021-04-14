#include "scenedata.h"
#include <json/json.h>
#include "../utils/fileoperation.h"

SceneData::SceneData(std::string camfolder,
	std::string backgroundfolder,
	std::string scenemaskpath,
	std::vector<int> camids)
{
	m_camDir = camfolder;
	m_camids = camids;
	m_camNum = m_camids.size(); 
	readBackground(backgroundfolder); 
	readSceneMask(scenemaskpath); 
}

void SceneData::readBackground(std::string backgroundfolder)
{
	m_backgrounds.clear();
	for (int camid = 0; camid < m_camNum; camid++)
	{
		std::stringstream ss_bg;
		ss_bg << backgroundfolder << "/bg" << m_camids[camid] << "_undist";
		std::string bg = ss_bg.str();
		cv::Mat img2;
		if (IsFileExistent(bg + ".png"))
			img2 = cv::imread(bg + ".png");
		else if (IsFileExistent(bg + ".jpg"))
			img2 = cv::imread(bg + ".jpg"); 
		if (img2.empty())
		{
			std::cout << "can not open background image" << ss_bg.str() << std::endl;
			system("pause");
			exit(-1);
		}
		m_backgrounds.push_back(img2);
	}

	std::stringstream ss;
	ss << backgroundfolder << "undist_mask.png";
	m_undist_mask = cv::imread(ss.str());
	if (m_undist_mask.empty())
	{
		std::cout << "can not open undist mask " << ss.str() << std::endl;
		system("pause");
		exit(-1);
	}
	cv::cvtColor(m_undist_mask, m_undist_mask, cv::COLOR_BGR2GRAY);
	m_undist_mask_chamfer = get_dist_trans(m_undist_mask);

}


void SceneData::readSceneMask(std::string scenemaskpath)
{
	m_scene_masks.resize(m_camNum);

	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(scenemaskpath);
	if (!instream.is_open())
	{
		std::cout << "can not open " << scenemaskpath << std::endl;
		system("pause"); 
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		system("pause");
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
		//cv::imwrite("G:/pig_results/scenemask" + std::to_string(camid) + ".png", sceneimage); 
		m_scene_masks[i] = sceneimage;
	}

	m_scene_mask_chamfer.resize(m_camNum);
	for (int i = 0; i < m_camNum; i++)
	{
		m_scene_mask_chamfer[i] = get_dist_trans(255 - m_scene_masks[i]);
	}
}