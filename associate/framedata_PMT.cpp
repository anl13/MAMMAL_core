#include "framedata.h"
#include "matching.h"
#include "tracking.h" 
#include <sstream> 

void FrameData::matching()
{
	m_skels3d_last = m_skels3d; 

    EpipolarMatching matcher; 
    matcher.set_cams(m_camsUndist); 
    matcher.set_dets(m_detUndist); 
    matcher.set_epi_thres(m_epi_thres); 
    matcher.set_epi_type(m_epi_type); 
    matcher.set_topo(m_topo); 
    matcher.match(); // main match func 
    matcher.truncate(4); // retain only 4 clusters 
    matcher.get_clusters(m_clusters); 
    matcher.get_skels3d(m_skels3d); 
    
    m_matched.clear(); 
    m_matched.resize(m_clusters.size()); 
    for(int i = 0; i < m_clusters.size(); i++)
    {
        for(int camid = 0; camid < m_camNum; camid++)
        {
            int candid = m_clusters[i][camid]; 
            if(candid < 0) continue; 
            m_matched[i].view_ids.push_back(camid); 
            m_matched[i].cand_ids.push_back(candid); 
            m_matched[i].dets.push_back(m_detUndist[camid][candid]); 
        }
    }
}

void FrameData::tracking() // naive 3d 2 3d tracking
{
    if(m_frameid == m_startid) {
        m_skels3d_last = m_skels3d; 
        return; 
    }
    NaiveTracker m_tracker; 
    m_tracker.set_skels_curr(m_skels3d); 
    m_tracker.set_skels_last(m_skels3d_last); 
    m_tracker.track(); 

    vector<int> map = m_tracker.get_map(); 
    vector<MatchedInstance> rematch;
    rematch.resize(m_matched.size()); 
    for(int i = 0; i < map.size(); i++)
    {
        int id = map[i];
        if(id>-1)
        rematch[i] = m_matched[id]; 
    }
    m_matched = rematch;
}

void FrameData::solve_parametric_model()
{
	m_smalDir = "D:/Projects/animal_calib/data/pig_model/"; 
	if(mp_bodysolver.empty()) mp_bodysolver.resize(4);
	for (int i = 0; i < 4; i++)
	{
		if (mp_bodysolver[i] == nullptr)
		{
			mp_bodysolver[i] = std::make_shared<PigSolver>(m_smalDir);
			mp_bodysolver[i]->setMapper(getPigMapper()); 
			mp_bodysolver[i]->setCameras(m_camsUndist);
			mp_bodysolver[i]->normalizeCamera();
			mp_bodysolver[i]->setId(i); 
			std::cout << "init model " << i << std::endl; 
		}
	}

	m_skels3d.resize(4); 
	for (int i = 0; i < 4; i++)
	{
		std::cout << "solving ... " << i << std::endl; 
		mp_bodysolver[i]->setSource(m_matched[i]); 
		mp_bodysolver[i]->normalizeSource(); 
		mp_bodysolver[i]->globalAlign(); 
		mp_bodysolver[i]->optimizePose(100, 0.001); 
		mp_bodysolver[i]->computePivot();
		//std::string savefolder = "E:/pig_results/"; 
		//std::stringstream ss; 
		//ss << savefolder << "state_" << i << "_" <<
		//	std::setw(6) << std::setfill('0') << m_frameid
		//	<< ".pig";
		//auto body = mp_bodysolver[i]->getBodyState(); 
		//body.saveState(ss.str()); 

		auto skels = mp_bodysolver[i]->getRegressedSkel();
		m_skels3d[i] = convertMatToVec(skels); 
	}
}

void FrameData::read_parametric_data()
{
	m_smalDir = "D:/Projects/animal_calib/data/pig_model/";
	if (mp_bodysolver.empty()) mp_bodysolver.resize(4);
	for (int i = 0; i < 4; i++)
	{
		if (mp_bodysolver[i] == nullptr)
		{
			mp_bodysolver[i] = std::make_shared<PigSolver>(m_smalDir);
			mp_bodysolver[i]->setMapper(getPigMapper());
			mp_bodysolver[i]->setCameras(m_camsUndist);
			mp_bodysolver[i]->normalizeCamera();
			mp_bodysolver[i]->setId(i);
			std::cout << "init model " << i << std::endl;
		}
	}

	for (int i = 0; i < 4; i++)
	{
		std::string savefolder = "E:/pig_results/";
		std::stringstream ss;
		ss << savefolder << "state_" << i << "_" <<
			std::setw(6) << std::setfill('0') << m_frameid
			<< ".pig";
		mp_bodysolver[i]->readBodyState(ss.str()); 
	}

}

void FrameData::matching_by_tracking()
{
	m_skels3d_last = m_skels3d;

	EpipolarMatching matcher;
	matcher.set_cams(m_camsUndist);
	matcher.set_dets(m_detUndist);
	matcher.set_epi_thres(m_epi_thres);
	matcher.set_epi_type(m_epi_type);
	matcher.set_topo(m_topo);
	if (m_frameid == m_startid || m_match_alg=="match")
	{
		matcher.match(); 
		matcher.truncate(4); // retain only 4 clusters 
	}
	else
	{
		matcher.set_skels_t_1(m_skels3d_last);
		matcher.match_by_tracking();
	}
	matcher.get_clusters(m_clusters);
	matcher.get_skels3d(m_skels3d);

	m_matched.clear();
	m_matched.resize(m_clusters.size());
	for (int i = 0; i < m_clusters.size(); i++)
	{
		for (int camid = 0; camid < m_camNum; camid++)
		{
			int candid = m_clusters[i][camid];
			if (candid < 0) continue;
			m_matched[i].view_ids.push_back(camid);
			m_matched[i].cand_ids.push_back(candid);
			m_matched[i].dets.push_back(m_detUndist[camid][candid]);
		}
	}
	if (m_match_alg == "match")
	{
		tracking(); 
	}
}

void FrameData::debug_fitting(int pig_id)
{
	visualizeDebug(pig_id); 
	std::vector<cv::Mat> crop_list; 
	for (int i = 0; i < m_matched[pig_id].dets.size(); i++)
	{
		Eigen::Vector4d box = m_matched[pig_id].dets[i].box;
		int view_id = m_matched[pig_id].view_ids[i];
		cv::Mat raw_img = m_imgsDetect[view_id];
		cv::Rect2i roi(box[0], box[1], box[2] - box[0], box[3] - box[1]);
		cv::Mat img = raw_img(roi);
		cv::Mat img2 = resizeAndPadding(img, 256, 256);
		crop_list.push_back(img2); 
	}
	cv::Mat output; 
	packImgBlock(crop_list, output);
	std::stringstream ss; 
	ss << "E:/debug_pig/debug/output" << m_frameid << "_" << pig_id << ".png";
	cv::imwrite(ss.str(), output); 
	//cv::imshow("test", output); 
	//cv::waitKey(); 
	return; 
}

void FrameData::visualizeDebug(int pid)
{
	cloneImgs(m_imgsUndist, m_imgsDetect);

	for (int id = 0; id < m_matched.size(); id++)
	{
		if (pid >= 0 && id != pid) continue;
		for (int i = 0; i < m_matched[id].view_ids.size(); i++)
		{
			int camid = m_matched[id].view_ids[i];
			int candid = m_matched[id].cand_ids[i];
			if (candid < 0) continue;
			drawSkelDebug(m_imgsDetect[camid], m_detUndist[camid][candid].keypoints);
			my_draw_box(m_imgsDetect[camid], m_detUndist[camid][candid].box, m_CM[id]);
			//my_draw_mask(m_imgsDetect[camid], m_detUndist[camid][candid].mask, m_CM[id], 0.5);
		}
	}
}