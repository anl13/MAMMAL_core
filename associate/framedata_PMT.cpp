#include "framedata.h"
#include "matching.h"
#include "tracking.h" 
#include <sstream> 

void FrameData::matching()
{
    vector<vector<vector<Eigen::Vector3d> > > keypoints; 
    keypoints.resize(m_camNum); 
    for(int camid = 0; camid < m_camNum; camid++)
    {
        int candnum = m_detUndist[camid].size(); 
        keypoints[camid].resize(candnum); 
        for(int candid = 0; candid < candnum; candid++)
        {
            keypoints[camid][candid] = m_detUndist[camid][candid].keypoints; 
        }
    }
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

void FrameData::tracking()
{
    if(m_frameid == m_startid) {
        m_skels3d_last = m_skels3d; 
        return; 
    }
    NaiveTracker m_tracker; 
    m_tracker.set_skels_curr(m_skels3d); 
    m_tracker.set_skels_last(m_skels3d_last); 
    m_tracker.track(); 
    m_skels3d = m_tracker.get_skels_curr_track(); 

    m_skels3d_last = m_skels3d; 

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

	for (int i = 0; i < 4; i++)
	{
		std::cout << "solving ... " << i << std::endl; 
		mp_bodysolver[i]->setSource(m_matched[i]); 
		mp_bodysolver[i]->normalizeSource(); 
		mp_bodysolver[i]->globalAlign(); 
		mp_bodysolver[i]->optimizePose(100, 0.001); 
		mp_bodysolver[i]->computePivot();
		std::string savefolder = "E:/pig_results/"; 
		std::stringstream ss; 
		ss << savefolder << "state_" << i << "_" <<
			std::setw(6) << std::setfill('0') << m_frameid
			<< ".pig";
		auto body = mp_bodysolver[i]->getBodyState(); 
		body.saveState(ss.str()); 
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