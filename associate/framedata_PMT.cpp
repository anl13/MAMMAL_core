#include "framedata.h"
#include "matching.h"
#include "tracking.h" 
#include <sstream> 
#include "../utils/timer_util.h"

void FrameData::tracking() // naive 3d 2 3d tracking
{
    if(m_frameid == m_startid) {
        m_skels3d_last = m_skels3d; 
        return; 
    }
    NaiveTracker m_tracker; 
    m_tracker.set_skels_last(m_skels3d_last); 
	m_tracker.m_cameras = m_camsUndist;
	m_tracker.m_topo = m_topo; 
    m_tracker.track(m_matched); 

    vector<int> map = m_tracker.get_map(); 
    vector<MatchedInstance> rematch;
	vector<vector<int>> recluster(4); 
    rematch.resize(m_matched.size()); 
    for(int i = 0; i < map.size(); i++)
    {
        int id = map[i];
		if (id > -1)
		{
			rematch[i] = m_matched[id];
			recluster[i] = m_clusters[id];
		}
    }
    m_matched = rematch;
	m_clusters = recluster;
	m_skels3d_last = m_skels3d; 
}

void FrameData::solve_parametric_model()
{
	if(mp_bodysolverdevice.empty()) mp_bodysolverdevice.resize(4);
	for (int i = 0; i < 4; i++)
	{
		if (mp_bodysolverdevice[i] == nullptr)
		{
			mp_bodysolverdevice[i] = std::make_shared<PigSolverDevice>(m_pigConfig); 
			mp_bodysolverdevice[i]->setCameras(m_camsUndist);
			//mp_bodysolver[i]->InitNodeAndWarpField();
			mp_bodysolverdevice[i]->setRenderer(mp_renderEngine);
			std::cout << "init model " << i << std::endl; 
		}
	}

	m_skels3d.resize(4); 
	for (int i = 0; i <4; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]); 
		mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist; 
		if( (m_frameid - m_startid) % 25 == 0) // update scale parameter every  seconds 
			mp_bodysolverdevice[i]->globalAlign();
		setConstDataToSolver(i);
		mp_bodysolverdevice[i]->optimizePose(); 
		mp_bodysolverdevice[i]->m_pig_id = i; 
		TimerUtil::Timer<std::chrono::milliseconds> tt; 
		if (i < 4) {
			//std::vector<ROIdescripter> rois;
			//getROI(rois, i);
			//mp_bodysolverdevice[i]->setROIs(rois);
			
			tt.Start();
			mp_bodysolverdevice[i]->optimizePoseSilhouette(m_solve_sil_iters);
			std::cout << "solve sil elapsed: " << tt.Elapsed() << " ms" << std::endl;

		}

		std::vector<Eigen::Vector3f> skels = mp_bodysolverdevice[i]->getRegressedSkel_host();
		m_skels3d[i] = skels; 
	}
}

//void FrameData::solve_parametric_model_cpu()
//{
//	if (mp_bodysolver.empty()) mp_bodysolver.resize(4);
//	for (int i = 0; i < 4; i++)
//	{
//		if (mp_bodysolver[i] == nullptr)
//		{
//			mp_bodysolver[i] = std::make_shared<PigSolver>(m_pigConfig);
//			mp_bodysolver[i]->setCameras(m_camsUndist);
//			mp_bodysolver[i]->normalizeCamera();
//			mp_bodysolver[i]->mp_renderEngine = (mp_renderEngine);
//			std::cout << "init model " << i << std::endl;
//		}
//	}
//
//	m_skels3d.resize(4);
//	for (int i = 0; i < 1; i++)
//	{
//		mp_bodysolver[i]->setSource(m_matched[i]);
//		mp_bodysolver[i]->normalizeSource();
//		mp_bodysolver[i]->globalAlign();
//		mp_bodysolver[i]->optimizePose();
//
//		if (i < 1) {
//			std::vector<ROIdescripter> rois;
//			getROI(rois, 0);
//			mp_bodysolver[i]->m_rois = rois;
//			mp_bodysolver[i]->optimizePoseSilhouette(1);
//		}
//
//		Eigen::MatrixXf skel_eigen = mp_bodysolver[i]->getRegressedSkel();
//		std::vector<Eigen::Vector3f> skels = convertMatToVec(skel_eigen);
//		m_skels3d[i] = skels;
//	}
//}

void FrameData::save_parametric_data()
{
	for (int i = 0; i < 4; i++)
	{
		std::string savefolder = result_folder + "/state";
		if (is_smth) savefolder = savefolder + "_smth";
		std::stringstream ss;
		ss << savefolder << "/pig_" << i << "_frame_" <<
			std::setw(6) << std::setfill('0') << m_frameid
			<< ".txt";
		mp_bodysolverdevice[i]->saveState(ss.str()); 
	}
}

void FrameData::read_parametric_data()
{
	if (mp_bodysolverdevice.empty()) mp_bodysolverdevice.resize(4);
	m_skels3d.resize(4); 
	for (int i = 0; i < 4; i++)
	{
		if (mp_bodysolverdevice[i] == nullptr)
		{
			mp_bodysolverdevice[i] = std::make_shared<PigSolverDevice>(m_pigConfig);
			mp_bodysolverdevice[i]->setCameras(m_camsUndist);
			mp_bodysolverdevice[i]->setRenderer(mp_renderEngine);
			mp_bodysolverdevice[i]->m_pig_id = i; 
			std::cout << "init model " << i << std::endl;
		}
	}

	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]);
		mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist; 
		std::string savefolder = result_folder + "/state";
		if (is_smth) savefolder = savefolder + "_smth";
		std::stringstream ss;
		ss << savefolder << "/pig_" << i << "_frame_" <<
			std::setw(6) << std::setfill('0') << m_frameid
			<< ".txt";
		mp_bodysolverdevice[i]->readState(ss.str()); 
		mp_bodysolverdevice[i]->UpdateVertices();
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host(); 
	}
}

void FrameData::matching_by_tracking()
{
	m_skels3d_last = m_skels3d;

	// get m_clusters
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

	// post processing to get matched data
	m_matched.clear();
	m_matched.resize(m_clusters.size());
	vector<vector<bool> > be_matched;
	be_matched.resize(m_camNum);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		be_matched[camid].resize(m_detUndist[camid].size(),false);
	}
	for (int i = 0; i < m_clusters.size(); i++)
	{
		for (int camid = 0; camid < m_camNum; camid++)
		{
			int candid = m_clusters[i][camid];
			if (candid < 0) continue;
			be_matched[camid][candid] = true;
			m_matched[i].view_ids.push_back(camid);
			m_matched[i].dets.push_back(m_detUndist[camid][candid]);
		}
	}

	m_unmatched.clear();
	m_unmatched.resize(m_camNum);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int candid = 0; candid < be_matched[camid].size(); candid++)
		{
			if (!be_matched[camid][candid])
			{
				m_unmatched[camid].push_back(m_detUndist[camid][candid]);
			}
		}
	}

	// match between frames
	if (m_match_alg == "match")
	{
		tracking(); 
	}
}

void FrameData::pureTracking()
{
	m_skels3d_last = m_skels3d;
	m_clusters.resize(4);
	for (int pid = 0; pid < 4; pid++)m_clusters[pid].resize(m_camNum, -1); 

	for (int camid = 0; camid < m_camNum; camid++)
	{
		Eigen::MatrixXd sim; 
		int boxnum = m_boxes_processed[camid].size();
		sim.resize(4, boxnum);
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < boxnum; j++)
			{
				double center_conf = m_keypoints_undist[camid][j][20](2);
				if (center_conf < 0.3)
				{
					sim(i, j) = 10; continue; 
				}
				double iou = IoU_xyxy(m_projectedBoxesLast[i][camid], m_boxes_processed[camid][j]);
				if (iou < 0.5) sim(i, j) = 10;
				else sim(i, j) = 1 / iou;
			}
		}
		std::vector<int> mm = solveHungarian(sim);
		for (int i = 0; i < 4; i++)
		{
			if (mm[i] >= 0)
			{
				int candid = mm[i];
				if (sim(i, candid) == 10)continue; 
				m_clusters[i][camid] = candid;
			}
		}
	}

	// post processing to get matched data
	m_matched.clear();
	m_matched.resize(m_clusters.size());
	vector<vector<bool> > be_matched;
	be_matched.resize(m_camNum);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		be_matched[camid].resize(m_detUndist[camid].size(), false);
	}
	for (int i = 0; i < m_clusters.size(); i++)
	{
		for (int camid = 0; camid < m_camNum; camid++)
		{
			int candid = m_clusters[i][camid];
			if (candid < 0) continue;
			be_matched[camid][candid] = true;
			m_matched[i].view_ids.push_back(camid);
			m_matched[i].dets.push_back(m_detUndist[camid][candid]);
		}
	}

	m_unmatched.clear();
	m_unmatched.resize(m_camNum);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int candid = 0; candid < be_matched[camid].size(); candid++)
		{
			if (!be_matched[camid][candid])
			{
				m_unmatched[camid].push_back(m_detUndist[camid][candid]);
			}
		}
	}
}

void FrameData::save_clusters()
{
	std::stringstream ss; 
	ss << result_folder << "/clusters/" << std::setw(6) << std::setfill('0') << m_frameid << ".txt";
	std::ofstream stream(ss.str());
	if (!stream.is_open())
	{
		std::cout << "cluster saving stream not open. " << std::endl;
		return; 
	}
	for (int i = 0; i < m_clusters.size(); i++)
	{
		for (int k = 0; k < m_clusters[i].size(); k++)
		{
			stream << m_clusters[i][k] << " ";
		}
		stream << std::endl; 
	}
	stream.close(); 
}

void FrameData::load_clusters()
{
	std::stringstream ss;
	ss << result_folder << "/clusters/" << std::setw(6) << std::setfill('0') << m_frameid << ".txt";
	std::ifstream stream(ss.str());
	if (!stream.is_open())
	{
		std::cout << "cluster loading stream not open. " << std::endl;
		return;
	}
	m_clusters.resize(4); 
	for(int i = 0; i < 4; i++)
	{
		m_clusters[i].resize(m_camNum);
		for (int k = 0; k < m_camNum; k++)
		{
			stream >> m_clusters[i][k];
		}
	}
	stream.close(); 

	m_matched.clear();
	m_matched.resize(m_clusters.size());
	vector<vector<bool> > be_matched;
	be_matched.resize(m_camNum);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		be_matched[camid].resize(m_detUndist[camid].size(), false);
	}
	for (int i = 0; i < m_clusters.size(); i++)
	{
		for (int camid = 0; camid < m_camNum; camid++)
		{
			int candid = m_clusters[i][camid];
			if (candid < 0) continue;
			be_matched[camid][candid] = true;
			m_matched[i].view_ids.push_back(camid);
			m_matched[i].dets.push_back(m_detUndist[camid][candid]);
		}
	}

	m_unmatched.clear();
	m_unmatched.resize(m_camNum);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int candid = 0; candid < be_matched[camid].size(); candid++)
		{
			if (!be_matched[camid][candid])
			{
				m_unmatched[camid].push_back(m_detUndist[camid][candid]);
			}
		}
	}
}

// draw masks according to 
vector<cv::Mat> FrameData::drawMask()
{
	vector<cv::Mat> m_imgsMask; 
	m_imgsMask.resize(m_camNum);
	for (int i = 0; i < m_camNum; i++)
	{
		m_imgsMask[i].create(cv::Size(m_imw, m_imh), CV_8UC1);
	}
	for (int pid = 0; pid < 4; pid++)
	{
		cv::Mat temp(cv::Size(m_imw, m_imh), CV_8UC1);
		for (int k = 0; k < m_matched[pid].view_ids.size(); k++)
		{
			int viewid = m_matched[pid].view_ids[k];
			my_draw_mask_gray(temp, m_matched[pid].dets[k].mask, 1 << pid);
			m_imgsMask[viewid] = temp + m_imgsMask[viewid];
		}
	}
	return m_imgsMask;
}

void FrameData::drawRawMaskImgs()
{
	m_rawMaskImgs.resize(m_camNum);
	for (int i = 0; i < m_camNum; i++)
	{
		m_rawMaskImgs[i].create(cv::Size(m_imw, m_imh), CV_8UC1);
	}
	for (int camid = 0; camid < m_camNum; camid++)
	{
		cv::Mat temp(cv::Size(m_imw, m_imh), CV_8UC1);
		for (int k = 0; k < m_detUndist[camid].size(); k++)
		{
			my_draw_mask_gray(temp, m_detUndist[camid][k].mask, 1 << k);
			m_rawMaskImgs[camid] = temp + m_rawMaskImgs[camid];
		}
	}
}