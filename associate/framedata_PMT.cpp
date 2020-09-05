#include "framedata.h"
#include "matching.h"
#include "tracking.h" 
#include <sstream> 
#include "../utils/timer_util.h"

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
            //m_matched[i].cand_ids.push_back(candid); 
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
}

void FrameData::solve_parametric_model()
{

	if(mp_bodysolverdevice.empty()) mp_bodysolverdevice.resize(4);
	for (int i = 0; i < 1; i++)
	{
		if (mp_bodysolverdevice[i] == nullptr)
		{
			mp_bodysolverdevice[i] = std::make_shared<PigSolverDevice>(m_pigConfig);
			mp_bodysolverdevice[i]->debug();
			system("pause");
			exit(-1); 
			mp_bodysolverdevice[i]->setCameras(m_camsUndist);
			mp_bodysolverdevice[i]->normalizeCamera();
			//mp_bodysolver[i]->InitNodeAndWarpField();
			mp_bodysolverdevice[i]->setRenderer(mp_renderEngine);
			std::cout << "init model " << i << std::endl; 
		}
	}

	m_skels3d.resize(4); 
	for (int i = 0; i < 1; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]); 
		mp_bodysolverdevice[i]->normalizeSource();
		mp_bodysolverdevice[i]->globalAlign();

		TimerUtil::Timer<std::chrono::milliseconds> timer;
		timer.Start(); 
		mp_bodysolverdevice[i]->optimizePose(); 
		std::cout << "solve pose " << timer.Elapsed() << "  ms" << std::endl; 
		std::cout << "debug: " << std::endl; 
		
		if (i < 1) {
			timer.Start(); 
			std::vector<ROIdescripter> rois;
			getROI(rois, 0);
			std::cout << "get roi: " << timer.Elapsed() << " ms" << std::endl; 

			mp_bodysolverdevice[i]->setROIs(rois);
			timer.Start(); 
			
			mp_bodysolverdevice[i]->optimizePoseSilhouette(1);
			std::cout << "solve by sil gpu: " << timer.Elapsed() << " ms" << std::endl; 
		}

		std::vector<Eigen::Vector3f> skels = mp_bodysolverdevice[i]->getRegressedSkel_host();
		m_skels3d[i] = skels; 
	}
}

void FrameData::solve_parametric_model_cpu()
{
	if (mp_bodysolver.empty()) mp_bodysolver.resize(4);
	for (int i = 0; i < 4; i++)
	{
		if (mp_bodysolver[i] == nullptr)
		{
			mp_bodysolver[i] = std::make_shared<PigSolver>(m_pigConfig);
			mp_bodysolver[i]->setCameras(m_camsUndist);
			mp_bodysolver[i]->normalizeCamera();
			mp_bodysolver[i]->mp_renderEngine = (mp_renderEngine);
			std::cout << "init model " << i << std::endl;
		}
	}

	m_skels3d.resize(4);
	for (int i = 0; i < 1; i++)
	{
		mp_bodysolver[i]->setSource(m_matched[i]);
		mp_bodysolver[i]->normalizeSource();
		mp_bodysolver[i]->globalAlign();

		TimerUtil::Timer<std::chrono::milliseconds> timer;
		timer.Start();
		mp_bodysolver[i]->optimizePose();
		std::cout << "solve pose " << timer.Elapsed() << "  ms" << std::endl;

		if (i < 1) {
			timer.Start();
			std::vector<ROIdescripter> rois;
			getROI(rois, 0);
			std::cout << "get roi: " << timer.Elapsed() << " ms" << std::endl;

			mp_bodysolver[i]->m_rois = rois;
			timer.Start();
			mp_bodysolver[i]->optimizePoseSilhouette(18);
			std::cout << "solve by sil cpu: " << timer.Elapsed() << " ms"  << std::endl;
		}

		Eigen::MatrixXf skel_eigen = mp_bodysolver[i]->getRegressedSkel();
		std::vector<Eigen::Vector3f> skels = convertMatToVec(skel_eigen);
		m_skels3d[i] = skels;
	}
}

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
	for (int i = 0; i < 4; i++)
	{
		if (mp_bodysolverdevice[i] == nullptr)
		{
			mp_bodysolverdevice[i] = std::make_shared<PigSolverDevice>(m_pigConfig);
			mp_bodysolverdevice[i]->setCameras(m_camsUndist);
			mp_bodysolverdevice[i]->normalizeCamera();
			std::cout << "init model " << i << std::endl;
		}
	}

	for (int i = 0; i < 4; i++)
	{
		std::string savefolder = result_folder + "/state";
		if (is_smth) savefolder = savefolder + "_smth";
		std::stringstream ss;
		ss << savefolder << "/pig_" << i << "_frame_" <<
			std::setw(6) << std::setfill('0') << m_frameid
			<< ".txt";
		mp_bodysolverdevice[i]->readState(ss.str()); 
		mp_bodysolverdevice[i]->UpdateVertices();
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

void FrameData::debug_fitting(int pig_id)
{
	visualizeDebug(pig_id); 
	std::vector<cv::Mat> crop_list; 
	for (int i = 0; i < m_matched[pig_id].dets.size(); i++)
	{
		Eigen::Vector4f box = m_matched[pig_id].dets[i].box;
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
	ss << result_folder << "/fitting/" << pig_id << "/output" << m_frameid << "_" << pig_id << ".png";
	cv::imwrite(ss.str(), output); 
	//cv::imshow("test", output); 
	//cv::waitKey(); 
	return; 
}

void FrameData::debug_chamfer(int pid)
{
	// init gray images 
	std::vector<cv::Mat> grays; 
	grays.resize(10);
	for (int i = 0; i < 10; i++)
	{
		grays[i].create(cv::Size(1920, 1080), CV_8UC1);
	}
	// create chamfer 
	for (int i = 0; i < m_matched[pid].view_ids.size(); i++)
	{
		int camid = m_matched[pid].view_ids[i];
		//int candid = m_matched[pid].cand_ids[i];
		//if (candid < 0) continue;
		my_draw_mask_gray(grays[camid],
			//m_detUndist[camid][candid].mask, 255);
			m_matched[pid].dets[i].mask, 255);
		cv::Mat chamfer = get_dist_trans(grays[camid]);
		cv::namedWindow("mask"); 
		int key = cv::waitKey(); 
		if (key == 27)exit(-1); 
	}
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
			//int candid = m_matched[id].cand_ids[i];
			//if (candid < 0) continue;
			drawSkelDebug(m_imgsDetect[camid], m_matched[id].dets[i].keypoints);
			my_draw_box(m_imgsDetect[camid], m_matched[id].dets[i].box, m_CM[id]);
			//my_draw_mask(m_imgsDetect[camid], m_matched[id].dets[i].mask, m_CM[id], 0.5);
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

void FrameData::getChamferMap(int pid, int viewid,
	cv::Mat& chamfer)
{
	cv::Mat inner, outer;
	// innner
	cv::Mat mask;
	mask.create(cv::Size(m_imw, m_imh), CV_8UC1); 
	my_draw_mask_gray(mask,
		m_matched[pid].dets[viewid].mask, 255);
	inner = get_dist_trans(mask);
	// outer 
	cv::Mat mask2;
	mask2.create(cv::Size(m_imw, m_imh), CV_8UC1);
	mask2.setTo(cv::Scalar(255));
	my_draw_mask_gray(mask, m_matched[pid].dets[viewid].mask, 0);
	outer = get_dist_trans(mask2);
	// final chamfer as sdf
	chamfer = inner - outer;

	return; 
}