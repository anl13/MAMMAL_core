#include "framesolver.h"

#include "matching.h"
#include "tracking.h"
#include "../utils/timer_util.h" 

void FrameSolver::configByJson(std::string jsonfile) 
{
	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(jsonfile);
	if (!instream.is_open())
	{
		std::cout << "can not open " << jsonfile << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}
	m_sequence = root["sequence"].asString();
	m_keypointsDir = m_sequence + "/keypoints_hrnet_pr/";
	m_imgDir = m_sequence + "/images/";
	m_boxDir = m_sequence + "/boxes_pr/";
	m_maskDir = m_sequence + "/masks_pr/";
	m_camDir = root["camfolder"].asString();
	m_imgExtension = root["imgExtension"].asString();
	m_startid = root["startid"].asInt();
	m_framenum = root["framenum"].asInt();
	m_epi_thres = root["epipolar_threshold"].asDouble();
	m_epi_type = root["epipolartype"].asString();
	m_boxExpandRatio = root["box_expand_ratio"].asDouble();
	m_skelType = root["skel_type"].asString();
	m_topo = getSkelTopoByType(m_skelType);
	m_match_alg = root["match_alg"].asString();
	m_pigConfig = root["pig_config"].asString();
	m_use_gpu = root["use_gpu"].asBool();
	m_solve_sil_iters = root["solve_sil_iters"].asInt();

	std::vector<int> camids;
	for (auto const &c : root["camids"])
	{
		int id = c.asInt();
		camids.push_back(id);
	}
	setCamIds(camids);

	instream.close();
	mp_sceneData = std::make_shared<SceneData>(); 
}


void FrameSolver::reproject_skels()
{
	m_projs.clear();
	int pig_num = m_clusters.size();
	pig_num = pig_num > 4 ? 4 : pig_num;
	m_projs.resize(m_camNum);
	for (int c = 0; c < m_camNum; c++) m_projs[c].resize(pig_num);

	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int id = 0; id < pig_num; id++)
		{
			m_projs[camid][id].resize(m_topo.joint_num, Eigen::Vector3f::Zero());
			for (int kpt_id = 0; kpt_id < m_topo.joint_num; kpt_id++)
			{
				if (m_skels3d[id][kpt_id].norm() == 0) continue;
				Eigen::Vector3f p = m_skels3d[id][kpt_id];
				m_projs[camid][id][kpt_id] = project(m_camsUndist[camid], p);
			}
		}
	}
}

cv::Mat FrameSolver::visualizeSkels2D()
{
	vector<cv::Mat> imgdata;
	cloneImgs(m_imgsUndist, imgdata);
	for (int i = 0; i < m_camNum; i++)
	{
		for (int k = 0; k < m_detUndist[i].size(); k++)
		{
			drawSkelMonoColor(imgdata[i], m_detUndist[i][k].keypoints, k, m_topo);
			Eigen::Vector3i color = m_CM[k];
			my_draw_box(imgdata[i], m_detUndist[i][k].box, color);
			my_draw_mask(imgdata[i], m_detUndist[i][k].mask, color, 0.5);
		}
	}
	cv::Mat output;
	packImgBlock(imgdata, output);

	return output;
}

cv::Mat FrameSolver::visualizeIdentity2D(int viewid, int vid)
{
	cloneImgs(m_imgsUndist, m_imgsDetect);

	for (int id = 0; id < 4; id++)
	{
		if (vid >= 0 && id != vid)continue;
		for (int i = 0; i < m_matched[id].view_ids.size(); i++)
		{
			Eigen::Vector3i color;
			color(0) = m_CM[id](2);
			color(1) = m_CM[id](1);
			color(2) = m_CM[id](0);
			int camid = m_matched[id].view_ids[i];
			//int candid = m_matched[id].cand_ids[i];
			//if(candid < 0) continue; 
			if (m_matched[id].dets[i].keypoints.size() > 0)
				drawSkelMonoColor(m_imgsDetect[camid], m_matched[id].dets[i].keypoints, id, m_topo);
			my_draw_box(m_imgsDetect[camid], m_matched[id].dets[i].box, color);

			if (m_matched[id].dets[i].mask.size() > 0)
				my_draw_mask(m_imgsDetect[camid], m_matched[id].dets[i].mask, color, 0.5);
		}
	}
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int i = 0; i < m_unmatched[camid].size(); i++)
		{
			Eigen::Vector3i color;
			color(0) = m_CM[5](2);
			color(1) = m_CM[5](1);
			color(2) = m_CM[5](0);
			if (m_unmatched[camid][i].keypoints.size() > 0)
				drawSkelMonoColor(m_imgsDetect[camid], m_unmatched[camid][i].keypoints, 5, m_topo);
			my_draw_box(m_imgsDetect[camid], m_unmatched[camid][i].box, color);
			if (m_unmatched[camid][i].mask.size() > 0)
				my_draw_mask(m_imgsDetect[camid], m_unmatched[camid][i].mask, color, 0.5);
		}
	}
	if (viewid < 0)
	{
		cv::Mat packed;
		packImgBlock(m_imgsDetect, packed);
		return packed;
	}
	else
	{
		if (viewid >= m_camNum)
		{
			return m_imgsDetect[0];
		}
		else {
			return m_imgsDetect[viewid];
		}
	}
}

cv::Mat FrameSolver::visualizeProj()
{
	std::vector<cv::Mat> imgdata;
	cloneImgs(m_imgsUndist, imgdata);
	reproject_skels();

	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int id = 0; id < m_projs[camid].size(); id++)
		{
			drawSkelMonoColor(imgdata[camid], m_projs[camid][id], id, m_topo);
		}
	}

	cv::Mat packed;
	packImgBlock(imgdata, packed);
	return packed;
}

void FrameSolver::writeSkel3DtoJson(std::string jsonfile)
{
	std::ofstream os;
	os.open(jsonfile);
	if (!os.is_open())
	{
		std::cout << "file " << jsonfile << " cannot open" << std::endl;
		return;
	}

	Json::Value root;
	Json::Value pigs(Json::arrayValue);
	for (int index = 0; index < m_skels3d.size(); index++)
	{
		Json::Value pose(Json::arrayValue);
		for (int i = 0; i < m_topo.joint_num; i++)
		{
			// if a joint is empty, it is (0,0,0)^T
			pose.append(Json::Value(m_skels3d[index][i](0)));
			pose.append(Json::Value(m_skels3d[index][i](1)));
			pose.append(Json::Value(m_skels3d[index][i](2)));
		}
		pigs.append(pose);
	}
	root["pigs"] = pigs;

	// Json::StyledWriter stylewriter; 
	// os << stylewriter.write(root); 
	Json::StreamWriterBuilder builder;
	builder["commentStyle"] = "None";
	builder["indentation"] = "    ";
	std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
	writer->write(root, &os);
	os.close();
}

void FrameSolver::readSkel3DfromJson(std::string jsonfile)
{
	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(jsonfile);
	if (!instream.is_open())
	{
		std::cout << "can not open " << jsonfile << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}

	m_skels3d.clear();
	for (auto const &pig : root["pigs"])
	{
		std::vector<Eigen::Vector3f> a_pig;
		a_pig.resize(m_topo.joint_num);
		for (int index = 0; index < m_topo.joint_num; index++)
		{
			double x = pig[index * 3 + 0].asDouble();
			double y = pig[index * 3 + 1].asDouble();
			double z = pig[index * 3 + 2].asDouble();
			Eigen::Vector3f vec(x, y, z);
			a_pig[index] = vec;
		}
		m_skels3d.push_back(a_pig);
	}
	instream.close();
	std::cout << "read " << jsonfile << " done. " << std::endl;
}


int FrameSolver::_compareSkel(const std::vector<Eigen::Vector3f>& skel1, const std::vector<Eigen::Vector3f>& skel2)
{
	int overlay = 0;
	for (int i = 0; i < m_topo.joint_num; i++)
	{
		Eigen::Vector3f p1 = skel1[i];
		Eigen::Vector3f p2 = skel2[i];
		if (p1(2) < m_topo.kpt_conf_thresh[i] || p2(2) < m_topo.kpt_conf_thresh[i])continue;
		Eigen::Vector2f diff = p1.segment<2>(0) - p2.segment<2>(0);
		float dist = diff.norm();
		if (dist < 10) overlay++;
	}
	return overlay;
}
int FrameSolver::_countValid(const std::vector<Eigen::Vector3f>& skel)
{
	int valid = 0;
	for (int i = 0; i < skel.size(); i++)
	{
		if (skel[i](2) >= m_topo.kpt_conf_thresh[i]) valid++;
	}
	return valid;
}


void FrameSolver::detNMS()
{
	// cornor case
	if (m_detUndist.size() == 0) return;

	// discard some ones with large overlap 
	for (int camid = 0; camid < m_camNum; camid++)
	{
		// do nms on each view 
		int cand_num = m_detUndist[camid].size();
		std::vector<int> is_discard(cand_num, 0);
		for (int i = 0; i < cand_num; i++)
		{
			for (int j = i + 1; j < cand_num; j++)
			{
				if (is_discard[i] > 0 || is_discard[j] > 0) continue;
				int overlay = _compareSkel(m_detUndist[camid][i].keypoints,
					m_detUndist[camid][i].keypoints);
				int validi = _countValid(m_detUndist[camid][i].keypoints);
				int validj = _countValid(m_detUndist[camid][j].keypoints);
				float iou, iou1, iou2;
				IoU_xyxy_ratio(m_detUndist[camid][i].box, m_detUndist[camid][j].box,
					iou, iou1, iou2);
				if (overlay >= 3 && (iou1 > 0.8 || iou2 > 0.8))
				{
					if (validi > validj && iou2 > 0.8) is_discard[j] = 1;
					else if (validi<validj && iou1 > 0.8) is_discard[i] = 1;
				}
			}
		}
		std::vector<DetInstance> clean_dets;
		for (int i = 0; i < cand_num; i++)
		{
			if (is_discard[i] > 0) continue;
			clean_dets.push_back(m_detUndist[camid][i]);
		}
		m_detUndist[camid] = clean_dets;
	}

	// clean leg joints 
	drawRawMaskImgs();
	std::vector<std::pair<int, int> > legs = {
		{7,9},{5,7},
	{8,10},{6,8},
	{13,15},{11,13},
	{14,16},{12,14}
	};
	std::vector<int> leg_up = { 5,6,11,12 };
	std::vector<int> leg_middle = { 7,8,13,14 };
	std::vector<int> leg_down = { 9,10,15,16 };
	std::vector<int> all_legs = { 5,6,11,12 ,7,8,13,14,9,10,15,16 };
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int candid = 0; candid < m_detUndist[camid].size(); candid++)
		{
			// remove those out of box 
			for (int i = 0; i < all_legs.size(); i++)
			{
				DetInstance& det = m_detUndist[camid][candid];
				Eigen::Vector3f& point = det.keypoints[all_legs[i]];
				Eigen::Vector2f uv = point.segment<2>(0);
				if (!in_box_test(uv, det.box))
				{
					point(2) = 0; continue;
				}
				int idcode = 1 << candid;
				int x = int(round(uv(0)));
				int y = int(round(uv(1)));
				if (m_rawMaskImgs[camid].at<uchar>(y, x) != idcode) {
					point(2) = 0; continue;
				}
			}
			// remove those bones that cross over background. 
			for (int i = 0; i < legs.size(); i++)
			{
				int p1_index = legs[i].first;
				int p2_index = legs[i].second;
				Eigen::Vector3f& p1 = m_detUndist[camid][candid].keypoints[p1_index];
				Eigen::Vector3f& p2 = m_detUndist[camid][candid].keypoints[p2_index];
				if (p1(2) > m_topo.kpt_conf_thresh[p1_index] &&
					p2(2) > m_topo.kpt_conf_thresh[p2_index])
				{
					int n = 20;
					double dx = (p2(0) - p1(0)) / 20;
					double dy = (p2(1) - p1(1)) / 20;
					for (int k = 1; k < 20; k++)
					{
						int x = int(round(p1(0) + dx * k));
						int y = int(round(p1(1) + dy * k));
						if (m_rawMaskImgs[camid].at<uchar>(y, x) == 0)
						{
							p2(2) = 0; break;
						}
					}
				}
			}
		}
	}

}



void FrameSolver::tracking() // naive 3d 2 3d tracking
{
	if (m_frameid == m_startid) {
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
	for (int i = 0; i < map.size(); i++)
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

void FrameSolver::solve_parametric_model()
{
	if (mp_bodysolverdevice.empty()) mp_bodysolverdevice.resize(4);
	for (int i = 0; i < 4; i++)
	{
		if (mp_bodysolverdevice[i] == nullptr)
		{
			mp_bodysolverdevice[i] = std::make_shared<PigSolverDevice>(m_pigConfig);
			mp_bodysolverdevice[i]->setCameras(m_camsUndist);
			//mp_bodysolver[i]->InitNodeAndWarpField();
			mp_bodysolverdevice[i]->setRenderer(mp_renderEngine);
			mp_bodysolverdevice[i]->m_undist_mask_chamfer = mp_sceneData->m_undist_mask_chamfer;
			mp_bodysolverdevice[i]->m_scene_mask_chamfer = mp_sceneData->m_scene_mask_chamfer;
			mp_bodysolverdevice[i]->m_pig_id = i;
			std::cout << "init model " << i << std::endl;
		}
	}

	m_skels3d.resize(4);
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]);
		mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist;
		//if( (m_frameid - m_startid) % 25 == 0) // update scale parameter every  seconds 
		mp_bodysolverdevice[i]->globalAlign();
		setConstDataToSolver(i);
		mp_bodysolverdevice[i]->optimizePose();
		TimerUtil::Timer<std::chrono::milliseconds> tt;
		if (m_solve_sil_iters > 0)
		{
			if (i < 4) {
				//std::vector<ROIdescripter> rois;
				//getROI(rois, i);
				//mp_bodysolverdevice[i]->setROIs(rois);

				tt.Start();
				mp_bodysolverdevice[i]->optimizePoseSilhouette(m_solve_sil_iters);
				std::cout << "solve sil elapsed: " << tt.Elapsed() << " ms" << std::endl;

			}
		}
		mp_bodysolverdevice[i]->postProcessing();

		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host();
	}
}

//void FrameSolver::solve_parametric_model_cpu()
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

void FrameSolver::save_parametric_data()
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

void FrameSolver::read_parametric_data()
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
			mp_bodysolverdevice[i]->m_undist_mask_chamfer = mp_sceneData->m_undist_mask_chamfer;
			mp_bodysolverdevice[i]->m_scene_mask_chamfer = mp_sceneData->m_scene_mask_chamfer;
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
		mp_bodysolverdevice[i]->postProcessing();
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host();
	}
}

void FrameSolver::matching_by_tracking()
{
	m_skels3d_last = m_skels3d;

	// get m_clusters
	EpipolarMatching matcher;
	matcher.set_cams(m_camsUndist);
	matcher.set_dets(m_detUndist);
	matcher.set_epi_thres(m_epi_thres);
	matcher.set_epi_type(m_epi_type);
	matcher.set_topo(m_topo);
	if (m_frameid == m_startid || m_match_alg == "match")
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
		be_matched[camid].resize(m_detUndist[camid].size(), false);
	}
	for (int i = 0; i < m_clusters.size(); i++)
	{
		m_matched[i].view_ids.clear();
		m_matched[i].dets.clear();
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

void FrameSolver::pureTracking()
{
	m_skels3d_last = m_skels3d;
	std::vector<std::vector<std::vector<Eigen::Vector3f> > > skels2d;
	skels2d.resize(4);
	for (int i = 0; i < 4; i++)
	{
		skels2d[i] = mp_bodysolverdevice[i]->getSkelsProj();
	}
	m_clusters.clear();
	m_clusters.resize(4);
	for (int pid = 0; pid < 4; pid++)m_clusters[pid].resize(m_camNum, -1);

	for (int camid = 0; camid < m_camNum; camid++)
	{
		Eigen::MatrixXf sim;
		int boxnum = m_detUndist[camid].size();
		sim.resize(4, boxnum);
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < boxnum; j++)
			{

				float dist = distSkel2DTo2D(skels2d[i][camid],
					m_detUndist[camid][j].keypoints,
					m_topo);
				sim(i, j) = dist;
			}

		}
		std::vector<int> mm = solveHungarian(sim);

		for (int i = 0; i < 4; i++)
		{
			if (mm[i] >= 0)
			{
				int candid = mm[i];
				if (sim(i, candid) > 100000)continue;
				else m_clusters[i][camid] = candid;
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
		m_matched[i].view_ids.clear();
		m_matched[i].dets.clear();
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

void FrameSolver::save_clusters()
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

void FrameSolver::load_clusters()
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
	for (int i = 0; i < 4; i++)
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
vector<cv::Mat> FrameSolver::drawMask()
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

void FrameSolver::drawRawMaskImgs()
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

void FrameSolver::getROI(std::vector<ROIdescripter>& rois, int id)
{
	std::vector<cv::Mat> masks = drawMask();
	rois.resize(m_matched[id].view_ids.size());
	for (int view = 0; view < m_matched[id].view_ids.size(); view++)
	{
		int camid = m_matched[id].view_ids[view];
		rois[view].setId(id);
		rois[view].setT(m_frameid);
		rois[view].viewid = camid;
		rois[view].setCam(m_camsUndist[camid]);
		rois[view].mask_list = m_matched[id].dets[view].mask;
		rois[view].mask_norm = m_matched[id].dets[view].mask_norm;
		rois[view].keypoints = m_matched[id].dets[view].keypoints;
		cv::Mat mask;
		mask.create(cv::Size(m_imw, m_imh), CV_8UC1);
		my_draw_mask_gray(mask,
			m_matched[id].dets[view].mask, 255);
		rois[view].area = cv::countNonZero(mask);

		rois[view].chamfer = computeSDF2d(mask);

		rois[view].mask = masks[camid];
		rois[view].box = m_matched[id].dets[view].box;
		rois[view].undist_mask = mp_sceneData->m_undist_mask; // valid area for image distortion 
		rois[view].scene_mask = mp_sceneData->m_scene_masks[camid];
		rois[view].pid = id;
		rois[view].idcode = 1 << id;
		rois[view].valid = rois[view].keypointsMaskOverlay();
		computeGradient(rois[view].chamfer, rois[view].gradx, rois[view].grady);
	}
}

void FrameSolver::setConstDataToSolver(int id)
{
	assert((id >= 0 && id <= 3));

	if (mp_bodysolverdevice[id] == nullptr)
	{
		std::cout << "solver is empty! " << std::endl;
		exit(-1);
	}

	if (!mp_bodysolverdevice[id]->init_backgrounds)
	{
		for (int i = 0; i < 10; i++)
			cudaMemcpy(mp_bodysolverdevice[id]->d_const_scene_mask[i], 
				mp_sceneData->m_scene_masks[i].data,
				1920 * 1080 * sizeof(uchar), cudaMemcpyHostToDevice);
		cudaMemcpy(mp_bodysolverdevice[id]->d_const_distort_mask, 
			mp_sceneData->m_undist_mask.data,
			1920 * 1080 * sizeof(uchar), cudaMemcpyHostToDevice);
		mp_bodysolverdevice[id]->init_backgrounds = true;
	}
	mp_bodysolverdevice[id]->m_pig_id = id;
	mp_bodysolverdevice[id]->m_det_masks = drawMask();

}
