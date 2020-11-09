#include "framesolver.h"

#include "matching.h"
#include "tracking.h"
#include "../utils/timer_util.h" 

//#define VIS_ASSOC_STEP 

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
	m_camids = camids;
	m_camNum = m_camids.size();

	instream.close();
	mp_sceneData = std::make_shared<SceneData>();

	d_interDepth.resize(m_camNum); 
	int H = WINDOW_HEIGHT;
	int W = WINDOW_WIDTH;
	for (int i = 0; i < m_camNum; i++)
	{
		cudaMalloc((void**)&d_interDepth[i], H * W * sizeof(float));
	}
}
FrameSolver::~FrameSolver()
{
	for (int i = 0; i < m_camNum; i++)
	{
		cudaFree(d_interDepth[i]); 
	}

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
	m_skels3d.resize(4);
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]);
		mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist;
		mp_bodysolverdevice[i]->globalAlign();
		setConstDataToSolver(i);
		mp_bodysolverdevice[i]->optimizePose();
	}

	if (m_solve_sil_iters > 0)
	{
		optimizeSil(m_solve_sil_iters); 
	}

	for(int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->postProcessing();
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host();
	}

	// postprocess
	m_last_matched = m_matched; 
}


void FrameSolver::solve_parametric_model_pipeline2()
{
	m_skels3d.resize(4);
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]);
		mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist;
		mp_bodysolverdevice[i]->globalAlign();
		setConstDataToSolver(i);
		
		std::vector<ROIdescripter> rois;
		getROI(rois, i);
		mp_bodysolverdevice[i]->setROIs(rois);
		mp_bodysolverdevice[i]->searchAnchorSpace();
		mp_bodysolverdevice[i]->optimizeAnchor(mp_bodysolverdevice[i]->m_anchor_id);

		//mp_bodysolverdevice[i]->optimizePoseWithAnchor();
	}

	if (m_solve_sil_iters > 0)
	{
		optimizeSilWithAnchor(m_solve_sil_iters);
	}

	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->postProcessing();
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host();
	}

	// postprocess
	m_last_matched = m_matched;
}

// This pipeline only search for best anchor point
void FrameSolver::solve_parametric_model_pipeline3()
{
	init_parametric_solver(); 

	m_skels3d.resize(4);
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]);
		mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist;
		mp_bodysolverdevice[i]->globalAlign();
		setConstDataToSolver(i);
		// mask are necessary for measure anchor point
		std::vector<ROIdescripter> rois;
		getROI(rois, i);
		mp_bodysolverdevice[i]->setROIs(rois);
		mp_bodysolverdevice[i]->searchAnchorSpace();
		mp_bodysolverdevice[i]->optimizeAnchor(mp_bodysolverdevice[i]->m_anchor_id);
	}

	// postprocess
	m_last_matched = m_matched;

	for (int i = 0; i < 4; i++)
	{
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
	init_parametric_solver(); 

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
			m_matched[i].candids.push_back(candid); 
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
			m_matched[i].candids.push_back(candid); 
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
		for (int k = 0; k < m_matched[pid].view_ids.size(); k++)
		{
			cv::Mat temp(cv::Size(m_imw, m_imh), CV_8UC1);

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
		rois[view].binary_mask = mask; 

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

	if (m_use_gpu)
	{
		if (!mp_bodysolverdevice[id]->init_backgrounds)
		{
			for (int i = 0; i < m_camNum; i++)
				cudaMemcpy(mp_bodysolverdevice[id]->d_const_scene_mask[i],
					mp_sceneData->m_scene_masks[i].data,
					1920 * 1080 * sizeof(uchar), cudaMemcpyHostToDevice);
			cudaMemcpy(mp_bodysolverdevice[id]->d_const_distort_mask,
				mp_sceneData->m_undist_mask.data,
				1920 * 1080 * sizeof(uchar), cudaMemcpyHostToDevice);
			mp_bodysolverdevice[id]->init_backgrounds = true;
			mp_bodysolverdevice[id]->c_const_scene_mask = mp_sceneData->m_scene_masks;
			mp_bodysolverdevice[id]->c_const_distort_mask = mp_sceneData->m_undist_mask;
		}
	}
	else {
		if (!mp_bodysolverdevice[id]->init_backgrounds)
		{
			mp_bodysolverdevice[id]->c_const_scene_mask = mp_sceneData->m_scene_masks;
			mp_bodysolverdevice[id]->c_const_distort_mask = mp_sceneData->m_undist_mask; 
			mp_bodysolverdevice[id]->init_backgrounds = true; 
		}
	}
	mp_bodysolverdevice[id]->m_pig_id = id;
	mp_bodysolverdevice[id]->m_det_masks = drawMask();

	// mask necessary for measure anchor point
	std::vector<ROIdescripter> rois;
	getROI(rois, id);
	mp_bodysolverdevice[id]->setROIs(rois);
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

	float threshold = 500; 
	//renderInteractDepth(true); 
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

				int viewid = find_in_list(camid, m_last_matched[i].view_ids);
				float dist2;
				if (viewid < 0) dist2 = dist; 
				else
				{
					dist2 = distSkel2DTo2D(m_last_matched[i].dets[viewid].keypoints,
						m_detUndist[camid][j].keypoints,
						m_topo); // calc last 2D detection to current detection
				}
				sim(i, j) = dist + dist2;

				if (sim(i, j) > threshold) sim(i, j) = threshold;
			}
		}

		//std::cout << "sim: " << camid << std::endl << sim << std::endl;

		std::vector<int> mm = solveHungarian(sim);

		//for (int i = 0; i < mm.size(); i++)
		//	std::cout << mm[i] << "  ";
		//std::cout << std::endl; 

		for (int i = 0; i < 4; i++)
		{
			if (mm[i] >= 0)
			{
				int candid = mm[i];
				if (sim(i, candid) >= threshold)continue;
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

void FrameSolver::init_parametric_solver()
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
			if (mp_bodysolverdevice[i]->m_use_gpu != m_use_gpu)
			{
				std::cout << "Sorry! please agree on use gpu or not. " << std::endl; 
				system("pause"); 
				exit(-1); 
			}
			std::cout << "init model " << i << std::endl;
		}
	}
	m_skels3d.resize(4); 
}


void FrameSolver::renderInteractDepth(bool withmask)
{
	if(withmask)
		if (m_interMask.size() != m_camNum) m_interMask.resize(m_camNum); 
	std::vector<Eigen::Vector3f> id_colors = {
		{1.0f, 0.0f,0.0f},
	{0.0f, 1.0f, 0.0f},
	{0.0f, 0.0f, 1.0f},
	{1.0f, 1.0f, 0.0f}
	};
	mp_renderEngine->clearAllObjs();
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->UpdateNormalFinal();
		RenderObjectColor* p_model = new RenderObjectColor();
		p_model->SetVertices(mp_bodysolverdevice[i]->GetVertices());
		p_model->SetFaces(mp_bodysolverdevice[i]->GetFacesVert());
		p_model->SetNormal(mp_bodysolverdevice[i]->GetNormals());
		p_model->SetColor(id_colors[i]);
		mp_renderEngine->colorObjs.push_back(p_model);
	}

	for (int view = 0; view < m_camNum; view++)
	{
		int camid = view; 
		Camera cam = m_camsUndist[camid];
		mp_renderEngine->s_camViewer.SetExtrinsic(cam.R, cam.T);

		float * depth_device = mp_renderEngine->renderDepthDevice();
		cudaMemcpy(d_interDepth[view], depth_device, 
			WINDOW_WIDTH*WINDOW_HEIGHT * sizeof(float),
			cudaMemcpyDeviceToDevice);
		
		if (withmask)
		{
			mp_renderEngine->Draw("mask"); 
			m_interMask[view] = mp_renderEngine->GetImage();
		}
	}

	mp_renderEngine->clearAllObjs();
	
	for (int pid = 0; pid < mp_bodysolverdevice.size(); pid++)
	{
		for (int i = 0; i < m_camNum; i++)
		{
			cudaMemcpy(mp_bodysolverdevice[pid]->d_depth_renders_interact[i],
				d_interDepth[i], WINDOW_WIDTH*WINDOW_HEIGHT * sizeof(float),
				cudaMemcpyDeviceToDevice);
		}
	}
}

void FrameSolver::optimizeSil(int maxIterTime)
{
	for (int pid = 0; pid < 4; pid++)
	{
		mp_bodysolverdevice[pid]->generateDataForSilSolver(); 
		if (!m_use_gpu)
		{
			std::vector<ROIdescripter> rois;
			getROI(rois, pid);
			mp_bodysolverdevice[pid]->setROIs(rois);
		}
	}

	int iter = 0; 
	for (; iter < maxIterTime; iter++)
	{
		renderInteractDepth(); 
		for (int pid = 0; pid < 4; pid++)
		{
			mp_bodysolverdevice[pid]->optimizePoseSilOneStep(iter); 
		}
	}
}

void FrameSolver::optimizeSilWithAnchor(int maxIterTime)
{
	for (int pid = 0; pid < 4; pid++)
	{
		mp_bodysolverdevice[pid]->generateDataForSilSolver();
	}

	int iter = 0;
	for (; iter < maxIterTime; iter++)
	{
		renderInteractDepth();
		for (int pid = 0; pid < 4; pid++)
		{
			mp_bodysolverdevice[pid]->optimizePoseSilWithAnchorOneStep(iter);
		}
	}
}

void FrameSolver::saveAnchors(std::string folder)
{
	std::stringstream ss; 
	ss << folder << "/anchor_" << std::setw(6) << std::setfill('0') << m_frameid <<
		".txt"; 
	std::ofstream outfile(ss.str()); 
	for(int i = 0; i < 4; i++)
	    outfile << mp_bodysolverdevice[i]->m_anchor_id << std::endl; 
	outfile.close(); 
}

void FrameSolver::loadAnchors(std::string folder, bool andsolve)
{
	DARKOV_Step1_setsource(); 
	DARKOV_Step2_loadanchor(); 

	if (andsolve)
	{
		DARKOV_Step2_optimanchor(); 
		DARKOV_Step4_fitrawsource(); 
		//DARKOV_Step3_reassoc_type2(); 
		//DARKOV_Step4_fitreassoc(); 
		DARKOV_Step5_postprocess(); 
	}
}

void FrameSolver::nmsKeypointCands(std::vector<Eigen::Vector3f>& list)
{
	std::vector<Eigen::Vector3f> raw = list; 
	list.clear(); 
	for (int i = 0; i < raw.size(); i++)
	{
		if (i == 0) list.push_back(raw[i]); 
		else
		{
			bool repeat = false; 
			for (int j = 0; j < list.size(); j++)
			{
				Eigen::Vector3f diff = list[j] - raw[i];
				float d = diff.norm(); 
				if (d < 30)
				{
					repeat = true;
					break; 
				}
			}
			if (!repeat) list.push_back(raw[i]);
		}
	}
}

// 20201103: split all detected keypoints without fine track
void FrameSolver::splitDetKeypoints()
{
	m_keypoints_pool.resize(m_camNum); 
	for (int view = 0; view < m_camNum; view++)
	{
		m_keypoints_pool[view].resize(m_topo.joint_num); 
		for (int candid = 0; candid < m_keypoints_undist[view].size(); candid++)
		{
			for (int jid = 0; jid < m_topo.joint_num; jid++)
			{
				if (m_keypoints_undist[view][candid][jid](2) > m_topo.kpt_conf_thresh[jid])
				{
					m_keypoints_pool[view][jid].push_back(m_keypoints_undist[view][candid][jid]);
				}
			}
		}
	}

	// nms 
	for (int view = 0; view < m_camNum; view++)
	{
		for (int jid = 0; jid < m_topo.joint_num; jid++)
		{
			nmsKeypointCands( m_keypoints_pool[view][jid]);
		}
	}
}


void FrameSolver::reAssociateKeypoints()
{
	m_keypoints_associated.resize(4); 
	m_skelVis.resize(4);
	renderInteractDepth();
	for (int i = 0; i < 4; i++)
	{
		m_keypoints_associated[i].resize(m_camNum); 
		m_skelVis[i].resize(m_camNum); 
		mp_bodysolverdevice[i]->computeAllSkelVisibility(); 
		m_skelVis[i] = mp_bodysolverdevice[i]->m_skel_vis;
		for (int camid = 0; camid < m_camNum; camid++)
		{
			m_keypoints_associated[i][camid].resize(m_topo.joint_num, Eigen::Vector3f::Zero()); 
		}
	}

	//std::cout << "CHECK VISI: " << std::endl;
	//std::cout << m_skelVis[1][3][2] << std::endl; 
	//std::cout << m_skelVis[2][8][6] << std::endl; 
	//std::cout << m_skelVis[3][8][6] << std::endl; 

	if (m_skels3d.size() < 1) m_skels3d.resize(4); 

	for (int i = 0; i < 4; i++)
	{
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host(); 
	}
	reproject_skels(); 
	

	for (int camid = 0; camid < m_camNum; camid++)
	{
		// associate for each camera 
		for (int i = 0; i < m_topo.joint_num; i++)
		{
			std::vector<int> id_table; 
			for (int pid = 0; pid < 4; pid++)
			{
				if(m_skelVis[pid][camid][i] > 0)
					id_table.push_back(pid); 
			}
			int M = id_table.size(); // candidate number for associate  
			int N = m_keypoints_pool[camid][i].size(); 
			Eigen::MatrixXf sim(M, N); 
			for (int rowid = 0; rowid < M; rowid++)
			{
				for (int colid = 0; colid < N; colid++)
				{
					sim(rowid, colid) = (m_keypoints_pool[camid][i][colid].segment<2>(0)
						- m_projs[camid][id_table[rowid]][i].segment<2>(0)).norm();
					if (sim(rowid, colid) > 200) sim(rowid, colid) = 200;
				}
			}
			std::vector<int> assign = solveHungarian(sim);
			for (int rowid = 0; rowid < M; rowid++)
			{
				int pid = id_table[rowid];
				int colid = assign[rowid];
				if (colid < 0) continue; 
				if (sim(rowid, colid) >= 200) continue; 
				m_keypoints_associated[pid][camid][i] = m_keypoints_pool[camid][i][colid];
				m_keypoints_pool[camid][i][colid].setZero();
			}
		}
	}

	// reassoc swap 
	std::vector<std::vector<int> > joint_levels = {
		{9, 10, 15, 16}, // bottom level: foot 
	{7,8,13,14}, // middle level: elbow
	{5,6,11,12} // top level: shoulder
	};
	for (int k = 0; k < joint_levels.size(); k++)
	{
		std::vector<int> ids_to_swap = joint_levels[k];

		for (int camid = 0; camid < m_camNum; camid++)
		{
			std::vector<Eigen::Vector3f> remain_pool;
			for (int i = 0; i < ids_to_swap.size(); i++)
			{
				int id = ids_to_swap[i];
				for (int candid = 0; candid < m_keypoints_pool[camid][id].size(); candid++)
				{
					if (m_keypoints_pool[camid][id][candid](2) == 0)continue;
					remain_pool.push_back(m_keypoints_pool[camid][id][candid]);
				}
			}
			std::vector<int> pig_id_table;
			std::vector<int> joint_id_table;
			std::vector<Eigen::Vector3f> projPool;
			for (int pid = 0; pid < 4; pid++)
			{
				for (int i = 0; i < ids_to_swap.size(); i++)
				{
					int id = ids_to_swap[i];
					if (m_keypoints_associated[pid][camid][id](2) == 0
						&& m_skelVis[pid][camid][id] > 0)
					{
						pig_id_table.push_back(pid);
						joint_id_table.push_back(id);
						projPool.push_back(m_projs[camid][pid][id]);
					}
				}
			}
			int M = remain_pool.size();
			int N = projPool.size();
			Eigen::MatrixXf sim = Eigen::MatrixXf::Zero(M, N);
			for (int i = 0; i < M; i++)
			{
				for (int j = 0; j < N; j++)
				{
					float dist = (remain_pool[i].segment<2>(0) - projPool[j].segment<2>(0)).norm();
					sim(i, j) = dist > 200 ? 200 : dist;
				}
			}
			std::vector<int> match = solveHungarian(sim);
			for (int i = 0; i < match.size(); i++)
			{
				if (match[i] < 0) continue;
				if (sim(i, match[i]) >= 200) continue;
				int j = match[i];
				m_keypoints_associated[pig_id_table[j]][camid][joint_id_table[j]]
					= remain_pool[i];
			}
		}
	}
}



void FrameSolver::solve_parametric_model_optimonly()
{
	m_skels3d.resize(4);
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]);
		mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist;
		mp_bodysolverdevice[i]->globalAlign();
		setConstDataToSolver(i);

		std::vector<ROIdescripter> rois;
		getROI(rois, i);
		mp_bodysolverdevice[i]->setROIs(rois);
		mp_bodysolverdevice[i]->m_w_anchor_term = 0.001;
	}

	if (m_solve_sil_iters > 0)
	{
		for (int i = 0; i < 4; i++)
			mp_bodysolverdevice[i]->m_isReAssoc = false; 
		optimizeSilWithAnchor(m_solve_sil_iters);
		//reAssocProcessStep1(); 
		reAssocWithoutTracked();
		optimizeSilWithAnchor(m_solve_sil_iters);
	}

	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->postProcessing();
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host();
	}

	// postprocess
	m_last_matched = m_matched;
}

cv::Mat FrameSolver::visualizeReassociation()
{
	std::vector<cv::Mat> reassoc; 
	cloneImgs(m_imgsUndist, reassoc);

	for (int id = 0; id < 4; id++)
	{
		for (int i= 0; i < m_matched[id].view_ids.size(); i++)
		{
			Eigen::Vector3i color;
			color(0) = m_CM[id](2);
			color(1) = m_CM[id](1);
			color(2) = m_CM[id](0);
			int camid = m_matched[id].view_ids[i];
			
			my_draw_box(reassoc[camid], m_matched[id].dets[i].box, color);

			if (m_matched[id].dets[i].mask.size() > 0)
				my_draw_mask(reassoc[camid], m_matched[id].dets[i].mask, color, 0.5);
		}
		for (int camid = 0; camid < m_camNum; camid++)
		{
			drawSkelMonoColor(reassoc[camid], m_keypoints_associated[id][camid], id, m_topo);
		}
	}

	cv::Mat packed;
	packImgBlock(reassoc, packed);
	return packed; 
}

cv::Mat FrameSolver::visualizeVisibility()
{
	std::vector<cv::Mat> imgdata;
	cloneImgs(m_imgsUndist, imgdata);
	reproject_skels();

	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int id = 0; id < m_projs[camid].size(); id++)
		{
			std::vector<Eigen::Vector3f> joints = m_projs[camid][id];
			for (int i = 0; i < m_topo.joint_num; i++)
			{
				if (m_skelVis[id][camid][i] == 0) m_projs[camid][id][i](2) = 0; 
			}
			drawSkelMonoColor(imgdata[camid], m_projs[camid][id], id, m_topo);
		}
	}

	cv::Mat packed;
	packImgBlock(imgdata, packed);
	return packed;
}

cv::Mat FrameSolver::visualizeSwap()
{
	std::vector<cv::Mat> swap_list(4); 
	for (int i = 0; i < 4; i++)
	{
		swap_list[i] = mp_bodysolverdevice[i]->debug_vis_reassoc_swap(); 
	}
	cv::Mat output; 
	packImgBlock(swap_list, output); 
	return output; 
}

void FrameSolver::reAssocProcessStep1()
{
	splitDetKeypoints();
	reAssociateKeypoints();
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->m_isReAssoc = true;
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
}

cv::Mat FrameSolver::visualizeRawAssoc()
{
	std::vector<cv::Mat> imglist(4); 
	for (int i = 0; i < 4; i++)
	{
		imglist[i] = mp_bodysolverdevice[i]->debug_source_visualize(); 
	}
	cv::Mat output;
	packImgBlock(imglist, output); 
	return output; 
}


// 2020/11/07
// reassoc without tracked joints 

void FrameSolver::determineTracked()
{
	m_detTracked.resize(m_camNum);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		m_detTracked[camid].resize(m_detUndist[camid].size());
		for (int k = 0; k < m_detTracked[camid].size(); k++)
		{
			m_detTracked[camid][k].resize(m_topo.joint_num, -1);
		}
	}

	m_modelTracked.resize(4);
	for (int pid = 0; pid < 4; pid++)
	{
		m_modelTracked[pid].resize(m_camNum);
		for (int camid = 0; camid < m_modelTracked[pid].size(); camid++)
		{
			m_modelTracked[pid][camid].resize(m_topo.joint_num, -1);
		}
	}

	if (m_skels3d.size() < 1) m_skels3d.resize(4);

	for (int i = 0; i < 4; i++)
	{
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host();
	}
	reproject_skels();

	for (int pid = 0; pid < 4; pid++)
	{
		for (int view = 0; view < m_matched[pid].view_ids.size(); view++)
		{
			int camid = m_matched[pid].view_ids[view];
			int candid = m_matched[pid].candids[view];
			for (int i = 0; i < m_topo.joint_num; i++)
			{
				Eigen::Vector3f pointDetect = m_detUndist[camid][candid].keypoints[i];
				Eigen::Vector3f pointProj = m_projs[camid][pid][i];
				if (pointProj.norm() == 0)
				{
					m_detTracked[camid][candid][i] = -1;
					m_modelTracked[pid][camid][i] = -1;
					continue;
				}
				if (pointDetect(2) < m_topo.kpt_conf_thresh[i])
				{
					m_detTracked[camid][candid][i] = -1;
					m_modelTracked[pid][camid][i] = -1;
					continue;
				}
				if ((pointDetect.segment<2>(0) - pointProj.segment<2>(0)).norm() < 30)
				{
					m_detTracked[camid][candid][i] = pid;
					m_modelTracked[pid][camid][i] = candid;
				}
			}
		}
	}
}

cv::Mat FrameSolver::debug_visDetTracked()
{
	std::vector<cv::Mat> track_list;
	cloneImgs(m_imgsUndist, track_list);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for(int pid = 0; pid < 4; pid++)
		{
			std::vector<Eigen::Vector3f> keypoints(m_topo.joint_num, Eigen::Vector3f::Zero());
			int candid_gt = m_clusters[pid][camid];
			for (int i = 0; i < m_topo.joint_num; i++)
			{
				if (m_modelTracked[pid][camid][i] >= 0)
				{
					int candid = m_modelTracked[pid][camid][i];
					if (candid != candid_gt)
					{
						std::cout << "pig " << pid << ", cam " << camid << "  jid: " << i << std::endl;
					}
					keypoints[i] = m_detUndist[camid][candid].keypoints[i];
				}
			}
			drawSkelMonoColor(track_list[camid], keypoints, pid, m_topo);
		}
	}
	cv::Mat output;
	packImgBlock(track_list, output);
	return output;
}


// 20201103: split all detected keypoints without fine track
void FrameSolver::nms2(std::vector<Eigen::Vector3f>& pool, int jointid, const std::vector<std::vector<Eigen::Vector3f> >& ref)
{
	std::vector<std::vector<int> > joint_levels = {
		{9, 10, 15, 16}, // bottom level: foot 
	{7,8,13,14}, // middle level: elbow
	{5,6,11,12} // top level: shoulder
	};
	std::vector<Eigen::Vector3f> pool_bk = pool;
	std::vector<bool> repeat(pool.size(), false);

	if (jointid < 5 || jointid > 16)
	{
		for (int i = 0; i < pool.size(); i++)
		{
			for (int j = 0; j < ref[jointid].size(); j++)
			{
				if ((pool[i].segment<2>(0) - ref[jointid][j].segment<2>(0)).norm() < 10)
					repeat[i] = true;
				if (repeat[i]) break;
			}
		}
	}
	else {

		int level = -1;
		for (int i = 0; i < 3; i++)
		{
			if (in_list(jointid, joint_levels[i]))level = i;
		}
		for (int i = 0; i < pool.size(); i++)
		{
			for (int k = 0; k < joint_levels[level].size(); k++)
			{
				if (repeat[i]) break;
				int jid = joint_levels[level][k];
				for (int j = 0; j < ref[jid].size(); j++)
				{
					if ((pool[i].segment<2>(0) - ref[jid][j].segment<2>(0)).norm() < 10)
					{
						repeat[i] = true;
					}
					if (repeat[i])break;
				}
			}
		}
	}

	pool.clear();
	for (int i = 0; i < pool_bk.size(); i++)
	{
		if (!repeat[i])pool.push_back(pool_bk[i]);
	}
}

void FrameSolver::splitDetKeypointsWithoutTracked()
{
	std::vector<std::vector<std::vector<Eigen::Vector3f> > >  keypoints_trackedPool;
	m_keypoints_pool.resize(m_camNum);
	keypoints_trackedPool.resize(m_camNum);
	for (int view = 0; view < m_camNum; view++)
	{
		m_keypoints_pool[view].resize(m_topo.joint_num);
		keypoints_trackedPool[view].resize(m_topo.joint_num);
		for (int candid = 0; candid < m_detUndist[view].size(); candid++)
		{
			for (int jid = 0; jid < m_topo.joint_num; jid++)
			{
				if (m_detUndist[view][candid].keypoints[jid](2) < m_topo.kpt_conf_thresh[jid])continue;

				if (m_detTracked[view][candid][jid] >= 0) {
					keypoints_trackedPool[view][jid].push_back(m_detUndist[view][candid].keypoints[jid]);
					continue; // only consider untracked points
				}
				else
				{
					m_keypoints_pool[view][jid].push_back(m_detUndist[view][candid].keypoints[jid]);
				}
			}
		}
	}

	// nms : nms for same type, just merge 
	for (int view = 0; view < m_camNum; view++)
	{
		for (int jid = 0; jid < m_topo.joint_num; jid++)
		{
			nmsKeypointCands(m_keypoints_pool[view][jid]);
		}
	}

	// nms: nms for same level. could not simply merge

	for (int view = 0; view < m_camNum; view++)
	{
		for (int jid = 0; jid < m_topo.joint_num; jid++)
		{
			nms2(m_keypoints_pool[view][jid], jid, keypoints_trackedPool[view]);
		}
	}
}


void FrameSolver::reAssocKeypointsWithoutTracked()
{
	m_keypoints_associated.resize(4);
	m_skelVis.resize(4);
	renderInteractDepth();
	for (int i = 0; i < 4; i++)
	{
		m_keypoints_associated[i].resize(m_camNum);
		m_skelVis[i].resize(m_camNum);
		mp_bodysolverdevice[i]->computeAllSkelVisibility();
		m_skelVis[i] = mp_bodysolverdevice[i]->m_skel_vis;
		for (int camid = 0; camid < m_camNum; camid++)
		{
			m_keypoints_associated[i][camid].resize(m_topo.joint_num, Eigen::Vector3f::Zero());
			for (int k = 0; k < m_topo.joint_num; k++)
			{
				if (m_modelTracked[i][camid][k] >= 0)
				{
					int candid = m_modelTracked[i][camid][k];
					m_keypoints_associated[i][camid][k] = m_detUndist[camid][candid].keypoints[k];
				}
			}
		}
	}

#ifdef VIS_ASSOC_STEP
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
	cv::Mat output = visualizeSwap();
	cv::imwrite("H:/pig_results_anchor/before_swap/step1.png", output); 
#endif 
	for (int camid = 0; camid < m_camNum; camid++)
	{
		// associate for each camera 
		for (int i = 0; i < m_topo.joint_num; i++)
		{
			std::vector<int> id_table;
			for (int pid = 0; pid < 4; pid++)
			{
				if (m_keypoints_associated[pid][camid][i](2) > 0) continue; 
				if (m_skelVis[pid][camid][i] > 0)
					id_table.push_back(pid);
			}
			int M = id_table.size(); // candidate number for associate  
			int N = m_keypoints_pool[camid][i].size();
			Eigen::MatrixXf sim(M, N);
			for (int rowid = 0; rowid < M; rowid++)
			{
				for (int colid = 0; colid < N; colid++)
				{
					sim(rowid, colid) = (m_keypoints_pool[camid][i][colid].segment<2>(0)
						- m_projs[camid][id_table[rowid]][i].segment<2>(0)).norm();
					if (sim(rowid, colid) > 200) sim(rowid, colid) = 200;
				}
			}
			std::vector<int> assign = solveHungarian(sim);
			for (int rowid = 0; rowid < M; rowid++)
			{
				int pid = id_table[rowid];
				int colid = assign[rowid];
				if (colid < 0) continue;
				if (sim(rowid, colid) >= 200) continue;
				m_keypoints_associated[pid][camid][i] = m_keypoints_pool[camid][i][colid];
				m_keypoints_pool[camid][i][colid].setZero();
			}
		}
	}

#ifdef  VIS_ASSOC_STEP
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
	output = visualizeSwap();
	cv::imwrite("H:/pig_results_anchor/before_swap/step2.png", output);
#endif 
	// reassoc swap 
	std::vector<std::vector<int> > joint_levels = {
		{9, 10, 15, 16}, // bottom level: foot 
	{7,8,13,14}, // middle level: elbow
	{5,6,11,12} // top level: shoulder
	};
	for (int k = 0; k < joint_levels.size(); k++)
	{
		std::vector<int> ids_to_swap = joint_levels[k];

		for (int camid = 0; camid < m_camNum; camid++)
		{
			std::vector<Eigen::Vector3f> remain_pool;
			for (int i = 0; i < ids_to_swap.size(); i++)
			{
				int id = ids_to_swap[i];
				for (int candid = 0; candid < m_keypoints_pool[camid][id].size(); candid++)
				{
					if (m_keypoints_pool[camid][id][candid](2) == 0)continue;
					remain_pool.push_back(m_keypoints_pool[camid][id][candid]);
				}
			}
			std::vector<int> pig_id_table;
			std::vector<int> joint_id_table;
			std::vector<Eigen::Vector3f> projPool;
			for (int pid = 0; pid < 4; pid++)
			{
				for (int i = 0; i < ids_to_swap.size(); i++)
				{
					int id = ids_to_swap[i];
					if (m_keypoints_associated[pid][camid][id](2) == 0
						&& m_skelVis[pid][camid][id] > 0)
					{
						pig_id_table.push_back(pid);
						joint_id_table.push_back(id);
						projPool.push_back(m_projs[camid][pid][id]);
					}
				}
			}
			int M = remain_pool.size();
			int N = projPool.size();
			Eigen::MatrixXf sim = Eigen::MatrixXf::Zero(M, N);
			for (int i = 0; i < M; i++)
			{
				for (int j = 0; j < N; j++)
				{
					float dist = (remain_pool[i].segment<2>(0) - projPool[j].segment<2>(0)).norm();
					sim(i, j) = dist > 200 ? 200 : dist;
				}
			}
			std::vector<int> match = solveHungarian(sim);
			for (int i = 0; i < match.size(); i++)
			{
				if (match[i] < 0) continue;
				if (sim(i, match[i]) >= 200) continue;
				int j = match[i];
				m_keypoints_associated[pig_id_table[j]][camid][joint_id_table[j]]
					= remain_pool[i];
			}
		}
	}

#ifdef VIS_ASSOC_STEP
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
	
	output = visualizeSwap();
	cv::imwrite("H:/pig_results_anchor/before_swap/step3.png", output);
#endif 
}

void FrameSolver::reAssocWithoutTracked()
{
	determineTracked(); 
	splitDetKeypointsWithoutTracked();
	reAssocKeypointsWithoutTracked();
	
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->m_isReAssoc = true;
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
}