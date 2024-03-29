#include <filesystem>
#include <json/writer.h> 

#include "../utils/timer_util.h" 

#include "framesolver.h"
#include "matching.h"
#include "tracking.h"

//#define VIS_ASSOC_STEP 
//#define DEBUG_TRACK
//#define DEBUG_SIL

FrameSolver::FrameSolver()
{
	m_epi_thres = -1;
	m_epi_type = "p2l";

}

void FrameSolver::configByJson(std::string jsonfile)
{
	m_input_configfile = jsonfile; 

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
	m_keypointsDir = m_sequence + "/" + root["keypointsdir"].asString() + "/";
	m_imgDir = m_sequence + "/" + root["imgdir"].asString() + "/";
	m_boxDir = m_sequence + "/" + root["boxdir"].asString() + "/";
	m_maskDir = m_sequence + "/" + root["maskdir"].asString() + "/";
	m_camDir = root["camfolder"].asString();
	m_imgExtension = root["imgExtension"].asString();
	m_pignum = root["pignum"].asInt();

	m_is_read_image = root["is_read_image"].asBool();
	m_videotype = root["videotype"].asInt();
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
	m_anchor_folder = root["anchor_folder"].asString(); 
	m_annotation_folder = root["annotation_folder"].asString(); 
	m_use_reassoc = root["use_reassoc"].asBool(); 
	m_solve_sil_iters_2nd_phase = root["solve_sil_iters_2nd_phase"].asInt(); 
	m_initialization_iters = root["initialization_iters"].asInt(); 
	m_terminal_thresh = root["terminal_thresh"].asFloat(); 
	m_result_folder = root["result_folder"].asString(); 
	m_use_given_scale = root["use_given_scale"].asBool(); 
	m_use_init_cluster = root["use_init_cluster"].asBool(); 
	m_scenedata_path = root["scenedata"].asString(); 
	m_background_folder = root["background_folder"].asString(); 
	m_try_load_anno = root["try_load_anno"].asBool(); 
	m_use_triangulation_only = root["use_triangulation_only"].asBool();
	m_use_init_pose = root["use_init_pose"].asBool(); 
	m_use_init_anchor = root["use_init_anchor"].asBool();
	m_pig_names.resize(m_pignum); 
	m_restart_threshold = root["restart_thresold"].asInt(); 
	m_tracking_distance = root["tracking_distance"].asFloat(); 
	m_intrinsic_type = root["intrinsic_type"].asInt(); 
	m_use_per_frame_anchor = root["use_per_frame_anchor"].asBool(); 
	m_render_resize_ratio = root["render_resize_ratio"].asFloat(); 
	m_params.loadParams(root["optim_params"]); 
	

	for (int i = 0; i < m_pignum; i++)
	{
		m_pig_names[i] = root["pig_names"][i].asInt(); 
	}
	std::vector<int> camids;
	for (auto const &c : root["camids"])
	{
		int id = c.asInt();
		camids.push_back(id);
	}
	m_camids = camids;
	m_camNum = m_camids.size();

	m_render_views.clear(); 
	for (auto const & c : root["render_views"])
	{
		int id = c.asInt();
		m_render_views.push_back(id);
	}

	m_given_scales.resize(m_pignum);
	for (int i = 0; i < m_pignum; i++)
	{
		m_given_scales[i] = root["scales"][i].asFloat();
	}

	instream.close();
	readCameras(); 

	if (m_videotype==1)
	{
		m_hourid = root["hourid"].asInt();
		m_caps.clear();
		m_caps.resize(m_camNum);
		for (int camid = 0; camid < m_camNum; camid++)
		{
			std::stringstream name;
			name << m_sequence << "videos/cam" << m_camids[camid] << "/hour" <<
				std::setw(6) << std::setfill('0') << m_hourid << ".mp4";
			m_caps[camid] = cv::VideoCapture(name.str());

			if (!m_caps[camid].isOpened())
			{
				std::cout << "cannot open video " << name.str() << std::endl;
				system("pause");
				exit(-1);
			}
		}
	}
	else if (m_videotype == 2)
	{
		m_caps.clear();
		m_caps.resize(m_camNum);
		for (int camid = 0; camid < m_camNum; camid++)
		{
			std::stringstream name;
			name << m_sequence << "videos/" << m_camids[camid] << ".mp4";
			m_caps[camid] = cv::VideoCapture(name.str());
			if (!m_caps[camid].isOpened())
			{
				std::cout << "cannot open video " << name.str() << std::endl;
				system("pause");
				exit(-1);
			}
		}
	}

	m_video_frameid = 0;

	mp_sceneData = std::make_shared<SceneData>(
		m_camDir, m_background_folder, m_scenedata_path, m_camids
		);

	d_interDepth.resize(m_camNum); 
	int H = WINDOW_HEIGHT;
	int W = WINDOW_WIDTH;
	for (int i = 0; i < m_camNum; i++)
	{
		cudaMalloc((void**)&d_interDepth[i], H * W * sizeof(float));
	}

	initRectifyMap(); 
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
	m_projs.resize(m_camNum);
	for (int c = 0; c < m_camNum; c++) m_projs[c].resize(m_pignum);

	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int id = 0; id < m_pignum; id++)
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

	for (int id = 0; id < m_pignum; id++)
	{
		if (vid >= 0 && id != vid)continue;
		int colorid = m_pig_names[id];
		for (int i = 0; i < m_matched[id].view_ids.size(); i++)
		{
			if (viewid >= 0 && m_matched[id].view_ids[i] != viewid) continue;
			Eigen::Vector3i color;
			color(0) = m_CM[colorid](2);
			color(1) = m_CM[colorid](1);
			color(2) = m_CM[colorid](0);
			int camid = m_matched[id].view_ids[i];
			//int candid = m_matched[id].cand_ids[i];
			//if(candid < 0) continue; 
			if (m_matched[id].dets[i].keypoints.size() > 0)
				drawSkelMonoColor(m_imgsDetect[camid], m_matched[id].dets[i].keypoints, color, m_topo);
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
				drawSkelMonoColor(m_imgsDetect[camid], m_unmatched[camid][i].keypoints, color, m_topo);
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

cv::Mat FrameSolver::visualizeProj(int viewid, int pid, bool withimg)
{
	std::vector<cv::Mat> imgdata;
	if(withimg)
		cloneImgs(m_imgsUndist, imgdata);
	else
	{
		imgdata.resize(m_camNum); 
		for (int i = 0; i < m_camNum; i++)
		{
			cv::Mat rawimg;
			rawimg.create(cv::Size(1920, 1080), CV_8UC3);
			rawimg.setTo(cv::Scalar(255, 255, 255));
			imgdata[i] = rawimg; 
		}
	}
	reproject_skels();

	for (int camid = 0; camid < m_camNum; camid++)
	{
		if (viewid >= 0 && viewid != camid) continue; 
		for (int id = 0; id < m_projs[camid].size(); id++)
		{
			if (pid >= 0 && id != pid) continue;
			/*if (pid == 0 || pid == 2) continue;*/
			Eigen::Vector3i color; 
			int colorid = m_pig_names[id];
			color(0) = m_CM[colorid](2);
			color(1) = m_CM[colorid](1);
			color(2) = m_CM[colorid](0);
			drawSkelProj(imgdata[camid], m_projs[camid][id], color, m_topo);
		}
	}

	if (viewid >= 0)
		return imgdata[viewid];

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
	vector<vector<int>> recluster(m_pignum);
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

void FrameSolver::save_parametric_data()
{
	for (int i = 0; i < m_pignum; i++)
	{
		std::string savefolder = m_result_folder + "/state";
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
	//std::vector<int> mpp = { 0,1,3,2 };
	for (int i = 0; i < m_pignum; i++)
	{
		if (m_matched.size() > 0)
		{
			mp_bodysolverdevice[i]->setSource(m_matched[i]);
			mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist;
		}
		std::string savefolder =
			m_result_folder + "/state";
			//"E:/results/paper_teaser/0704_eval2-5views/state";
		if (is_smth) savefolder = savefolder + "_smth";
		std::stringstream ss;
		ss << savefolder << "/pig_" << i << "_frame_" <<
			std::setw(6) << std::setfill('0') << m_frameid
			<< ".txt";
		mp_bodysolverdevice[i]->readState(ss.str());
		mp_bodysolverdevice[i]->UpdateVertices();
		mp_bodysolverdevice[i]->m_isUpdated = true; 
		if(m_matched.size() > 0) 
			mp_bodysolverdevice[i]->postProcessing();
	}
#ifdef USE_SIFT
	readSIFTandTrack(); 
#endif 
}

bool FrameSolver::try_load_anno()
{
	bool loaded = false;
	init_parametric_solver(); 
	for (int i = 0; i < m_pignum; i++)
	{
		if (m_matched.size() > 0)
		{
			mp_bodysolverdevice[i]->setSource(m_matched[i]);
			mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist;
		}
		std::string savefolder = m_annotation_folder;
		std::stringstream ss;
		ss << savefolder << "/pig_" << i << "_frame_" <<
			std::setw(6) << std::setfill('0') << m_frameid
			<< ".txt";
		if (std::filesystem::exists(ss.str()))
		{
			mp_bodysolverdevice[i]->readState(ss.str()); 
			mp_bodysolverdevice[i]->UpdateVertices(); 
			mp_bodysolverdevice[i]->m_isUpdated = true; 
			loaded = true; 
			std::cout << "load " << ss.str() << std::endl; 
		}
	}
	return loaded; 
}

void FrameSolver::matching_by_tracking()
{
	m_skels3d_last = m_skels3d;

	// get m_clusters
	EpipolarMatching matcher;
	matcher.set_pignum(m_pignum);
	matcher.set_cams(m_camsUndist);
	matcher.set_dets(m_detUndist);
	matcher.set_epi_thres(m_epi_thres);
	matcher.set_epi_type(m_epi_type);
	matcher.set_topo(m_topo);
	if (m_frameid == m_startid || m_match_alg == "match")
	{
		matcher.match();
		matcher.truncate(m_pignum);
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

void FrameSolver::restart()
{
	// step1: determine which pig to restart
	vector<vector<Eigen::Vector3f> > skels3d_last; 
	vector<int> restart_id; 
	for (int i = 0; i < m_pignum; i++)
	{
		std::cout << "pig " << i << "  trackconf: " << mp_bodysolverdevice[i]->m_trackConf << std::endl; 

		if (mp_bodysolverdevice[i]->m_trackConf < m_restart_threshold)
		{
			restart_id.push_back(i); 
			skels3d_last.push_back(m_skels3d[i]);
		}
	}
	if (restart_id.size() < 1) return;

	
	// get m_clusters
	EpipolarMatching matcher;
	matcher.set_pignum(restart_id.size());
	matcher.set_cams(m_camsUndist);
	matcher.set_dets(m_unmatched);
	matcher.set_epi_thres(m_epi_thres);
	matcher.set_epi_type(m_epi_type);
	matcher.set_topo(m_topo);
	
	matcher.match();
	matcher.truncate(restart_id.size());

	vector<vector<int> > clusters; 
	vector<vector<Eigen::Vector3f> > skels3d; 
	matcher.get_clusters(clusters);
	matcher.get_skels3d(skels3d);

	if (clusters.size() == 0) return;

	// post processing to get matched data
	// this match data is unordered
	vector<vector<bool> > be_matched;
	be_matched.resize(m_camNum);
	auto unmatched_previous = m_unmatched; 
	for (int camid = 0; camid < m_camNum; camid++)
	{
		be_matched[camid].resize(unmatched_previous[camid].size(), false);
	}
	vector<MatchedInstance> currentMatched; 
	currentMatched.resize(clusters.size()); 
	for (int i = 0; i < clusters.size(); i++)
	{
		currentMatched[i].view_ids.clear();
		currentMatched[i].dets.clear();
		for (int camid = 0; camid < m_camNum; camid++)
		{
			int candid = clusters[i][camid];
			if (candid < 0) continue;
			be_matched[camid][candid] = true;
			currentMatched[i].view_ids.push_back(camid);
			currentMatched[i].dets.push_back(unmatched_previous[camid][candid]);
			currentMatched[i].candids.push_back(candid);
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
				m_unmatched[camid].push_back(unmatched_previous[camid][candid]);
			}
		}
	}

	// track 
	NaiveTracker m_tracker;
	m_tracker.set_skels_last(skels3d_last);
	m_tracker.m_cameras = m_camsUndist;
	m_tracker.m_topo = m_topo;
	m_tracker.track(currentMatched);

	vector<int> map = m_tracker.get_map();
	vector<MatchedInstance> rematch;
	vector<vector<int>> recluster(clusters.size());
	rematch.resize(currentMatched.size());
	for (int i = 0; i < map.size(); i++)
	{
		int id = map[i];
		if (id > -1)
		{
			rematch[i] = currentMatched[id];
			recluster[i] = clusters[id];
		}
	}

	// add restarted results to global tracking results 
	for (int i = 0; i < restart_id.size(); i++)
	{
		int id = restart_id[i];
		m_matched[id] = rematch[i];
		m_clusters[id] = recluster[i];
	}
}

void FrameSolver::updateTrackConf()
{
	for (int i = 0; i < m_pignum; i++)
	{
		if (m_matched[i].dets.size() < 1)
		{
			mp_bodysolverdevice[i]->m_trackConf -= 1;
		}
		else {
			if (mp_bodysolverdevice[i]->m_trackConf < m_restart_threshold)
			{
				if (m_matched[i].dets.size() >= 2)
				{
					mp_bodysolverdevice[i]->optimizePose();
					mp_bodysolverdevice[i]->m_trackConf = 0;
				}			
				else
				{
					mp_bodysolverdevice[i]->m_trackConf -= 1;
					/*mp_bodysolverdevice[i]->m_trackConf += 1; 
					if (mp_bodysolverdevice[i]->m_trackConf > 0)
						mp_bodysolverdevice[i]->m_trackConf = 0; */
				}
			}

		}
	}
}

void FrameSolver::save_clusters()
{
	std::stringstream ss;
	ss << m_result_folder << "/clusters/" << std::setw(6) << std::setfill('0') << m_frameid << ".txt";
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
	ss << m_result_folder << "/clusters/" << std::setw(6) << std::setfill('0') << m_frameid << ".txt";
	std::ifstream stream(ss.str());
	if (!stream.is_open())
	{
		std::cout << "cluster loading stream not open. " << std::endl;
		return;
	}
	m_clusters.resize(m_pignum);
	for (int i = 0; i < m_pignum; i++)
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
void FrameSolver::drawMaskMatched()
{
	m_masksMatched.resize(m_camNum);
	for (int i = 0; i < m_camNum; i++)
	{
		m_masksMatched[i].create(cv::Size(m_imw, m_imh), CV_8UC1);
		m_masksMatched[i].setTo(0); 
	} 
	for (int pid = 0; pid < m_pignum; pid++)
	{
		for (int k = 0; k < m_matched[pid].view_ids.size(); k++)
		{
			cv::Mat temp(cv::Size(m_imw, m_imh), CV_8UC1);

			int viewid = m_matched[pid].view_ids[k];
			my_draw_mask_gray(temp, m_matched[pid].dets[k].mask, 1 << pid);
			m_masksMatched[viewid] = temp + m_masksMatched[viewid];
		}


	}

	//for (int i = 0; i < m_camNum; i++)
	//{
	//	std::stringstream ss;
	//	ss << "G:/pig_results/maskmatched" << i << ".png";
	//	cv::imwrite(ss.str(), m_masksMatched[i] * 63);
	//}
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

		rois[view].mask = m_masksMatched[camid];
		rois[view].box = m_matched[id].dets[view].box;
		rois[view].undist_mask = mp_sceneData->m_undist_mask; // valid area for image distortion 
		rois[view].scene_mask = mp_sceneData->m_scene_masks[camid];
		rois[view].pid = id;
		rois[view].idcode = 1 << id;
		rois[view].valid = rois[view].keypointsMaskOverlay();
		computeGradient(rois[view].chamfer, rois[view].gradx, rois[view].grady);
	}
}

void FrameSolver::setConstDataToSolver()
{
	drawMaskMatched();
	for (int id = 0; id < m_pignum; id++)
	{

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
		mp_bodysolverdevice[id]->m_det_masks = m_masksMatched;

		// mask necessary for measure anchor point
		std::vector<ROIdescripter> rois;
		getROI(rois, id);
		mp_bodysolverdevice[id]->setROIs(rois);
	}
}


void FrameSolver::pureTracking()
{
	m_skels3d_last = m_skels3d;
	std::vector<std::vector<std::vector<Eigen::Vector3f> > > skels2d;
	skels2d.resize(m_pignum);
	for (int i = 0; i < m_pignum; i++)
	{
		skels2d[i] = mp_bodysolverdevice[i]->getSkelsProj();
	}
	m_clusters.clear();
	m_clusters.resize(m_pignum);
	for (int pid = 0; pid < m_pignum; pid++)m_clusters[pid].resize(m_camNum, -1);

	float threshold = m_tracking_distance; 
	//renderInteractDepth(true); 
	for (int camid = 0; camid < m_camNum; camid++)
	{
		Eigen::MatrixXf sim;
		int boxnum = m_detUndist[camid].size();
		sim.resize(m_pignum, boxnum);
		Eigen::MatrixXf sim2; 
		sim2.resize(m_pignum, boxnum); 
		Eigen::MatrixXf valids; 
		valids.resize(m_pignum, boxnum); 
		for (int i = 0; i < m_pignum; i++)
		{
			for (int j = 0; j < boxnum; j++)
			{
				float valid;
				float valid2 = -1; 
				float dist = distSkel2DTo2D(skels2d[i][camid],
					m_detUndist[camid][j].keypoints,
					m_topo, valid);

				int viewid = find_in_list(camid, m_last_matched[i].view_ids);
				float dist2;
				if (viewid < 0) dist2 = dist; 
				else
				{
					int candid = m_last_matched[i].candids[viewid];
					
					dist2 = distSkel2DTo2D(m_last_matched[i].dets[viewid].keypoints,
						m_detUndist[camid][j].keypoints,
						m_topo, valid2); // calc last 2D detection to current detection
				}
				//sim(i, j) = dist + dist2;
				sim(i, j) = dist; 
				sim2(i, j) = dist2; 
				if (valid2 < 0) valid2 = valid;
				valids(i, j) = valid + valid2;
				if (sim(i, j) < 30) sim(i, j) /= valids(i, j); 

				if (sim(i, j) > threshold) sim(i, j) = threshold;
			}
		}

#ifdef DEBUG_TRACK
		std::cout << "sim of view: " << camid << std::endl << sim << std::endl;
		std::cout << "valid of view: " << std::endl << valids<< std::endl; 
		std::cout << "dist2 : " << std::endl << sim2 << std::endl; 
#endif 
		std::vector<int> mm = solveHungarian(sim);

#ifdef DEBUG_TRACK
		for (int i = 0; i < mm.size(); i++)
			std::cout << mm[i] << "  ";
		std::cout << std::endl; 
#endif 
		for (int i = 0; i < m_pignum; i++)
		{
			if (mm[i] >= 0)
			{
				int candid = mm[i];
				if (sim(i, candid) >= threshold)continue;
				else m_clusters[i][camid] = candid;
				// 20210418: adapt to 1003 data. 
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
}

void FrameSolver::init_parametric_solver()
{
	if (mp_bodysolverdevice.empty()) mp_bodysolverdevice.resize(m_pignum);
	for (int i = 0; i < m_pignum; i++)
	{
		if (mp_bodysolverdevice[i] == nullptr)
		{
			std::cout << m_pigConfig << std::endl;
			mp_bodysolverdevice[i] = std::make_shared<PigSolverDevice>(m_pigConfig, m_use_gpu);
			mp_bodysolverdevice[i]->m_gtscale = m_given_scales[i];
			mp_bodysolverdevice[i]->setParams(m_params); 
			mp_bodysolverdevice[i]->setCameras(m_camsUndist);
			//mp_bodysolver[i]->InitNodeAndWarpField();
			mp_bodysolverdevice[i]->setRenderer(mp_renderEngine);
			mp_bodysolverdevice[i]->m_undist_mask_chamfer = mp_sceneData->m_undist_mask_chamfer;
			mp_bodysolverdevice[i]->m_scene_mask_chamfer = mp_sceneData->m_scene_mask_chamfer;
			mp_bodysolverdevice[i]->m_pig_id = i;
			std::cout << "init model " << i << std::endl;
		}
	}
	m_skels3d.resize(m_pignum); 
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
	for (int i = 0; i < m_pignum; i++)
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

// in use. 
void FrameSolver::optimizeSilWithAnchor(int maxIterTime, int startIter)
{
	int iter = startIter;
	std::vector<int> totalIters(m_pignum, 0); 
	std::vector<float> deltas(m_pignum, 1);

	for (; iter < maxIterTime; iter++)
	{
		//std::cout << "iter: " << iter << " ..... " << std::endl; 
		if (m_params.m_w_sil_term > 0)
		{
			if (iter == 0)
			{
				renderInteractDepth(true);
				computeIOUs();
			}
			else {
				renderInteractDepth(true);
			}
		}
#ifdef DEBUG_SIL
		//for (int pid = 0; pid < 4; pid++)
		//{
		//	std::cout << pid << " :: ";
		//	for (int i = 0; i < m_camNum; i++)
		//	{
		//		std::cout << m_ious[pid][i] << ", ";
		//	}
		//	std::cout << std::endl; 
		//}
		// render interactive mask 

		std::vector<cv::Mat> pseudos;
		cv::Mat depth_cv(cv::Size(1920, 1080), CV_32FC1);
		for (int view = 0; view < m_camNum; view++)
		{
			cudaMemcpy(depth_cv.data, d_interDepth[view], 1920 * 1080 * sizeof(float), cudaMemcpyDeviceToHost);
			cv::Mat colored = pseudoColor(depth_cv);
			pseudos.push_back(colored);
		}
		cv::Mat pack_pseudo;
		packImgBlock(pseudos, pack_pseudo);
		std::stringstream ss_pseudo;
		ss_pseudo << m_result_folder << "/debug/" << std::setw(6) << std::setfill('0') << iter << "_depthinteract.jpg";
		cv::imwrite(ss_pseudo.str(), pack_pseudo);

		//cv::Mat blended;
		//cv::Mat pack_det;
		//packImgBlock(color_mask_dets, pack_det);
		//blended = pack_pseudo * 0.5 + pack_det * 0.5;
		//std::stringstream ss;
		//ss << "D:/results/paper_teaser/0704_demo/debug/" << std::setw(6) << std::setfill('0')
		//	<< iter << "_blend.jpg";
		//cv::imwrite(ss.str(), blended);

		cv::Mat output;
		packImgBlock(m_interMask, output);
		std::stringstream ss;
		ss << m_result_folder << "/debug/mask_" << m_frameid << "_iter_" << iter << ".png";
		cv::imwrite(ss.str(), output);
		//cv::Mat output2;
		//packImgBlock(m_masksMatched, output2);
		//output2 = output2 * 10;
		//std::stringstream ss1;
		//ss1 << m_result_folder << "/debug/maskdet_" << m_frameid << "_iter_" << iter << ".png";
		//cv::imwrite(ss1.str(), output2);

		m_skels3d.resize(m_pignum); 
		for (int i = 0; i < m_pignum; i++)
		{
			m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host();
		}
		cv::Mat reproj = visualizeProj(-1, false);
		std::stringstream ss_proj;
		ss_proj << m_result_folder << "/debug/proj_" << m_frameid << "_iter_" << iter << ".png";
		cv::imwrite(ss_proj.str(), reproj);

#endif 
		for (int i = 0; i < 4; i++)mp_bodysolverdevice[i]->map_reduced_vertices(); 
		for (int i = 0; i < 4; i++)
		{
			mp_bodysolverdevice[i]->m_other_pigs_reduced_vertices.resize(3);
			mp_bodysolverdevice[i]->m_other_centers.resize(3);
			for (int j = 0; j < 4; j++)
			{
				if (j == i) continue;
				if (j < i) {
					mp_bodysolverdevice[i]->m_other_centers[j] = mp_bodysolverdevice[j]->GetJoints()[2];
					mp_bodysolverdevice[i]->m_other_pigs_reduced_vertices[j] = mp_bodysolverdevice[j]->m_reduced_vertices;
				}
				else
				{
					mp_bodysolverdevice[i]->m_other_centers[j - 1] = mp_bodysolverdevice[j]->GetJoints()[2];
					mp_bodysolverdevice[i]->m_other_pigs_reduced_vertices[j - 1] = mp_bodysolverdevice[j]->m_reduced_vertices;
				}
			}
		}
		for (int pid = 0; pid < m_pignum; pid++)
		{
			if (mp_bodysolverdevice[pid]->m_isUpdated) continue; 
			if (deltas[pid] < m_terminal_thresh) continue; 
			if (m_params.m_w_sil_term > 0)
			{
				mp_bodysolverdevice[pid]->o_ious = m_ious[pid];
			}

			deltas[pid] = mp_bodysolverdevice[pid]->optimizePoseSilWithAnchorOneStep(iter);
			totalIters[pid]++;
		}
	}
	std::cout << "Iters: [";
	for (int i = 0; i < m_pignum; i++)
		std::cout << totalIters[i] << ",";
	std::cout << "]" << std::endl;
}

void FrameSolver::saveAnchors(std::string folder)
{
	std::stringstream ss; 
	ss << folder << "/anchor_" << std::setw(6) << std::setfill('0') << m_frameid <<
		".txt"; 
	std::ofstream outfile(ss.str()); 
	for(int i = 0; i < m_pignum; i++)
	    outfile << mp_bodysolverdevice[i]->m_anchor_id << std::endl; 
	outfile.close(); 
}

void FrameSolver::loadAnchors(std::string folder, bool andsolve)
{
	MAMMAL_Step1_setsource(); 
	MAMMAL_Step2_loadanchor(); 

	if (andsolve)
	{
		//MAMMAL_Step2_optimanchor(); 
		MAMMAL_Step4_fitrawsource(m_initialization_iters); 
		//MAMMAL_Step3_reassoc_type2(); 
		//MAMMAL_Step4_fitreassoc(); 
		MAMMAL_Step5_postprocess(); 
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
	m_keypoints_associated.resize(m_pignum); 
	m_skelVis.resize(m_pignum);
	renderInteractDepth();
	for (int i = 0; i < m_pignum; i++)
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
				m_keypoints_associated[i][camid][k] = Eigen::Vector3f::Zero(); 
			}
		}
	}

	//std::cout << "CHECK VISI: " << std::endl;
	//std::cout << m_skelVis[1][3][2] << std::endl; 
	//std::cout << m_skelVis[2][8][6] << std::endl; 
	//std::cout << m_skelVis[3][8][6] << std::endl; 

	if (m_skels3d.size() < 1) m_skels3d.resize(m_pignum); 

	for (int i = 0; i < m_pignum; i++)
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
			for (int pid = 0; pid < m_pignum; pid++)
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
			for (int pid = 0; pid < m_pignum; pid++)
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

cv::Mat FrameSolver::visualizeReassociation(
	std::vector<int> ids, 
	int viewid, bool withImg
)
{
	cv::Mat rawimg;
	rawimg.create(cv::Size(1920, 1080), CV_8UC3);
	rawimg.setTo(cv::Scalar(255, 255, 255));

	cv::Mat halfraw;
	if (withImg)
		halfraw = blend_images(rawimg, m_imgsUndist[viewid], 0.5);
	else
		halfraw = rawimg;

	for(int k = 0; k < ids.size(); k++)
	{
		int id = ids[k];
		int colorid = m_pig_names[id];
		for (int i= 0; i < m_matched[id].view_ids.size(); i++)
		{
			if (m_matched[id].view_ids[i] != viewid) continue; 
			Eigen::Vector3i color;
			
			color(0) = m_CM[colorid](2);
			color(1) = m_CM[colorid](1);
			color(2) = m_CM[colorid](0);
			int camid = m_matched[id].view_ids[i];
			
			//my_draw_box(halfraw, m_matched[id].dets[i].box, color);

			if (m_matched[id].dets[i].mask.size() > 0)
				my_draw_mask(halfraw, m_matched[id].dets[i].mask, color, 0.5);
		}

		Eigen::Vector3i color;
		color(0) = m_CM[colorid](2);
		color(1) = m_CM[colorid](1);
		color(2) = m_CM[colorid](0);
		drawSkelMonoColor(halfraw, m_keypoints_associated[id][viewid], color, m_topo);
		
	}

	return halfraw;
}

cv::Mat FrameSolver::visualizeVisibility()
{
	std::vector<cv::Mat> imgdata;
	cloneImgs(m_imgsUndist, imgdata);
	reproject_skels();
	std::cout << "m_projs.size(): " << m_projs.size() << std::endl; 
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int id = 0; id < m_projs[camid].size(); id++)
		{
			std::vector<Eigen::Vector3f> joints = m_projs[camid][id];
			for (int i = 0; i < m_topo.joint_num; i++)
			{
				if (m_skelVis[id][camid][i] == 0) m_projs[camid][id][i](2) = 0; 
			}
			Eigen::Vector3i color;

			int colorid = m_pig_names[id];
			color(0) = m_CM[colorid](2);
			color(1) = m_CM[colorid](1);
			color(2) = m_CM[colorid](0);
			drawSkelMonoColor(imgdata[camid], m_projs[camid][id], color, m_topo);
		}
	}

	cv::Mat packed;
	packImgBlock(imgdata, packed);
	return packed;
}

cv::Mat FrameSolver::visualizeSwap()
{
	std::vector<cv::Mat> swap_list(m_pignum); 
	for (int i = 0; i < m_pignum; i++)
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
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->m_isReAssoc = true;
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
}

cv::Mat FrameSolver::visualizeRawAssoc()
{
	std::vector<cv::Mat> imglist(m_pignum); 
	for (int i = 0; i < m_pignum; i++)
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
			for (int i = 0; i < m_topo.joint_num; i++)
				m_detTracked[camid][k][i] = -1; 
		}
	}

	m_modelTracked.resize(m_pignum);
	for (int pid = 0; pid < m_pignum; pid++)
	{
		m_modelTracked[pid].resize(m_camNum);
		for (int camid = 0; camid < m_modelTracked[pid].size(); camid++)
		{
			m_modelTracked[pid][camid].resize(m_topo.joint_num, -1);
			for (int i = 0; i < m_topo.joint_num; i++)
			{
				m_modelTracked[pid][camid][i] = -1; 
			}
		}
	}

	if (m_skels3d.size() < 1) m_skels3d.resize(m_pignum);

	for (int i = 0; i < m_pignum; i++)
	{
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host();
	}
	reproject_skels();

	for (int pid = 0; pid < m_pignum; pid++)
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
				if ((pointDetect.segment<2>(0) - pointProj.segment<2>(0)).norm() < 15)
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
		for(int pid = 0; pid < m_pignum; pid++)
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
			Eigen::Vector3i color;
			int colorid = m_pig_names[pid];
			color(0) = m_CM[colorid](2);
			color(1) = m_CM[colorid](1);
			color(2) = m_CM[colorid](0);
			drawSkelMonoColor(track_list[camid], keypoints, color, m_topo);
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
				if ((pool[i].segment<2>(0) - ref[jointid][j].segment<2>(0)).norm() < 20)
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
					if ((pool[i].segment<2>(0) - ref[jid][j].segment<2>(0)).norm() < 20)
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
	m_keypoints_pool.clear(); 
	m_keypoints_pool.resize(m_camNum);
	keypoints_trackedPool.resize(m_camNum);
	for (int view = 0; view < m_camNum; view++)
	{
		m_keypoints_pool[view].resize(m_topo.joint_num);
		for (int i = 0; i < m_keypoints_pool[view].size(); i++)
			m_keypoints_pool[view][i].clear(); 
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

#ifdef VIS_ASSOC_STEP
	for (int view = 0; view < m_camNum; view++)
	{
		std::cout << "view: " << view << std::endl; 
		for (int jid = 0; jid < m_topo.joint_num; jid++)
		{
			std::cout << " -- jointid: " << jid << std::endl; 
			for (int k = 0; k < m_keypoints_pool[view][jid].size(); k++)
			{
				std::cout << "    -- " <<  m_keypoints_pool[view][jid][k].transpose() << std::endl; 
			}
		}
	}
	std::cout << "----------------------------------------" << std::endl; 
#endif 
}


void FrameSolver::reAssocKeypointsWithoutTracked()
{
	m_keypoints_associated.resize(m_pignum);
	m_skelVis.resize(m_pignum);
	renderInteractDepth();
	for (int i = 0; i < m_pignum; i++)
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
				else
				{
					m_keypoints_associated[i][camid][k] = Eigen::Vector3f::Zero();
				}
			}
		}
	}

#ifdef VIS_ASSOC_STEP
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
	cv::Mat output = visualizeSwap();
	cv::imwrite("H:/pig_results_anchor/swap/" + std::to_string(m_frameid) + "_step1.png", output); 
#endif 
	for (int camid = 0; camid < m_camNum; camid++)
	{
#ifdef VIS_ASSOC_STEP
		std::cout << "camid: " << camid << std::endl; 
#endif 
		Camera cam = m_camsUndist[camid];
		// associate for each camera 
		for (int i = 0; i < m_topo.joint_num; i++)
		{
#ifdef VIS_ASSOC_STEP
			std::cout << "joint: " << i << std::endl; 
#endif 
			std::vector<int> id_table;
			for (int pid = 0; pid < m_pignum; pid++)
			{
				if (m_keypoints_associated[pid][camid][i](2) > 0) continue; 
				if (m_skelVis[pid][camid][i] > 0)
					id_table.push_back(pid);
			}
			int M = id_table.size(); // candidate number for associate  
			int N = m_keypoints_pool[camid][i].size();
			
			Eigen::MatrixXf sim(M, N);
			Eigen::MatrixXf sim2(M, N);
			for (int rowid = 0; rowid < M; rowid++)
			{
				for (int colid = 0; colid < N; colid++)
				{
					sim(rowid, colid) = (m_keypoints_pool[camid][i][colid].segment<2>(0)
						- m_projs[camid][id_table[rowid]][i].segment<2>(0)).norm();

					Eigen::Vector3f plocal = cam.R * m_skels3d[id_table[rowid]][i] + cam.T;
					float d = p2ldist(plocal, m_keypoints_pool[camid][i][colid]);
					sim2(rowid, colid) = d; 
					if (sim(rowid, colid) > 80) sim(rowid, colid) = 80;
				}
			}
#ifdef VIS_ASSOC_STEP
			std::cout << "sim: " << std::endl << sim << std::endl; 
			std::cout << "sim2: " << std::endl << sim2 << std::endl; 
#endif 
			std::vector<int> assign = solveHungarian(sim);
			for (int rowid = 0; rowid < M; rowid++)
			{
				int pid = id_table[rowid];
				int colid = assign[rowid];
				if (colid < 0) continue;
				if (sim(rowid, colid) >= 80) continue;
				m_keypoints_associated[pid][camid][i] = m_keypoints_pool[camid][i][colid];
				m_keypoints_pool[camid][i][colid] = Eigen::Vector3f::Zero(); 
			}
		}
	}

#ifdef  VIS_ASSOC_STEP
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
	output = visualizeSwap();
	cv::imwrite("H:/pig_results_anchor/swap/" + std::to_string(m_frameid) + "_step2.png", output);
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
			const Camera& cam = m_camsUndist[camid];
			std::vector<Eigen::Vector3f> remain_pool;
			for (int i = 0; i < ids_to_swap.size(); i++)
			{
				int id = ids_to_swap[i];
				for (int candid = 0; candid < m_keypoints_pool[camid][id].size(); candid++)
				{
					if (m_keypoints_pool[camid][id][candid](2) < m_topo.kpt_conf_thresh[id])continue;
					else remain_pool.push_back(m_keypoints_pool[camid][id][candid]);
				}
			}
			std::vector<int> pig_id_table;
			std::vector<int> joint_id_table;
			std::vector<Eigen::Vector3f> projPool;
			std::vector<Eigen::Vector3f> pool3d; 
			for (int pid = 0; pid < m_pignum; pid++)
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
						pool3d.push_back(m_skels3d[pid][id]);
					}
				}
			}
			int M = remain_pool.size();
			int N = projPool.size();
			Eigen::MatrixXf sim = Eigen::MatrixXf::Zero(M, N);
			Eigen::MatrixXf sim2 = Eigen::MatrixXf::Zero(M, N); 
			for (int i = 0; i < M; i++)
			{
				for (int j = 0; j < N; j++)
				{
					float dist = (remain_pool[i].segment<2>(0) - projPool[j].segment<2>(0)).norm(); 
					sim(i, j) = dist > 80 ? 80 : dist;
					Eigen::Vector3f plocal = cam.R * pool3d[j] + cam.T; 
					float d = p2ldist(plocal, remain_pool[i]);
					sim2(i, j) = d; 
				}
			}

#ifdef VIS_ASSOC_STEP
			std::cout << "camid: " << camid << std::endl; 
			for (int t = 0; t < pig_id_table.size(); t++)
			{
				std::cout << "(" << pig_id_table[t] << "," << joint_id_table[t] << ") ";
			}
			for (int i = 0; i < remain_pool.size(); i++)
			{
				std::cout << "  -- " << remain_pool[i].transpose() << std::endl; 
			}
			std::cout << std::endl << " [sim]: " <<  sim << std::endl;
			std::cout << " [sim2]: " << std::endl << sim2 << std::endl; 
#endif 

			std::vector<int> match = solveHungarian(sim);
			for (int i = 0; i < match.size(); i++)
			{
				if (match[i] < 0) continue;
				if (sim(i, match[i]) >= 80) continue;
				int j = match[i];
				m_keypoints_associated[pig_id_table[j]][camid][joint_id_table[j]]
					= remain_pool[i];
			}
		}
	}

#ifdef VIS_ASSOC_STEP
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
	
	output = visualizeSwap();
	cv::imwrite("H:/pig_results_anchor/swap/" + std::to_string(m_frameid) + "_step3.png", output);
#endif 
}

void FrameSolver::reAssocWithoutTracked()
{
	m_keypoints_pool.clear(); 
	m_keypoints_associated.clear(); 
	m_skelVis.clear(); 
	m_detTracked.clear(); 
	m_modelTracked.clear(); 

	determineTracked(); 
	splitDetKeypointsWithoutTracked();
	reAssocKeypointsWithoutTracked();
	
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->m_isReAssoc = true;
		mp_bodysolverdevice[i]->m_keypoints_reassociated = m_keypoints_associated[i];
	}
}

void FrameSolver::solve_scales()
{
	std::vector<float> scales(m_pignum, 0); 
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]);
		float scale = mp_bodysolverdevice[i]->computeScale();
		scales[i] = scale; 
	}
	std::stringstream ss; 
	ss << m_result_folder << "/scales/scale_" << std::setw(6) << std::setfill('0') << m_frameid << ".txt"; 
	std::ofstream scalefile(ss.str());
	for (int i = 0; i < m_pignum; i++)
	{
		scalefile << scales[i] << " "; 
	}
	scalefile.close(); 
}

void FrameSolver::computeIOUs()
{
	std::vector<Eigen::Vector3f> id_colors = {
	{1.0f, 0.0f,0.0f},
{0.0f, 1.0f, 0.0f},
{0.0f, 0.0f, 1.0f},
{1.0f, 1.0f, 0.0f}
	};
	std::vector<Eigen::Vector3i> id_colors_cv = {
		{0,0,255}, 
	{0,255,0},
	{255,0,0},
	{0,255,255}
	};

	std::vector<std::vector<float >  > ious; 
	ious.resize(m_pignum); 
	
	for (int pid = 0; pid < m_pignum; pid++)
	{
		ious[pid].resize(m_camNum, 0); 
		for (int camid = 0; camid < m_camNum; camid++)
		{
			float I = 0; 
			float U = 0; 
			for (int x = 0; x < 1920; x++)
			{
				for (int y = 0; y < 1080; y++)
				{
					if (mp_sceneData->m_undist_mask.at<uchar>(y, x) == 0) continue; 
					if (mp_sceneData->m_scene_masks[camid].at<uchar>(y, x) > 0) continue; 
					if ( (m_interMask[camid].at<cv::Vec3b>(y, x)[0] == id_colors_cv[pid](0)
						&& m_interMask[camid].at<cv::Vec3b>(y, x)[1] == id_colors_cv[pid](1)
						&& m_interMask[camid].at<cv::Vec3b>(y, x)[2] == id_colors_cv[pid](2)
						) && m_masksMatched[camid].at<uchar>(y,x) == (1 << pid)
						)
					{
						I += 1;
					}
					if ((m_interMask[camid].at<cv::Vec3b>(y, x)[0] == id_colors_cv[pid](0)
						&& m_interMask[camid].at<cv::Vec3b>(y, x)[1] == id_colors_cv[pid](1)
						&& m_interMask[camid].at<cv::Vec3b>(y, x)[2] == id_colors_cv[pid](2)
						) || m_masksMatched[camid].at<uchar>(y, x) == (1 << pid)
						)
					{
						U += 1; 
					}
				}
			}
			if (I < 1) ious[pid][camid] = 0;
			else
			{
				ious[pid][camid] = float(I) / float(U);
				//std::cout << "cam " << camid << " pig " << pid <<  " I: " << I << " U: " << U << 
				//	" iou: " << ious[pid][camid] << std::endl;
			}
		}
	}

	m_ious = ious; 
}

void FrameSolver::save_joints()
{
	for (int pid = 0; pid < m_pignum; pid++)
	{
		std::stringstream ss;
		ss << m_result_folder << "/joints_23_smth_center/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << m_frameid << ".txt";
		std::ofstream outputfile(ss.str());
		auto data = mp_bodysolverdevice[pid]->getRegressedSkel_host(); 
		for (int i = 0; i < data.size(); i++)
		{
			outputfile << data[i].transpose() << std::endl;
		}
		outputfile.close();
	}
}


/* 
20210227: An Liang
This function must run at the postprocessing step of the whole 
pipeline. And all correspnodences could be used to optimize.
Before this function, these functions should be run: 
renderMaskColor
renderFaceIndex
*/
/*
currently, it supports at most 4 pigs
*/
int determineColorid(const cv::Vec3b& pixel)
{
	std::vector<Eigen::Vector3i> id_colors_cv = {
		{0,0,255},
		{0,255,0},
		{255,0,0},
		{0,255,255}
	};
	for (int i = 0; i < 4; i++)
	{
		if (id_colors_cv[i](0) == pixel[0] &&
			id_colors_cv[i](1) == pixel[1] &&
			id_colors_cv[i](2) == pixel[2])
			return i; 
	}
	return -1; 
}


void FrameSolver::resetSolverStateMarker()
{
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->resetStateMarker(); 
	}
}

void FrameSolver::save_skels()
{
	for (int pid = 0; pid < m_pignum; pid++)
	{
		std::stringstream ss;
		ss << m_result_folder << "/skels/" << "pig_" << pid << "_" << std::setw(6) << std::setfill('0') << m_frameid << ".txt";

		std::ofstream os(ss.str());
		if (!os.is_open())
		{
			std::cout << "cant not open " << ss.str() << std::endl;
			return;
		}
		for (int k = 0; k < m_skels3d[pid].size(); k++)
		{
			os << m_skels3d[pid][k].transpose() << std::endl;
		}
		os.close();
	}
}

void FrameSolver::read_skels(std::string folder, int frameid)
{
	m_skels3d.resize(4); 
	
	for (int pid = 0; pid < 4; pid++)
	{
		m_skels3d[pid].resize(23); 
		std::stringstream ss;
		ss << folder << "/" << "pig_" << pid << "_" << std::setw(6) << std::setfill('0') << m_frameid << ".txt";
		std::ifstream is(ss.str()); 
		if (!is.is_open())
		{
			ss.str(""); 
			ss << folder << "/" << "pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << m_frameid << ".txt";
			is.open(ss.str()); 
		}
		for (int k = 0; k < m_skels3d[pid].size(); k++)
		{
			float x, y, z;
			is >> x >> y >> z; 
			m_skels3d[pid][k] = Eigen::Vector3f(x, y, z); 
		}
		is.close(); 
	}
}

// 2021/09/28: for each experiment, log its config
void FrameSolver::saveConfig()
{
	boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time(); 
	std::stringstream ss_time; 
	ss_time << timeLocal.date().year() << "-" << timeLocal.date().month() << "-"
		<< timeLocal.date().day() << "++" << std::setw(2) << std::setfill('0') 
		<< timeLocal.time_of_day().hours() << "-" << std::setw(2) << std::setfill('0')
		<< timeLocal.time_of_day().minutes() << "-" << std::setw(2) << std::setfill('0')
		<< timeLocal.time_of_day().seconds();
	std::string str_time = ss_time.str(); 
	
	std::stringstream ss_config_file;
	ss_config_file << m_result_folder << "/config_" << str_time << ".json"; 

	std::filesystem::copy_file(m_input_configfile, ss_config_file.str()); 

}


cv::Mat FrameSolver::tmp_visualizeRawDet(int viewid, int pid)
{
	cv::Mat rawimg = m_imgsUndist[viewid].clone(); 

		//Eigen::Vector3i color(71, 109, 245);
		//Eigen::Vector3i color(204, 175, 143);

	if (m_detUndist[viewid].size() > 0)
	{
		for (int candid = 0; candid < m_detUndist[viewid].size(); candid++)
		{
			if (pid >= 0 && candid != pid) continue; 
			int colorid = candid;
			Eigen::Vector3i color;
			color(0) = m_CM[colorid](2);
			color(1) = m_CM[colorid](1);
			color(2) = m_CM[colorid](0);
			//my_draw_box(rawimg, m_detUndist[viewid][candid].box, color); 
			my_draw_mask(rawimg, m_detUndist[viewid][candid].mask, Eigen::Vector3i(71, 109, 245), 0, true);
			//drawSkelMonoColor(rawimg, m_detUndist[viewid][candid].keypoints, color, m_topo);
			drawSkelDebug(rawimg, m_detUndist[viewid][candid].keypoints, m_topo); 
		}
	}
	
	return rawimg; 
}

cv::Mat FrameSolver::tmp_visualizeRawDetPure(int viewid, bool withImg)
{
	cv::Mat rawimg;
	rawimg.create(cv::Size(1920, 1080), CV_8UC3); 
	rawimg.setTo(cv::Scalar(255, 255, 255)); 

	cv::Mat halfraw; 
	if(withImg)
		halfraw = blend_images(rawimg, m_imgsUndist[viewid], 0.5); 
	else
		halfraw = rawimg; 

	//Eigen::Vector3i color(71, 109, 245);
	//Eigen::Vector3i color(244, 170, 2);
	Eigen::Vector3i color(0, 0, 0); 


	if (m_detUndist[viewid].size() > 0)
	{
		for (int candid = 0; candid < m_detUndist[viewid].size(); candid++)
		{
			//int colorid = candid;
			//Eigen::Vector3i color;
			//color(0) = m_CM_blend[colorid](2);
			//color(1) = m_CM_blend[colorid](1);
			//color(2) = m_CM_blend[colorid](0);
			my_draw_box(halfraw, m_detUndist[viewid][candid].box, color);
			my_draw_mask(halfraw, m_detUndist[viewid][candid].mask, color, 0, true);
			//drawSkelMonoColor(rawimg, m_detUndist[viewid][candid].keypoints, color, m_topo);
			drawSkelDebug(halfraw, m_detUndist[viewid][candid].keypoints, m_topo);
		}
	}
	
	return halfraw;
}

cv::Mat FrameSolver::tmp_visualizeIdentity2D(int viewid, int vid)
{
	cv::Mat rawimg;
	rawimg.create(cv::Size(1920, 1080), CV_8UC3);
	rawimg.setTo(cv::Scalar(255, 255, 255));
	for (int id = 0; id < m_pignum; id++)
	{
		int colorid = m_pig_names[id];
		if (vid > -1 && colorid != vid) continue; 
		for (int i = 0; i < m_matched[id].view_ids.size(); i++)
		{
			if (viewid >= 0 && m_matched[id].view_ids[i] != viewid) continue;
			Eigen::Vector3i color;
			color(0) = m_CM[colorid](2);
			color(1) = m_CM[colorid](1);
			color(2) = m_CM[colorid](0);
			int camid = m_matched[id].view_ids[i];
			if (m_matched[id].dets[i].mask.size() > 0)
				my_draw_mask(rawimg, m_matched[id].dets[i].mask, color, 0.5); 
			//if (m_matched[id].dets[i].keypoints.size() > 0)
			//	drawSkelMonoColor(rawimg, m_matched[id].dets[i].keypoints, color, m_topo);
			//my_draw_box(m_imgsDetect[camid], m_matched[id].dets[i].box, color);
		}
	}
	return rawimg; 
}