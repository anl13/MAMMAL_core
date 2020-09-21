#include "framedata.h" 
#include <json/json.h> 
#include <math.h> 
#include <algorithm>
#include <json/writer.h> 

void FrameData::setCamIds(std::vector<int> _camids)
{
    m_camids = _camids; 
    m_camNum = m_camids.size(); 
}

void FrameData::configByJson(std::string jsonfile)
{
    Json::Value root;
    Json::CharReaderBuilder rbuilder; 
    std::string errs; 
    std::ifstream instream(jsonfile); 
    if(!instream.is_open())
    {
        std::cout << "can not open " << jsonfile << std::endl; 
        exit(-1); 
    }
    bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs); 
    if(!parsingSuccessful)
    {
        std::cout << "Fail to parse \n" << errs << std::endl; 
        exit(-1); 
    }
    m_sequence     = root["sequence"].asString(); 
    m_keypointsDir = m_sequence + "/keypoints_hrnet_pr/"; 
    m_imgDir       = m_sequence + "/images/";  
    m_boxDir       = m_sequence + "/boxes_pr/"; 
    m_maskDir      = m_sequence + "/masks_pr/";
    m_camDir       = root["camfolder"].asString(); 
    m_imgExtension = root["imgExtension"].asString(); 
    m_startid      = root["startid"].asInt(); 
    m_framenum     = root["framenum"].asInt(); 
	m_epi_thres = root["epipolar_threshold"].asDouble();
    m_epi_type     = root["epipolartype"].asString(); 
    m_boxExpandRatio = root["box_expand_ratio"].asDouble(); 
    m_skelType     = root["skel_type"].asString(); 
    m_topo         = getSkelTopoByType(m_skelType); 
	m_match_alg = root["match_alg"].asString(); 
	m_pigConfig = root["pig_config"].asString(); 
	m_use_gpu = root["use_gpu"].asBool(); 
	m_solve_sil_iters = root["solve_sil_iters"].asInt(); 

    std::vector<int> camids; 
    for(auto const &c : root["camids"])
    {
        int id = c.asInt(); 
        camids.push_back(id); 
    }
    setCamIds(camids); 

    instream.close(); 

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


	readSceneMask();
}

void FrameData::readSceneMask()
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

void FrameData::fetchData()
{
    if(m_frameid < 0)
    {
        std::cout << "Error: wrong frame id " << std::endl;
        exit(-1); 
    }
    readCameras(); 

    readImages(); 
    undistImgs(); 

    readKeypoints(); 
    undistKeypoints(); 
	clean_by_mask_chamfer();

    readBoxes();
    processBoxes(); 

    readMask(); 
    undistMask(); 

    assembleDets(); 

	
    //detNMS(); 

	// read backgrounds 

}

void FrameData::readKeypoints() // load hrnet keypoints
{
    std::string jsonDir = m_keypointsDir;
    std::stringstream ss; 
    ss << jsonDir << std::setw(6) << std::setfill('0') << m_frameid << ".json";
    std::string jsonfile = ss.str(); 

    Json::Value root; 
    Json::CharReaderBuilder rbuilder; 
    std::string errs; 
    std::ifstream is(jsonfile); 
    if (!is.is_open())
    {
        std::cout << "can not open " << jsonfile << std::endl; 
        exit(-1); 
    }
    bool parsingSuccessful = Json::parseFromStream(rbuilder, is, &root, &errs);
    if(!parsingSuccessful)
    {
        std::cout << "Fail to parse doc \n" << errs << std::endl;
        exit(-1); 
    } 

    m_keypoints.clear(); 

    for(int camid = 0; camid < m_camNum; camid++)
    {
        Json::Value c = root[std::to_string(m_camids[camid])]; 
        vector<vector<Eigen::Vector3f> > aframe; 
        int cand_num = c.size(); 
        for(int candid = 0; candid < cand_num; candid++)
        {
            if(candid >=4) break; 
            vector<Eigen::Vector3f> pig; 
            pig.resize(m_topo.joint_num); 
            for(int pid = 0; pid < m_topo.joint_num; pid++)
            {
                Eigen::Vector3f v;
                for(int idx = 0; idx < 3; idx++)
                {
                    v(idx) = c[candid][pid*3+idx].asDouble(); 
                }
				if(v(2) > m_topo.kpt_conf_thresh[pid])
					pig[pid] = v; 
				else pig[pid] = Eigen::Vector3f::Zero(); 
            }
            aframe.push_back(pig);
        }
        m_keypoints.push_back(aframe); 
    }
    is.close(); 
}

void FrameData::readBoxes()
{
    std::string jsonDir = m_boxDir;
    std::stringstream ss; 
    ss << jsonDir << "/" << std::setw(6) << std::setfill('0') << m_frameid << ".json";
    std::string jsonfile = ss.str(); 
    // parse
    Json::Value root; 
    Json::CharReaderBuilder rbuilder; 
    std::string errs; 
    std::ifstream is(jsonfile); 
    if (!is.is_open())
    {
        std::cout << "can not open " << jsonfile << std::endl; 
        exit(-1); 
    }
    bool parsingSuccessful = Json::parseFromStream(rbuilder, is, &root, &errs);
    if(!parsingSuccessful)
    {
        std::cout << "Fail to parse doc \n" << errs << std::endl;
        exit(-1); 
    } 
    // load data
    m_boxes_raw.clear(); 
    m_boxes_raw.resize(m_camNum); 
    for(int i = 0; i < m_camNum; i++)
    {
        int camid = m_camids[i]; 
        Json::Value c = root[std::to_string(camid)]; 
        int boxnum = c.size(); 
        std::vector<Eigen::Vector4f> bb; 
        for(int bid = 0; bid < boxnum; bid++)
        {
            if(bid >= 4) break; // remain only 4 top boxes 
            Json::Value box_jv = c[bid]; 
            Eigen::Vector4f B; 
            for(int k = 0; k < 4; k++)
            {
                double x = box_jv[k].asDouble(); 
                B(k) = x; 
            }
            bb.push_back(B); 
        }
        m_boxes_raw[i] = bb; 
    }
}

void FrameData::processBoxes()
{
    m_boxes_processed.clear(); 
    m_boxes_processed.resize(m_camNum); 
    for(int cid = 0; cid < m_camNum; cid++)
    {
        int boxnum = m_boxes_raw[cid].size(); 
        m_boxes_processed[cid].resize(boxnum); 
        for(int bid = 0; bid < boxnum; bid++)
        {
            Eigen::Vector4f box = my_undistort_box(
                m_boxes_raw[cid][bid], m_cams[cid], m_camsUndist[cid]
            ); 
            m_boxes_processed[cid][bid] = expand_box(box, m_boxExpandRatio); 
        }
    }
}

void FrameData::undistKeypoints()
{
    int camNum = m_keypoints.size(); 
    m_keypoints_undist = m_keypoints; 
    for(int camid = 0; camid < camNum; camid++)
    {
        Camera cam = m_cams[camid]; 
        Camera camnew = m_camsUndist[camid]; 
        int candnum = m_keypoints[camid].size(); 
        for(int candid = 0; candid < candnum; candid++)
        {
            my_undistort_points(m_keypoints[camid][candid], m_keypoints_undist[camid][candid], cam, camnew); 
        }
    }
}

void FrameData::undistImgs()
{
    m_imgsUndist.resize(m_camNum); 
    for(int i = 0; i < m_camNum; i++)
    {
        my_undistort(m_imgs[i], m_imgsUndist[i], m_cams[i], m_camsUndist[i]); 
    }
}

void FrameData::readImages()
{
    m_imgs.clear(); 
    for(int camid = 0; camid < m_camNum; camid++)
    {
        std::stringstream ss; 
        ss << m_imgDir << "/cam" << m_camids[camid] << "/" << std::setw(6) << std::setfill('0') << m_frameid << "." << m_imgExtension;
		cv::Mat img = cv::imread(ss.str()); 
		if (img.empty())
		{
			std::cout << "can not read image " << ss.str() << std::endl;
			exit(-1); 
		}
		m_imgs.emplace_back(img);
    }


}

void FrameData::readCameras()
{
    m_cams.clear(); 
    m_camsUndist.clear(); 
    for(int camid = 0; camid < m_camNum; camid++)
    {
        std::stringstream ss; 
        ss << m_camDir << std::setw(2) << std::setfill('0') << m_camids[camid] << ".txt";
        std::ifstream camfile; 
        camfile.open(ss.str());
        if(!camfile.is_open())
        {
            std::cout << "can not open file " << ss.str() << std::endl; 
            exit(-1); 
        }
        Eigen::Vector3f rvec, tvec; 
        for(int i = 0; i < 3; i++) {
            double a;
            camfile >> a; 
            rvec(i) = a; 
        }
        for(int i = 0; i < 3; i++)
        {
            double a; 
            camfile >> a; 
            tvec(i) = a; 
        }
        Camera cam = Camera::getDefaultCameraRaw(); 
        cam.SetRT(rvec,  tvec); 
        Camera camUndist = Camera::getDefaultCameraUndist(); 
        camUndist.SetRT(rvec, tvec); 
        m_cams.push_back(cam); 
        m_camsUndist.push_back(camUndist); 
        camfile.close(); 
    }
}

void FrameData::reproject_skels()
{
    m_projs.clear(); 
    int pig_num = m_clusters.size();
    pig_num = pig_num>4? 4:pig_num;  
    m_projs.resize(m_camNum); 
    for(int c = 0; c < m_camNum; c++) m_projs[c].resize(pig_num); 
    
    for(int camid = 0; camid < m_camNum; camid++)
    {
        for(int id = 0; id < pig_num; id++)
        {
            m_projs[camid][id].resize(m_topo.joint_num, Eigen::Vector3f::Zero()); 
            for(int kpt_id = 0; kpt_id < m_topo.joint_num; kpt_id++)
            {
                if(m_skels3d[id][kpt_id].norm() == 0) continue; 
                Eigen::Vector3f p = m_skels3d[id][kpt_id]; 
                m_projs[camid][id][kpt_id] = project(m_camsUndist[camid], p); 
            }
        }
    }
}

cv::Mat FrameData::visualizeSkels2D()
{
    vector<cv::Mat> imgdata; 
    cloneImgs(m_imgsUndist, imgdata); 
    for(int i = 0; i < m_camNum; i++)
    {
        for(int k = 0; k < m_detUndist[i].size(); k++)
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

cv::Mat FrameData::visualizeIdentity2D(int viewid, int vid)
{
    cloneImgs(m_imgsUndist, m_imgsDetect); 
    
    for(int id = 0; id < m_matched.size(); id++)
    {
		if (vid >= 0 && id != vid)continue;
        for(int i = 0; i < m_matched[id].view_ids.size(); i++)
        {
			Eigen::Vector3i color; 
			color(0) = m_CM[id](2);
			color(1) = m_CM[id](1);
			color(2) = m_CM[id](0); 
            int camid = m_matched[id].view_ids[i];
            //int candid = m_matched[id].cand_ids[i];
            //if(candid < 0) continue; 
			if(m_matched[id].dets[i].keypoints.size() > 0)
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
			if(m_unmatched[camid][i].keypoints.size()>0)
			drawSkelMonoColor(m_imgsDetect[camid], m_unmatched[camid][i].keypoints, 5, m_topo);
			my_draw_box(m_imgsDetect[camid], m_unmatched[camid][i].box, color);
			if (m_unmatched[camid][i].mask.size()>0)
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

cv::Mat FrameData::visualizeProj()
{
    std::vector<cv::Mat> imgdata;
    cloneImgs(m_imgsUndist, imgdata); 
	reproject_skels();
    
    for(int camid = 0; camid < m_camNum; camid++)
    {
        for(int id = 0; id < m_projs[camid].size(); id++)
        {
            drawSkelMonoColor(imgdata[camid], m_projs[camid][id], id, m_topo);
        }
    }
    
    cv::Mat packed; 
    packImgBlock(imgdata, packed); 
    return packed;
}

void FrameData::writeSkel3DtoJson(std::string jsonfile)
{
    std::ofstream os;
    os.open(jsonfile); 
    if(!os.is_open())
    {
        std::cout << "file " << jsonfile << " cannot open" << std::endl; 
        return; 
    }

    Json::Value root;
    Json::Value pigs(Json::arrayValue);  
    for(int index=0; index < m_skels3d.size(); index++)
    {
        Json::Value pose(Json::arrayValue); 
        for(int i = 0; i < m_topo.joint_num; i++)
        {
            // if a joint is empty, it is (0,0,0)^T
            pose.append(Json::Value(m_skels3d[index][i](0)) ); 
            pose.append(Json::Value(m_skels3d[index][i](1)) );
            pose.append(Json::Value(m_skels3d[index][i](2)) );
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

void FrameData::readSkel3DfromJson(std::string jsonfile)
{
    Json::Value root;
    Json::CharReaderBuilder rbuilder; 
    std::string errs; 
    std::ifstream instream(jsonfile); 
    if(!instream.is_open())
    {
        std::cout << "can not open " << jsonfile << std::endl; 
        exit(-1); 
    }
    bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs); 
    if(!parsingSuccessful)
    {
        std::cout << "Fail to parse \n" << errs << std::endl; 
        exit(-1); 
    }

    m_skels3d.clear(); 
    for(auto const &pig: root["pigs"])
    {
        std::vector<Eigen::Vector3f> a_pig; 
        a_pig.resize(m_topo.joint_num); 
        for(int index=0; index < m_topo.joint_num; index++)
        {
            double x = pig[index * 3 + 0].asDouble(); 
            double y = pig[index * 3 + 1].asDouble();
            double z = pig[index * 3 + 2].asDouble(); 
            Eigen::Vector3f vec(x,y,z);
            a_pig[index] = vec; 
        }
        m_skels3d.push_back(a_pig);
    }
    instream.close(); 
    std::cout << "read " << jsonfile << " done. " << std::endl; 
}


int FrameData::_compareSkel(const std::vector<Eigen::Vector3f>& skel1, const std::vector<Eigen::Vector3f>& skel2)
{
    int overlay = 0; 
    for(int i = 0; i < m_topo.joint_num; i++)
    {
        Eigen::Vector3f p1 = skel1[i];
        Eigen::Vector3f p2 = skel2[i];
        if(p1(2) < m_topo.kpt_conf_thresh[i] || p2(2) < m_topo.kpt_conf_thresh[i])continue; 
        Eigen::Vector2f diff = p1.segment<2>(0) - p2.segment<2>(0); 
        float dist = diff.norm(); 
        if(dist < 10) overlay ++; 
    }
    return overlay; 
} 
int FrameData::_countValid(const std::vector<Eigen::Vector3f>& skel)
{
    int valid = 0; 
    for(int i = 0; i < skel.size(); i++) 
    {
        if(skel[i](2) >= m_topo.kpt_conf_thresh[i]) valid++; 
    }
    return valid; 
}
void FrameData::detNMS()
{
    // cornor case
    if(m_detUndist.size()==0) return; 
    
	// discard some ones with large overlap 
    for(int camid = 0; camid < m_camNum; camid++)
    {
        // do nms on each view 
        int cand_num = m_detUndist[camid].size(); 
        std::vector<int> is_discard(cand_num, 0); 
        for(int i = 0; i < cand_num; i++)
        {
            for(int j = i+1; j < cand_num; j++)
            {
                if(is_discard[i] > 0 || is_discard[j] > 0) continue; 
                int overlay = _compareSkel(m_detUndist[camid][i].keypoints, 
                    m_detUndist[camid][i].keypoints); 
                int validi = _countValid(m_detUndist[camid][i].keypoints); 
                int validj = _countValid(m_detUndist[camid][j].keypoints); 
                float iou, iou1, iou2;
                IoU_xyxy_ratio(m_detUndist[camid][i].box,m_detUndist[camid][j].box,
                    iou, iou1, iou2); 
                if(overlay >= 3 && (iou1 > 0.8 || iou2>0.8) )
                {
                    if(validi > validj && iou2 > 0.8) is_discard[j] = 1; 
                    else if (validi<validj && iou1 > 0.8) is_discard[i] = 1; 
                }
            }
        }
        std::vector<DetInstance> clean_dets; 
        for(int i = 0; i < cand_num; i++)
        {
            if(is_discard[i] > 0) continue; 
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

void FrameData::readMask()
{
    std::string jsonDir = m_maskDir;
    std::stringstream ss; 
    ss << jsonDir << "/" << std::setw(6) << std::setfill('0') << m_frameid << ".json";
    std::string jsonfile = ss.str(); 
    // parse
    Json::Value root; 
    Json::CharReaderBuilder rbuilder; 
    std::string errs; 
    std::ifstream is(jsonfile); 
    if (!is.is_open())
    {
        std::cout << "can not open " << jsonfile << std::endl; 
        exit(-1); 
    }
    bool parsingSuccessful = Json::parseFromStream(rbuilder, is, &root, &errs);
    if(!parsingSuccessful)
    {
        std::cout << "Fail to parse doc \n" << errs << std::endl;
        exit(-1); 
    } 
    // load data
    m_masks.clear(); 
    m_masks.resize(m_camNum); 

    for(int i = 0; i < m_camNum; i++)
    {
        int camid = m_camids[i]; 
        Json::Value c = root[std::to_string(camid)]; 
        int id_num = c.size(); 
        std::vector<std::vector<std::vector<Eigen::Vector2f> > > masks; 
        for(int bid = 0; bid < id_num; bid++)
        {
            //if(bid >= 4) break; // remain only 4 top boxes 
            Json::Value mask_parts = c[bid]; 
            std::vector<std::vector<Eigen::Vector2f> > M_a_pig;  
            for(int k = 0; k < mask_parts.size(); k++)
            {
                std::vector<Eigen::Vector2f> M_a_part;
                Json::Value mask_a_part = mask_parts[k];  
                for(int p = 0; p < mask_a_part.size(); p++)
                {
                    double x = mask_a_part[p][0][0].asDouble(); 
                    double y = mask_a_part[p][0][1].asDouble(); 
                    M_a_part.push_back(Eigen::Vector2f(x,y)); 
                }
                M_a_pig.push_back(M_a_part); 
            }
            masks.push_back(M_a_pig); 
        }
        m_masks[i] = masks; 
    }
}

void FrameData::undistMask()
{
    m_masksUndist = m_masks; 
    for(int camid = 0; camid < m_camNum; camid++)
    {
        for(int cid = 0; cid < m_masks[camid].size(); cid++)
        {
            for(int partid = 0; partid < m_masks[camid][cid].size(); partid++)
            {
                std::vector<Eigen::Vector3f> points_homo; 
                for(int pid = 0; pid < m_masks[camid][cid][partid].size(); pid++)
                {
                    points_homo.push_back(ToHomogeneous(m_masks[camid][cid][partid][pid]));
                }
                std::vector<Eigen::Vector3f> points_undist; 
                my_undistort_points(points_homo, points_undist, m_cams[camid], m_camsUndist[camid]);
                for(int pid = 0; pid < m_masksUndist[camid][cid][partid].size(); pid++)
                {
                    m_masksUndist[camid][cid][partid][pid] = points_undist[pid].segment<2>(0); 
                }
            }
        }
    }
}

void FrameData::assembleDets()
{
    m_detUndist.resize(m_camNum); 
    for(int camid = 0; camid < m_camNum; camid++)
    {
        int candnum = m_boxes_raw[camid].size(); 
        m_detUndist[camid].resize(candnum); 
        for(int candid = 0; candid < candnum; candid++)
        {
			m_detUndist[camid][candid].valid = true;
            m_detUndist[camid][candid].keypoints = m_keypoints_undist[camid][candid];
            m_detUndist[camid][candid].box = m_boxes_processed[camid][candid]; 
            m_detUndist[camid][candid].mask = m_masksUndist[camid][candid]; 
			m_detUndist[camid][candid].mask_norm = computeContourNormalsAll(m_detUndist[camid][candid].mask);
        }
    }
}

void FrameData::clean_by_mask_chamfer()
{
	for (int camid = 0; camid < m_camNum; camid++)
	{

	}
}

//void FrameData::load_labeled_data()
//{
//	Annotator A;
//	A.result_folder = "E:/my_labels/";
//	A.frameid = m_frameid;
//	A.m_cams = m_camsUndist;
//	A.m_rawCams = m_cams;
//	A.m_camNum = m_camNum;
//	A.read_label_result();
//	A.getMatchedData(m_matched);
//}