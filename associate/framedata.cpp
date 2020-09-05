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
            drawSkel(imgdata[i], m_detUndist[i][k].keypoints, k); 
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
            int camid = m_matched[id].view_ids[i];
            //int candid = m_matched[id].cand_ids[i];
            //if(candid < 0) continue; 
			if(m_matched[id].dets[i].keypoints.size() > 0)
				drawSkel(m_imgsDetect[camid], m_matched[id].dets[i].keypoints, id);
            my_draw_box(m_imgsDetect[camid], m_matched[id].dets[i].box, m_CM[id]);

			if (m_matched[id].dets[i].mask.size() > 0)
            my_draw_mask(m_imgsDetect[camid], m_matched[id].dets[i].mask, m_CM[id], 0.5);
        }
    }
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int i = 0; i < m_unmatched[camid].size(); i++)
		{
			if(m_unmatched[camid][i].keypoints.size()>0)
			drawSkel(m_imgsDetect[camid], m_unmatched[camid][i].keypoints, 5);
			my_draw_box(m_imgsDetect[camid], m_unmatched[camid][i].box, m_CM[5]);
			if (m_unmatched[camid][i].mask.size()>0)
			my_draw_mask(m_imgsDetect[camid], m_unmatched[camid][i].mask, m_CM[5], 0.5);
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
    
    for(int camid = 0; camid < m_camNum; camid++)
    {
        for(int id = 0; id < m_projs[camid].size(); id++)
        {
            drawSkel(imgdata[camid], m_projs[camid][id], id);
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

cv::Mat FrameData::test()
{
    cv::Mat output = visualizeSkels2D(); 

    return output; 
}

void FrameData::drawSkel(cv::Mat& img, const vector<Eigen::Vector3f>& _skel2d, int colorid)
{
    Eigen::Vector3i color = m_CM[colorid];
    cv::Scalar cv_color(color(0), color(1), color(2)); 
    for(int i = 0; i < _skel2d.size(); i++)
    {
        cv::Point2d p(_skel2d[i](0), _skel2d[i](1)); 
        double conf = _skel2d[i](2); 
        if(conf < m_topo.kpt_conf_thresh[i]) continue; 
        cv::circle(img, p, 8, cv_color, -1); 
    }
    for(int k = 0; k < m_topo.bone_num; k++)
    {
        Eigen::Vector2i b = m_topo.bones[k]; 
        Eigen::Vector3f p1 = _skel2d[b(0)];
        Eigen::Vector3f p2 = _skel2d[b(1)]; 
        if(p1(2) < m_topo.kpt_conf_thresh[b(0)] || p2(2) < m_topo.kpt_conf_thresh[b(1)]) continue; 
        cv::Point2d p1_cv(p1(0), p1(1)); 
        cv::Point2d p2_cv(p2(0), p2(1)); 
        cv::line(img, p1_cv, p2_cv, cv_color, 4); 
    }
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

void FrameData::drawSkelDebug(cv::Mat& img, const vector<Eigen::Vector3f>& _skel2d)
{
	for (int i = 0; i < _skel2d.size(); i++)
	{
		int colorid = m_topo.kpt_color_ids[i];
		Eigen::Vector3i color = m_CM[colorid];
		cv::Scalar cv_color(color(0), color(1), color(2));

		cv::Point2d p(_skel2d[i](0), _skel2d[i](1));
		double conf = _skel2d[i](2);
		if (conf < m_topo.kpt_conf_thresh[i]) continue;
		cv::circle(img, p, int(12*conf), cv_color, -1);
	}
	for (int k = 0; k < m_topo.bone_num; k++)
	{
		int jid = m_topo.bones[k](0);
		int colorid = m_topo.kpt_color_ids[jid];
		Eigen::Vector3i color = m_CM[colorid];
		cv::Scalar cv_color(color(0), color(1), color(2));

		Eigen::Vector2i b = m_topo.bones[k];
		Eigen::Vector3f p1 = _skel2d[b(0)];
		Eigen::Vector3f p2 = _skel2d[b(1)];
		if (p1(2) < m_topo.kpt_conf_thresh[b(0)] || p2(2) < m_topo.kpt_conf_thresh[b(1)]) continue;
		cv::Point2d p1_cv(p1(0), p1(1));
		cv::Point2d p2_cv(p2(0), p2(1));
		cv::line(img, p1_cv, p2_cv, cv_color, 4);
	}
}

void FrameData::view_dependent_clean()
{
	std::vector<int> top_views = { 3,5,6 };
	
	for (int camid = 0; camid < m_camNum; camid++)
	{
		if (in_list(camid, top_views))
		{
			// for top views, we use center-tail to clean flipped legs 
			for (int i = 0; i < m_detUndist[camid].size(); i++)
			{
				//cv::Mat temp = m_imgsUndist[3]; // for debug only
				//drawSkelDebug(temp, m_detUndist[camid][i].keypoints);
				//cv::namedWindow("test", cv::WINDOW_NORMAL); 
				//cv::imshow("test", temp);
				//cv::waitKey(); 
				top_view_clean(m_detUndist[camid][i]);
			}
		}
		else
		{
			// for side views, we do visibility checking by number of leg points 
			for (int i = 0; i < m_detUndist[camid].size(); i++)
			{
				//if (camid == 8) // for debug
				//{
				//	cv::Mat temp = m_imgsUndist[8];
				//	drawSkelDebug(temp, m_detUndist[camid][i].keypoints);
				//	cv::namedWindow("test", cv::WINDOW_NORMAL);
				//	cv::imshow("test", temp);
				//	cv::waitKey();
				//}
				side_view_clean(m_detUndist[camid][i]); 
				//if (camid == 8 && i==2) // for debug
				//{
				//	cv::Mat temp = m_imgsUndist[8];
				//	drawSkelDebug(temp, m_detUndist[camid][i].keypoints);
				//	cv::namedWindow("test", cv::WINDOW_NORMAL);
				//	cv::imshow("test", temp);
				//	cv::waitKey();
				//}
			}
		}
	}
}

// Attention: z points towards ground 
// and note that, on image coordinate, to_left_test result should be reverse. 
void FrameData::to_left_clean(DetInstance& det)
{
	std::vector<int> left_front_leg = { 5,7,9 };
	std::vector<int> right_front_leg = { 6,8,10 };
	std::vector<int> left_back_leg = { 11,13,15 };
	std::vector<int> right_back_leg = { 12,14,16 };
	std::vector<int> left = { 5,7,9,11,13,15 };
	std::vector<int> right = { 6,8,10,12,14,16 };
	Eigen::Vector3f center = det.keypoints[20];
	Eigen::Vector3f tail = det.keypoints[18];
	for (int i = 0; i < 6; i++)
	{
		int kid = left[i];
		auto kpt = det.keypoints[kid];
		if (kpt(2) < m_topo.kpt_conf_thresh[kid])
		{
			det.keypoints[kid] = Eigen::Vector3f::Zero();
			continue;
		}
		else
		{
			bool is_left = to_left_test(tail, center, kpt);
			if (is_left)det.keypoints[kid] = Eigen::Vector3f::Zero();
		}
	}
	for (int i = 0; i < 6; i++)
	{
		int kid = right[i];
		auto kpt = det.keypoints[kid];
		if (kpt(2) < m_topo.kpt_conf_thresh[kid])
		{
			det.keypoints[kid] = Eigen::Vector3f::Zero();
			continue;
		}
		else
		{
			bool is_left = to_left_test(tail, center, kpt);
			if (!is_left)det.keypoints[kid] = Eigen::Vector3f::Zero();
		}
	}
}
void FrameData::top_view_clean(DetInstance& det)
{
	Eigen::Vector3f center = det.keypoints[20];
	Eigen::Vector3f tail = det.keypoints[18];
	if (tail(2) < m_topo.kpt_conf_thresh[18] || tail(2) < m_topo.kpt_conf_thresh[20]) return;
	
	to_left_clean(det); 
}

void FrameData::side_view_clean(DetInstance& det)
{
	std::vector<int> left_front_leg = { 5,7,9 };
	std::vector<int> right_front_leg = { 6,8,10 };
	std::vector<int> left_back_leg = { 11,13,15 };
	std::vector<int> right_back_leg = { 12,14,16 };
	std::vector<int> left = { 5,7,9,11,13,15 };
	std::vector<int> right = { 6,8,10,12,14,16 };
	std::vector<int> front = { 5,7,9,6,8,10 };
	std::vector<int> back = { 11,13,15,12,14,16 };
	Eigen::Vector3f center = det.keypoints[20];
	Eigen::Vector3f tail = det.keypoints[18];
	if (tail(2) < m_topo.kpt_conf_thresh[18] || tail(2) < m_topo.kpt_conf_thresh[20]) return;
	Eigen::Vector2f vec = (center - tail).segment<2>(0);
	float angle = vec2angle(vec);
	// if tail-center is nearly perpendicular to x-axis, use to_left_test
	float angle_range = 30;
	if (angle > 90-angle_range && angle < 90+angle_range)
	{
		to_left_clean(det);
	}
	if (angle > -90-angle_range && angle < -90+angle_range)
	{
		to_left_clean(det);
	}
	// else, use leg count to do visibility checking
	// a leg is reserved if and only if its 3 points are all visible
	// and we assume not all legs are visible, only at most 3 are visible.
	// 0(1)\    /1(2)
	//      \  /
	//       C
	//      /  \
	// 2(4)/    \3(8)
	// states: invisible parts
	// 12: front
	// 3:  back 
	// 5:  right 
	// 10: left
	// 13: right front 
	// 14: left front
	// 7:  right back 
	// 11: left back 
	// others: leave un-handled
	std::vector<int> vis_count(4, 0);
	for (int i = 0; i < 3; i++)
	{
		int kid = left_front_leg[i];
		if (det.keypoints[kid](2) > m_topo.kpt_conf_thresh[kid]) vis_count[0]++;
		kid = right_front_leg[i];
		if (det.keypoints[kid](2) > m_topo.kpt_conf_thresh[kid]) vis_count[1]++;
		kid = left_back_leg[i];
		if (det.keypoints[kid](2) > m_topo.kpt_conf_thresh[kid]) vis_count[2]++;
		kid = right_back_leg[i];
		if (det.keypoints[kid](2) > m_topo.kpt_conf_thresh[kid]) vis_count[3]++;
	}
	int state = 0;
	for (int i = 0; i < 4; i++)
	{
		if (vis_count[i] == 3)state += (1 << i);
	}
	std::vector<int> invisible_list; 
	switch (state)
	{
	case 12: invisible_list = front; break; 
	case 3: invisible_list = back; break; 
	case 5: invisible_list = right; break;
	case 10:invisible_list = left; break; 
	case 13: invisible_list = right_front_leg; break; 
	case 14:invisible_list = left_front_leg; break; 
	case 7: invisible_list = right_back_leg; break; 
	case 11: invisible_list = left_back_leg; break; 
	default:break; 
	}
	for (int i = 0; i < invisible_list.size(); i++)
	{
		int kid = invisible_list[i];
		det.keypoints[kid] = Eigen::Vector3f::Zero(); 
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