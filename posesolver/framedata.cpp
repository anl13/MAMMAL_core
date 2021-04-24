#include "framedata.h" 
#include <json/json.h> 
#include <math.h> 
#include <algorithm>
#include <json/writer.h> 

void FrameData::configByJson(std::string filename)
{
	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(filename);
	if (!instream.is_open())
	{
		std::cout << "can not open " << filename << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}
	m_sequence = root["sequence"].asString();
	m_keypointsDir = m_sequence + "/"+ root["keypointsdir"].asString() + "/";
	m_imgDir = m_sequence + "/"+ root["imgdir"].asString() +"/";
	m_boxDir = m_sequence + "/"+ root["boxdir"].asString() + "/";
	m_maskDir = m_sequence + "/" + root["maskdir"].asString() + "/";
	m_camDir = root["camfolder"].asString();
	m_imgExtension = root["imgExtension"].asString();
	m_pignum = root["pignum"].asInt(); 

	m_boxExpandRatio = root["box_expand_ratio"].asDouble();
	m_skelType = root["skel_type"].asString();
	m_topo = getSkelTopoByType(m_skelType);
	m_is_read_image = root["is_read_image"].asBool();
	m_videotype = root["videotype"].asInt(); 
	std::vector<int> camids;
	for (auto const &c : root["camids"])
	{
		int id = c.asInt();
		camids.push_back(id);
	}
	m_camids = camids;
	m_camNum = m_camids.size(); 

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
				std::setw(6) << std::setfill('0') << ".mp4"; 
			
			m_caps[camid] = cv::VideoCapture(name.str()); 

			if (!m_caps[camid].isOpened())
			{
				std::cout << "cannot open video " << name.str() << std::endl; 
				system("pause"); 
				exit(-1); 
			}
		}
	}
	else if(m_videotype==2)
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

	instream.close();

	m_video_frameid = 0;
	initRectifyMap(); 
}

void FrameData::set_frame_id(int _frameid)
{
	m_frameid = _frameid;
	if (m_videotype > 0)
	{
		if (m_frameid == m_video_frameid) return;
		for (int i = 0; i < m_camNum; i++)
		{
			m_caps[i].set(cv::CAP_PROP_POS_FRAMES, m_frameid); 
		}
	}
}

void FrameData::fetchData()
{
    if(m_frameid < 0)
    {
        std::cout << "Error: wrong frame id " << std::endl;
        exit(-1); 
    }

	if (m_is_read_image)
	{
		if (m_videotype > 0)
		{
			readImagesFromVideo(); 
			undistImgs(); 
		}
		else {
			//readImages(); 
			//undistImgs(); 
			readUndistImages();
		}
	}

    readKeypoints(); 
    undistKeypoints(); 

    readBoxes();
    processBoxes(); 

    readMask(); 
    undistMask(); 

    assembleDets(); 
}

void FrameData::readKeypoints() // load hrnet keypoints
{
    std::string jsonDir = m_keypointsDir;
    std::stringstream ss; 
	if (m_videotype == 1)
	{
		ss << jsonDir << "/hour" << m_hourid << "/" << std::setw(6) << std::setfill('0') << m_frameid << ".json";
	}
	else if (m_videotype == 0 || m_videotype == 2)
	{
		ss << jsonDir << std::setw(6) << std::setfill('0') << m_frameid << ".json";
	}
	else {
		std::cout << "read keypoints: m_videotype " << m_videotype << " not implement." << std::endl; 
		exit(-1); 
	}
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
            if(candid >=m_pignum) break; 
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
				//if (v(2) > 0)
					pig[pid] = v; 
				else pig[pid] = Eigen::Vector3f::Zero(); 
				if (v(0) < 10 || v(0) > 1910 || v(1) < 10 || v(1) > 1070)
					pig[pid] = Eigen::Vector3f::Zero(); 
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
	if (m_videotype == 1)
	{
		ss << jsonDir << "/hour" << m_hourid << "/" << std::setw(6) << std::setfill('0') << m_frameid << ".json";
	}
	else if (m_videotype == 0 || m_videotype == 2)
	{
		ss << jsonDir << std::setw(6) << std::setfill('0') << m_frameid << ".json";
	}
	else {
		std::cout << "read boxes: m_videotype " << m_videotype << " not implement." << std::endl;
		exit(-1);
	}
	
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
            if(bid >= m_pignum) break; // remain only 4 top boxes 
            Json::Value box_jv = c[bid]; 
            Eigen::Vector4f B; 
            for(int k = 0; k < 4; k++)
            {
                double x = box_jv[k].asDouble(); 
                B(k) = x; 
            }
            bb.push_back(B); 
			//std::cout << "(" << i << "," << bid << "):" << B << std::endl;
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

void FrameData::initRectifyMap()
{
	auto camera = m_cams[0];
	auto newcam = m_camsUndist[0];
	cv::Vec<float, 5> distCoef;
	distCoef[0] = camera.k(0);
	distCoef[1] = camera.k(1);
	distCoef[2] = camera.p(0);
	distCoef[3] = camera.p(1);
	distCoef[4] = camera.k(2);

	cv::Mat K(3, 3, CV_32FC1, 0.0f);
	cv::eigen2cv(camera.K, K);
	cv::Mat K2(3, 3, CV_32FC1, 0.0f);
	cv::eigen2cv(newcam.K, K2);

	cv::Mat R;

	cv::initUndistortRectifyMap(
		K, distCoef, R, K2, cv::Size(1920,1080), CV_32FC1, m_map1, m_map2
	);
}

void FrameData::undistImgs()
{
    m_imgsUndist.resize(m_camNum); 
    for(int i = 0; i < m_camNum; i++)
    {
		cv::remap(m_imgs[i], m_imgsUndist[i], m_map1, m_map2, cv::INTER_LINEAR);
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

void FrameData::readImagesFromVideo()
{
	m_imgs.clear(); 
	m_imgs.resize(m_camNum); 
	for (int camid = 0; camid < m_camNum; camid++)
	{
		m_caps[camid].read(m_imgs[camid]);
		if (m_imgs[camid].empty())
		{
			std::cout << "can not fetch image " << std::endl;
			system("pause"); 
			exit(-1);
		}
	}
	m_video_frameid++;
}

void FrameData::readUndistImages()
{
	m_imgs.clear();
	m_imgsUndist.clear(); 
	std::string undistImgDir = m_sequence + "/undistort_images/";
	for (int camid = 0; camid < m_camNum; camid++)
	{
		std::stringstream ss;
		ss << undistImgDir << "/cam" << m_camids[camid] << "/" << std::setw(6) << std::setfill('0') << m_frameid << "." << m_imgExtension;
		cv::Mat img = cv::imread(ss.str());
		if (img.empty())
		{
			std::cout << "can not read image " << ss.str() << std::endl;
			exit(-1);
		}
		m_imgsUndist.emplace_back(img);
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


void FrameData::readMask()
{
    std::string jsonDir = m_maskDir;
    std::stringstream ss; 
	if (m_videotype == 1)
	{
		ss << jsonDir << "/hour" << m_hourid << "/" << std::setw(6) << std::setfill('0') << m_frameid << ".json";
	}
	else if (m_videotype == 0 || m_videotype == 2)
	{
		ss << jsonDir << std::setw(6) << std::setfill('0') << m_frameid << ".json";
	}
	else {
		std::cout << "read masks: m_videotype " << m_videotype << " not implement." << std::endl;
		exit(-1);
	}
	
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
        int candnum = m_boxes_processed[camid].size(); 
        m_detUndist[camid].resize(candnum); 
        for(int candid = 0; candid < candnum; candid++)
        {
			m_detUndist[camid][candid].valid = true;
            m_detUndist[camid][candid].keypoints = m_keypoints_undist[camid][candid];
			if (m_detUndist[camid][candid].keypoints.size() == 0)
				m_detUndist[camid][candid].keypoints.resize(m_topo.joint_num, Eigen::Vector3f::Zero()); 
            m_detUndist[camid][candid].box = m_boxes_processed[camid][candid]; 

            m_detUndist[camid][candid].mask = m_masksUndist[camid][candid]; 
			//m_detUndist[camid][candid].mask_norm = computeContourNormalsAll(m_detUndist[camid][candid].mask);
			for (int i = 0; i < m_detUndist[camid][candid].keypoints.size(); i++)
			{
				if (m_detUndist[camid][candid].keypoints[i](2) == 0) continue; 
				Eigen::Vector2f keypoint = m_detUndist[camid][candid].keypoints[i].segment<2>(0);
				if (!in_box_test(keypoint, m_detUndist[camid][candid].box))
					m_detUndist[camid][candid].keypoints[i] = Eigen::Vector3f::Zero(); 
			}
		}
    }
}


cv::Mat FrameData::visualizeSkels2D()
{
	vector<cv::Mat> imgdata;
	cloneImgs(m_imgsUndist, imgdata);
	for (int i = 0; i < m_camNum; i++)
	{
		for (int k = 0; k < m_detUndist[i].size(); k++)
		{
			drawSkelMonoColor(imgdata[i], m_detUndist[i][k].keypoints, k, m_topo);
			Eigen::Vector3i color = m_CM[k];
			Eigen::Vector3i c_bgr; 
			c_bgr(0) = color(2); 
			c_bgr(1) = color(1);
			c_bgr(2) = color(0); 
			my_draw_box(imgdata[i], m_detUndist[i][k].box, c_bgr);
			my_draw_mask(imgdata[i], m_detUndist[i][k].mask, c_bgr, 0.5);
		}
	}
	cv::Mat output;
	packImgBlock(imgdata, output);

	return output;
}
