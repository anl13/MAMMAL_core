#include "exp1.h" 

void Part1Data::init()
{
	m_imw = 1920;
	m_imh = 1080;
	getColorMap("anliang_paper", m_CM);

	std::string skelType = "UNIV"; 
	m_topo = getSkelTopoByType(skelType); 

	m_camids = { 0,1,2,5,6,7,8,9,10,11 };
	m_camNum = m_camids.size(); 
	readCameras(); 
}

void Part1Data::read_labeling()
{
	/// init empty variable 
	// keypoint: [camid, pigid, jointid]
	m_keypoints_undist.clear(); 
	m_keypoints_undist.resize(m_camNum);
	for (int i = 0; i < m_camNum; i++)
	{
		m_keypoints_undist[i].resize(4);
		for (int j = 0; j < 4; j++) m_keypoints_undist[i][j].resize(m_topo.joint_num, Eigen::Vector3f::Zero()); 
	}
	// mask: [camid, pigid, partid, pointid] 
	m_masks_undist.clear(); 
	m_masks_undist.resize(m_camNum); 
	for (int i = 0; i < m_camNum; i++)
	{
		m_masks_undist[i].resize(4); 
	}

	for (int i = 0; i < m_camNum; i++)
	{
		int camid = m_camids[i]; 
		std::stringstream ss;
		ss << "E:/evaluation_dataset/part1/dataset_process/output_label/" << camid << "/" << std::setw(6) << std::setfill('0') << m_frameid << ".json"; 
		std::string labelpath = ss.str(); 
		Json::Value root; 
		Json::CharReaderBuilder rbuilder; 
		std::string errs; 
		std::ifstream is(labelpath); 
		if (!is.is_open())
		{
			std::cout << "can not open " << labelpath << std::endl; 
			continue; 
		}
		bool parsingSuccessful = Json::parseFromStream(rbuilder, is, &root, &errs); 
		if (!parsingSuccessful)
		{
			std::cout << "parsing " << labelpath << " error!" << std::endl; 
			exit(-1); 
		}
		Json::Value shapes = root["shapes"]; 
		for (int k = 0; k < shapes.size(); k++)
		{
			Json::Value dict = shapes[k]; 
			if (dict["shape_type"] == "point")
			{
				Eigen::Vector3f p;
				p[0] = dict["points"][0][0].asFloat(); 
				p[1] = dict["points"][0][1].asFloat(); 
				p[2] = 1; 
				int label = std::stoi(dict["label"].asString());
				int group = dict["group_id"].asInt(); 
				m_keypoints_undist[i][group][label] = p;
			}
			else if (dict["shape_type"] == "polygon")
			{
				int group = dict["group_id"].asInt(); 
				std::vector<Eigen::Vector2f> points; 
				int N = dict["points"].size(); 
				points.resize(N); 
				for (int m = 0; m < N; m++)
				{
					points[m][0] = dict["points"][m][0].asFloat(); 
					points[m][1] = dict["points"][m][1].asFloat(); 
				}
				m_masks_undist[i][group].push_back(points); 
			}
		}
		is.close(); 
	}
}

void Part1Data::readCameras()
{
	m_cams.clear();
	m_camsUndist.clear();
	std::string camDir = "D:/Projects/animal_calib/data/calibdata/extrinsic/"; 
	for (int camid = 0; camid < m_camNum; camid++)
	{
		std::stringstream ss;
		ss << camDir << std::setw(2) << std::setfill('0') << m_camids[camid] << ".txt";
		std::ifstream camfile;
		camfile.open(ss.str());
		if (!camfile.is_open())
		{
			std::cout << "can not open file " << ss.str() << std::endl;
			exit(-1);
		}
		Eigen::Vector3f rvec, tvec;
		for (int i = 0; i < 3; i++) {
			double a;
			camfile >> a;
			rvec(i) = a;
		}
		for (int i = 0; i < 3; i++)
		{
			double a;
			camfile >> a;
			tvec(i) = a;
		}
		Camera cam;
		Camera camUndist;

		cam = Camera::getDefaultCameraRaw();
		camUndist = Camera::getDefaultCameraUndist();

		cam.SetRT(rvec, tvec);
		camUndist.SetRT(rvec, tvec);
		m_cams.push_back(cam);
		m_camsUndist.push_back(camUndist);
		camfile.close();
	}
}

void Part1Data::compute_3d_gt()
{
	m_gt_keypoints_3d.clear(); 
	m_gt_keypoints_3d.resize(4); 
	for (int i = 0; i < 4; i++)
	{
		m_gt_keypoints_3d[i].resize(m_topo.joint_num, Eigen::Vector3f::Zero()); 
		for (int j = 0; j < m_topo.joint_num; j++)
		{
			std::vector<Camera> current_cams; 
			std::vector<Eigen::Vector3f> current_joints; 
			for (int k = 0; k < m_camNum; k++)
			{
				if (m_keypoints_undist[k][i][j](2) > 0)
				{
					current_joints.push_back(m_keypoints_undist[k][i][j]);
					current_cams.push_back(m_camsUndist[k]);
				}
			}
			if (current_cams.size() < 2) continue; 
			Eigen::Vector3f joint3d = triangulate_ceres(current_cams, current_joints); 
			//Eigen::Vector3f joint3d = NViewDLT(current_cams, current_joints); 
			m_gt_keypoints_3d[i][j] = joint3d;
		}
	}
}

void Part1Data::read_imgs()
{
	m_imgsUndist.resize(m_camNum); 
	for (int i = 0; i < m_camNum; i++)
	{
		int camid = m_camids[i]; 
		std::stringstream ss; 
		ss << "E:/evaluation_dataset/part1/dataset_process/output_label/" << camid << "/"
			<< std::setw(6) << std::setfill('0') << m_frameid << ".jpg"; 
		m_imgsUndist[i] = cv::imread(ss.str()); 
		if (m_imgsUndist.empty())
		{
			std::cout << "what happed to my img" << std::endl;
			exit(-1); 
		}
	}
}

cv::Mat Part1Data::visualizeProj(int pid)
{
	std::vector<cv::Mat> imgdata;
	cloneImgs(m_imgsUndist, imgdata);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int id = 0; id < m_projs[camid].size(); id++)
		{
			if (pid >= 0 && id != pid) continue;
			Eigen::Vector3i color;
			int colorid = id;
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

cv::Mat Part1Data::visualize2D(int pid)
{
	std::vector<cv::Mat> imgdata; 
	cloneImgs(m_imgsUndist, imgdata); 
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int id = 0; id < 4; id++)
		{
			if (pid >= 0 && id != pid) continue; 
			Eigen::Vector3i color; 
			int colorid = id; 
			color(0) = m_CM[colorid](2); 
			color(1) = m_CM[colorid](1);
			color(2) = m_CM[colorid](0);
			drawSkelMonoColor(imgdata[camid], m_keypoints_undist[camid][id], color, m_topo);
		}
	}
	cv::Mat packed; 
	packImgBlock(imgdata, packed); 
	return packed; 
}

void Part1Data::reproject_skels()
{
	m_projs.clear();
	m_projs.resize(m_camNum);
	for (int c = 0; c < m_camNum; c++) m_projs[c].resize(4);

	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int id = 0; id < 4; id++)
		{
			m_projs[camid][id].resize(m_topo.joint_num, Eigen::Vector3f::Zero());
			for (int kpt_id = 0; kpt_id < m_topo.joint_num; kpt_id++)
			{
				if (m_gt_keypoints_3d[id][kpt_id].norm() == 0) continue;
				Eigen::Vector3f p = m_gt_keypoints_3d[id][kpt_id];
				m_projs[camid][id][kpt_id] = project(m_camsUndist[camid], p);
			}
		}
	}
}

// other functions 
vector<vector<Eigen::Vector3f>> load_skel(std::string folder, int frameid)
{
	vector<vector<Eigen::Vector3f> > skels; 
	skels.resize(4);

	for (int pid = 0; pid < 4; pid++)
	{
		skels[pid].resize(23); 
		std::stringstream ss;
		ss << folder << "/pig_" << pid << "_" << std::setw(6) << std::setfill('0') << frameid << ".txt"; 
		std::ifstream is(ss.str()); 
		if (!is.is_open())
		{
			std::cout << "in load_skel, " << folder << ", " << frameid << ", can not open" << std::endl; 
			exit(-1); 
		}
		for (int i = 0; i < 23; i++)
		{
			for (int k = 0; k < 3; k++)
			{
				is >> skels[pid][i](k); 
			}
		}
		is.close(); 
	}
	return skels; 
}

vector<vector<Eigen::Vector3f>> load_joint23(std::string folder, int frameid)
{
	vector<vector<Eigen::Vector3f> > skels;
	skels.resize(4);

	for (int pid = 0; pid < 4; pid++)
	{
		skels[pid].resize(23);
		std::stringstream ss;
		ss << folder << "/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
		std::ifstream is(ss.str());
		if (!is.is_open())
		{
			std::cout << "in load_skel, " << folder << ", " << frameid << ", can not open" << std::endl;
			exit(-1);
		}
		for (int i = 0; i < 23; i++)
		{
			for (int k = 0; k < 3; k++)
			{
				is >> skels[pid][i](k);
			}
		}
		is.close();
	}
	return skels;
}

void process_generate_label3d()
{
	Part1Data loader; 

	loader.init(); 

	std::string folder = "E:/evaluation_dataset/part1/dataset_process/label3d/"; 
	for (int i = 0; i < 70; i++)
	{
		int frameid = 750 + 25 * i; 
		loader.set_frame_id(frameid); 
		loader.read_labeling(); 
		loader.compute_3d_gt(); 

		for (int pid = 0; pid < 4; pid++)
		{
			std::stringstream ss;
			ss << folder << "pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
			std::ofstream ostream(ss.str());
			if (!ostream.is_open())
			{
				std::cout << "what the hell. " << std::endl;
				exit(-1);
			}

			for (int k = 0; k < 23; k++)
			{
				ostream << loader.m_gt_keypoints_3d[pid][k].transpose() << std::endl; 
			}
			ostream.close(); 
		}
	}
}