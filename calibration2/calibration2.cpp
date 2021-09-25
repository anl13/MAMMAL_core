#include "calibration2.h"
#include <json/json.h> 

Calibrator2::Calibrator2()
{
	m_camids = { 1,2,5,6,7,8,9 }; 
	m_camNum = m_camids.size();
	m_folder = "D:/Projects/animal_calibration/calib_batch3/result_batch3/undist3/"; 

	read_results_rt("D:/Projects/animal_calibration/calib_batch3/result_batch3/");

	m_K = m_camsUndist[0].K; 
	
	getColorMap("anliang_rgb", m_CM); 
	m_draw_size = 100;

}

int Calibrator2::calib_pipeline()
{
	readAllMarkers(m_folder);
	readImgs();
	unprojectMarkers();
	readInit3DPoints(); 
	
	// fill all 3d points using DLT 
	int point3dnum = 0; /*out_points.size(); */
	int point2dnum = m_markers[0].size();
	if (point2dnum > point3dnum)
	{
		for (int i = point3dnum; i < point2dnum; i++)
		{
			std::vector<Camera> cams; 
			std::vector<Eigen::Vector3f> p; 
			for (int k = 0; k < m_camNum; k++)
			{
				if (m_markers[k][i](2) < 1) continue; 
				cams.push_back(m_camsUndist[k]); 
				p.push_back(m_markers[k][i].segment<3>(0)); 
			}
			if (cams.size() >= 2)
			{
				//Eigen::Vector3f q = NViewDLT(cams, p);
				Eigen::Vector3f q = triangulate_ceres(cams, p); 
				out_points.push_back(q);
			}
			else
				out_points.push_back(Eigen::Vector3f::Zero()); 
		}
	}

	for (int i = 0; i < out_points.size(); i++)
	{
		std::cout << out_points[i].transpose() << std::endl; 
	}

	// run ba 
	ba.setCamIds(m_camids); 
	ba.readInit("D:/Projects/animal_calibration/calib_batch3/result_batch3/init_result/"); 
	ba.setInit3DPoints(out_points); 
	ba.setObs(m_i_markers); 
	ba.solve_again(); 

	out_rvecs = ba.getRvecsF(); 
	out_tvecs = ba.getTvecsF(); 
	out_points = ba.getPointsF(); 

	save_results("D:/Projects/animal_calibration/calib_batch3/result_batch3/refined_result/"); 

}

void Calibrator2::readAllMarkers(std::string folder)
{
	m_markers.resize(m_camids.size()); 
	for (int i = 0; i < m_camids.size(); i++)
	{
		std::stringstream ss;
		//ss << folder << "/bg" << m_camids[i] << "_undist.json";
		ss << folder << "/points_" << m_camids[i] << ".txt"; 
		m_markers[i] = readMarkersTxt(ss.str()); 
	}
}

vector<Eigen::Vector3f> Calibrator2::readMarkers(std::string filename)
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

	vector<Eigen::Vector3f> points;
	points.resize(100, Eigen::Vector3f::Zero()); 
	auto data = root["shapes"]; 
	for (auto const &c : data)
	{
		std::string type = c["shape_type"].asString(); 
		if (type == "point")
		{
			float x = c["points"][0][0].asFloat();
			float y = c["points"][0][1].asFloat();
			std::string s_index = c["label"].asString();
			int i_index = std::stoi(s_index); 
			points[i_index](0) = x;
			points[i_index](1) = y;
			points[i_index](2) = 1;
		}
	}
	instream.close(); 
	return points; 
}

vector<Eigen::Vector3f> Calibrator2::readMarkersTxt(std::string filename)
{
	vector<Eigen::Vector3f> points; 
	points.resize(100, Eigen::Vector3f::Zero()); 
	std::ifstream stream(filename); 
	if (!stream.is_open())
	{
		std::cout << "can not open " << filename << std::endl;
		exit(-1);
	}
	while(!stream.eof())
	{
		float x, y, z; 
		stream >> x;
		if (stream.eof()) break; 
		stream >> y >> z;
		int i_index = int(z); 
		points[i_index](0) = x; 
		points[i_index](1) = y; 
		points[i_index](2) = 1; 
	}
	stream.close();
	return points; 
}

void Calibrator2::readImgs()
{
	m_imgsUndist.resize(m_camids.size()); 
	for (int i = 0; i < m_camids.size(); i++)
	{
		std::stringstream ss;
		ss << "D:/Projects/animal_calibration/calib_batch3/result_batch3/undist/bg"
			<< m_camids[i] << "_undist.png"; 
		cv::Mat img = cv::imread(ss.str()); 
		m_imgsUndist[i] = img; 
	}
}

void Calibrator2::unprojectMarkers()
{
	int camNum = m_camids.size();
	// init 
	m_i_markers.resize(camNum);
	// compute
	Eigen::Matrix3f invK = m_K.inverse();
	for (int camid = 0; camid < camNum; camid++)
	{
		int pNum = m_markers[camid].size();
		m_i_markers[camid].resize(pNum, Eigen::Vector3f::Zero());
		for (int i = 0; i < pNum; i++)
		{
			if (m_markers[camid][i].norm() == 0) continue; 
			Eigen::Vector3f ph;
			ph = m_markers[camid][i];
			ph(2) = 1;
			Eigen::Vector3f pImagePlane = invK * ph;
			m_i_markers[camid][i] = pImagePlane;
		}
	}
}

void Calibrator2::readInit3DPoints()
{
	std::string filename = "D:/Projects/animal_calibration/calib_batch3/result_batch3/init_result/points3d_flipx.txt";
	std::ifstream stream(filename); 
	out_points.clear(); 
	while (!stream.eof())
	{
		float x, y, z; 
		stream >> x; 
		if (stream.eof()) break; 
		stream >> y >> z; 
		out_points.push_back(Eigen::Vector3f(x, y, z)); 
	}
	stream.close();
}


void Calibrator2::save_results(std::string result_folder)
{
	if (!boost::filesystem::exists(result_folder))
	{
		boost::filesystem::create_directories(result_folder);
	}

	// save r and t
	for (int i = 0; i < m_camids.size(); i++)
	{
		std::stringstream ss;
		ss << result_folder << "/" << std::setw(2) << std::setfill('0') << m_camids[i] << ".txt";
		std::ofstream is;
		is.open(ss.str());
		if (!is.is_open())
		{
			std::cout << "error openning " << ss.str() << std::endl;
			exit(-1);
		}
		for (int j = 0; j < 3; j++)
		{
			is << out_rvecs[i][j] << "\n";
		}
		for (int j = 0; j < 3; j++)
		{
			is << out_tvecs[i][j] << "\n";
		}
		is.close();
	}

	// save points 
	std::stringstream ss;
	ss << result_folder << "/points3d.txt";
	std::ofstream os;
	os.open(ss.str());
	if (!os.is_open())
	{
		std::cout << "can not open " << ss.str() << std::endl;
		exit(-1);
	}
	for (int i = 0; i < out_points.size(); i++)
	{
		os << out_points[i].transpose() << "\n";
	}
	std::cout << "save out_points: " << out_points.size() << std::endl;
	os.close();
}


void Calibrator2::evaluate()
{
	// project initial markers 
	vector<vector<Eigen::Vector3f> > projs;
	projs.resize(m_camNum);
	std::cout << out_points.size() << std::endl;
	for (int v = 0; v < m_camNum; v++)
	{
		vector<Eigen::Vector3f> proj;
		project(m_camsUndist[v], out_points, proj);
		projs[v] = proj;
	}

	m_projs_markers = projs;

	// compute errors 
	double total_errs = 0;
	int num = 0;
	for (int v = 0; v < m_camNum; v++)
	{
		for (int i = 0; i < m_markers[v].size(); i++)
		{
			Eigen::Vector3f gt = m_markers[v][i];
			Eigen::Vector3f projection = projs[v][i];
			// std::cout << gt.transpose() << " ......  " << projection.transpose() << std::endl; 
			Eigen::Vector2f err = gt.segment<2>(0) - projection.segment<2>(0);
			total_errs += err.norm();
			num += 1;
		}
	}
	std::cout << "avg err: " << total_errs / num << std::endl;
}

void Calibrator2::draw_points()
{
	cloneImgs(m_imgsUndist, m_imgsDraw);
	for (int v = 0; v < m_camNum; v++)
	{
		std::vector<Eigen::Vector3f> points;
		for (int i = 0; i < m_markers[v].size(); i++)
		{
			Eigen::Vector3f p;
			p = m_markers[v][i];
			p(2) = 1;
			points.push_back(p);
		}
		my_draw_points(m_imgsDraw[v], points, m_CM[1], 10);
		my_draw_points(m_imgsDraw[v], m_projs_markers[v], m_CM[2], 8);
	}

	// cv::Mat output;
	// packImgBlock(m_imgsUndist, output); 
	// cv::namedWindow("raw", cv::WINDOW_NORMAL); 
	// cv::imshow("raw", output); 
	// int key = cv::waitKey(); 
}

void Calibrator2::read_results_rt(std::string result_folder)
{
	m_camsUndist.resize(m_camNum); 
	for (int i = 0; i < m_camids.size(); i++)
	{
		std::stringstream ss;
		ss << result_folder << "/" << std::setw(2) << std::setfill('0') << m_camids[i] << ".txt";
		std::ifstream is;
		is.open(ss.str());
		if (!is.is_open())
		{
			std::cout << "error openning " << ss.str() << std::endl;
			exit(-1);
		}
		Eigen::Vector3f r_vec;
		Eigen::Vector3f t_vec;
		for (int j = 0; j < 3; j++)
		{
			is >> r_vec(j);
		}
		for (int j = 0; j < 3; j++)
		{
			is >> t_vec(j);
		}
		m_camsUndist[i].SetRT(r_vec, t_vec);
		is.close();
	}
}