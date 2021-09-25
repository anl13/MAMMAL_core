#include "calibration2.h"
#include <json/json.h> 

Calibrator2::Calibrator2()
{
	m_camids = { 1,2,5,6,7,8,9 }; 
	m_folder = "D:/Projects/animal_calibration/calib_batch3/result_batch3/undist/"; 

	readInitResult(); 

	m_K = m_camsUndist[0].K; 
	m_camNum = m_camids.size(); 
	getColorMap("anliang_rgb", m_CM); 
	m_draw_size = 100;

}

int Calibrator2::calib_pipeline()
{
	readAllMarkers(m_folder);
	readImgs();
	unprojectMarkers();
	
	int point3dnum = out_points.size(); 
	int point2dnum = m_i_markers[0].size();
	if (point2dnum > point3dnum)
	{
		for (int i = point3dnum; i < point2dnum; i++)
		{
			
		}
	}
}

void Calibrator2::readAllMarkers(std::string folder)
{
	m_markers.resize(m_camids.size()); 
	for (int i = 0; i < m_camids.size(); i++)
	{
		std::stringstream ss;
		ss << folder << "/bg" << m_camids[i] << "_undist.json";
		m_markers[i] = readMarkers(ss.str()); 
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
			float x = c["points"][0].asFloat();
			float y = c["points"][1].asFloat();
			int index = c["label"].asInt();
			points[index](0) = x;
			points[index](1) = y;
			points[index](2) = 1;
		}
	}
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

void Calibrator2::readInitResult()
{
	m_camsUndist.resize(m_camids.size()); 
	for (int i = 0; i < m_camids.size(); i++)
	{
		m_camsUndist[i] = Camera::getFarCameraUndist(); 
		std::stringstream ss; 
		ss << "D:/Projects/animal_calibration/calib_batch3/result_batch3/"
			<< std::setw(2) << std::setfill('0') << m_camids[i] << ".txt"; 
		std::ifstream stream(ss.str()); 
		Eigen::Vector3f rvec, tvec; 
		for (int k = 0; k < 3; k++)
		{
			stream >> rvec(k); 
		}
		for (int k = 0; k < 3; k++)
		{
			stream >> tvec(k); 
		}
		m_camsUndist[i].SetRT(rvec, tvec); 
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

