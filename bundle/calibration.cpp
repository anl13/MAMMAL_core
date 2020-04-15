#include "calibration.h"

using std::cin; 
using std::cout; 
using std::endl; 

Calibrator::Calibrator(std::string _folder)
{
	folder = _folder;
	m_camids = {0,1,2,5,6,7,8,9,10,11}; 
	m_camNum = m_camids.size(); 
    for(int i = 0; i < m_camNum; i++)
    {
        auto raw_cam = getDefaultCameraRaw();
        m_cams.push_back(raw_cam); 
        auto undist_cam = getDefaultCameraUndist(); 
        m_camsUndist.push_back(undist_cam); 
    }
    readImgs(); 
    getColorMap("anliang_rgb", m_CM); 
}
 
vector<Eigen::Vector2d> Calibrator::readMarkers(std::string filename)
{
	vector<Eigen::Vector2d> points; 
	std::ifstream is(filename); 
	if (!is.is_open())
	{
		cout << "can not open file " << filename << endl;
		return points; 
	}
	while (!is.eof())
	{
		double x, y; 
		is >> x; 
		if(is.eof()) break; 
		is >> y; 
		points.push_back(Eigen::Vector2d(x, y)); 
	}
	is.close(); 

	return points; 
}

void Calibrator::readAllMarkers(std::string folder)
{
	vector<vector<Eigen::Vector2d> > allPoints; 
	for (int i = 0; i < m_camids.size(); i++)
	{
		std::stringstream ss; 
		ss << folder << "/" << std::setw(2) << std::setfill('0') << m_camids[i] << ".txt"; 
		auto points = readMarkers(ss.str()); 
		std::cout << "points num: " << points.size() << std::endl; 
		allPoints.push_back(points); 
	}
	m_markers = allPoints; 
	return; 
}

void Calibrator::readK(std::string filename)
{
	Eigen::Matrix3d K; 
	std::ifstream is;
	is.open(filename);
	if (!is.is_open())
	{
		cout << "can not open file " << filename << endl; 
		exit(-1); 
	}
	for (int i = 0; i < 9; i++)
	{
		double v;
		is >> v; 
		int r = i / 3; 
		int c = i % 3; 
		K(r, c) = v; 
	}
	is.close();
    m_K = K;  
	return; 
}

void Calibrator::unprojectMarkers()
{
    int camNum = m_camids.size(); 
    // init 
    m_i_markers.resize(camNum); 
    // compute
	Eigen::Matrix3d invK = m_K.inverse(); 
	for (int camid = 0; camid < camNum; camid++)
	{
		int pNum = m_markers[camid].size(); 
        m_i_markers[camid].resize(pNum); 
		for (int i = 0; i < pNum; i++)
		{
			Eigen::Vector3d ph; 
			ph.block<2, 1>(0, 0) = m_markers[camid][i];
			ph(2) = 1; 
			Eigen::Vector3d pImagePlane = invK * ph; 
			m_i_markers[camid][i] = pImagePlane.segment<2>(0); 
		}
	}
}

void Calibrator::save_results(std::string result_folder)
{
    if(!boost::filesystem::exists(result_folder))
    {
        boost::filesystem::create_directories(result_folder); 
    }

    // save r and t
    for(int i = 0; i < m_camids.size(); i++)
    {
        std::stringstream ss; 
        ss << result_folder << "/" << std::setw(2) << std::setfill('0') << m_camids[i]<< ".txt"; 
        std::ofstream is;
        is.open(ss.str()); 
        if(!is.is_open())
        {
            std::cout << "error openning " << ss.str() << std::endl; 
            exit(-1); 
        }
        for(int j = 0; j < 3; j++)
        {
            is << out_rvecs[i][j] << "\n"; 
        }
        for(int j = 0; j < 3; j++)
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
    if(!os.is_open())
    {
        std::cout << "can not open " << ss.str() << std::endl; 
        exit(-1); 
    } 
    for(int i = 0; i < out_points.size(); i++)
    {
        os << out_points[i].transpose() << "\n"; 
    }
    for(int i = 0; i < out_points_new.size(); i++)
    {
        os << out_points_new[i].transpose() << "\n"; 
    }
    os.close(); 

    // save ratio 
    std::stringstream ss1; 
    ss1 << result_folder << "/ratio.txt";
    std::ofstream os1; 
    os1.open(ss1.str());
    if(!os1.is_open())
    {
        std::cout << "can not open " << ss1.str() << std::endl; 
        exit(-1); 
    } 
    os1 << out_ratio << "\n"; 
    os1.close(); 
}


void Calibrator::evaluate()
{
	std::vector<Camera> cams; 
	for(int view = 0; view < m_camNum; view++)
	{
		Camera cam = getDefaultCameraUndist(); 
		cam.SetRT(out_rvecs[view], out_tvecs[view]); 
		cams.push_back(cam); 
	}

    // project initial markers 
	vector<vector<Vec3> > projs; 
	projs.resize(m_camNum); 
	std::cout << out_points.size() << std::endl; 
	for(int v = 0; v < m_camNum; v++)
	{
		vector<Vec3> proj;
		project(cams[v], out_points, proj); 
		projs[v] = proj; 
	}

    // porject added markers 
    vector<vector<Vec3> > projs_new; 
    projs_new.resize(m_added.size()); 
    for(int i = 0; i < projs_new.size(); i++)
    {
        projs_new[i].resize(m_camNum); 
        for(int camid = 0; camid < m_camNum; camid++)
        {
            Vec3 p2d = project(m_camsUndist[camid], out_points_new[i]);
            projs_new[i][camid] = p2d; 
        }
    }
    m_projs_markers = projs; 
    m_projs_added   = projs_new; 

    // compute errors 
    double total_errs = 0;
    int num = 0; 
    for(int v = 0; v < m_camNum; v++)
    {
        for(int i = 0; i < m_markers[v].size(); i++)
        {
            Vec2 gt = m_markers[v][i];
            Vec3 projection = projs[v][i];
            // std::cout << gt.transpose() << " ......  " << projection.transpose() << std::endl; 
            Vec2 err = gt - projection.segment<2>(0);
            total_errs += err.norm(); 
            num+=1; 
        }
    }
    for(int i = 0; i < m_added.size(); i++)
    {
        for(int camid = 0; camid < m_camNum; camid++)
        {
            Vec3 gt  = m_added[i][camid];
            Vec3 est = m_projs_added[i][camid];
            if(gt(0) < 0) continue; 
            // std::cout << gt.transpose() << " ......  " << est.transpose() << std::endl; 
            Vec2 err = gt.segment<2>(0) - est.segment<2>(0);
            total_errs += err.norm(); 
            num += 1; 
        }
    }
    std::cout << "avg err: " << total_errs / num << std::endl; 
}

void Calibrator::draw_points()
{
    cloneImgs(m_imgsUndist, m_imgsDraw); 
    for(int v = 0; v < m_camNum; v++)
    {
        std::vector<Vec3> points; 
        for(int i = 0; i < m_markers[v].size(); i++)
        {
            Vec3 p;
            p.segment<2>(0) = m_markers[v][i];
            p(2) = 1; 
            points.push_back(p); 
        }
        my_draw_points(m_imgsDraw[v], points, m_CM[1], 10); 
        my_draw_points(m_imgsDraw[v], m_projs_markers[v], m_CM[2], 5);
    
        std::vector<Vec3> points2d_gt_added;
        std::vector<Vec3> points2d_est_added; 
        for(int i = 0; i < m_added.size(); i++)
        {
            if(m_added[i][v](0) < 0) continue; 
            points2d_gt_added.push_back(m_added[i][v]);
            points2d_est_added.push_back(m_projs_added[i][v]);  
        }
        my_draw_points(m_imgsDraw[v], points2d_gt_added, m_CM[1], 10); 
        my_draw_points(m_imgsDraw[v], points2d_est_added, m_CM[2], 5); 
    }
    
    // cv::Mat output;
    // packImgBlock(m_imgsUndist, output); 
    // cv::namedWindow("raw", cv::WINDOW_NORMAL); 
    // cv::imshow("raw", output); 
    // int key = cv::waitKey(); 
}

void Calibrator::readImgs()
{
    std::string m_imgDir = folder + "/data/calib_1_color/";
    for(int camid = 0; camid < m_camNum; camid++)
    {
        std::stringstream ss; 
        ss << m_imgDir << std::setw(2) << std::setfill('0') << m_camids[camid] << ".jpg";
        cv::Mat img = cv::imread(ss.str()); 
        if(img.empty())
        {
            std::cout << "img is empty! " << ss.str() << std::endl; 
            exit(-1); 
        }
        m_imgs.push_back(img);
    }

    m_imgsUndist.resize(m_camNum); 
    for(int i = 0; i < m_camNum; i++)
    {
        my_undistort(m_imgs[i], m_imgsUndist[i], m_cams[i], m_camsUndist[i]); 
    }

    cloneImgs(m_imgsUndist, m_imgsDraw); 
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    Vec3 empty = {-1,-1,-1}; 
    vector<Vec3>* data = (vector<Vec3>*) userdata; 
    int length = data->size(); 
    int cols = sqrt(length); 
    if(cols * cols < length) cols+=1; 
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        int c = x / 1920;
        int r = y / 1080; 
        int dx = x - c * 1920; 
        int dy = y - r * 1080; 
        int id = r * cols + c; 
        data->data()[id](0) = dx; 
        data->data()[id](1) = dy; 
        data->data()[id](2) = 1; 
        std::cout << "Add point at cam " << id << " - position (" << dx << ", " << dy << ")" << std::endl;
    }
    else if  ( event == cv::EVENT_RBUTTONDOWN )
    {
        int c = x / 1920;
        int r = y / 1080; 
        int id = r * cols + c; 
        // data[id] = empty; 
        std::cout << "Cancel mark at cam " << id << " - position (" << x << ", " << y << ")" << std::endl;
    }
    else if  ( event == cv::EVENT_MBUTTONDOWN )
    {
        int c = x / 1920;
        int r = y / 1080; 
        std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
//  else if ( event == cv::EVENT_MOUSEMOVE )
//  {
//       std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
//  }
}

void Calibrator::save_added()
{
    std::string path = folder+"/data/add_marker/";
    for(int v = 0; v < m_camNum; v++)
    {
        int camid = m_camids[v];
        std::stringstream ss;
        ss << path << std::setw(2) << std::setfill('0') << camid << ".txt"; 
        std::ofstream outstream; 
        outstream.open(ss.str()); 
        if(!outstream.is_open())
        {
            std::cout << "can not open save file " << ss.str() << std::endl; 
            exit(-1); 
        }
        for(int i = 0; i < m_added.size(); i++)
        {
            outstream << m_added[i][v](0) << " " << m_added[i][v](1);
            if(i != m_added.size()-1) outstream << "\n";
        }
        outstream.close(); 
    }
}

void Calibrator::reload_added()
{
    std::string path = folder+"/data/add_marker/";
    m_added.clear(); 
    vector<vector<Vec3> > tmp; 
    tmp.resize(m_camNum); 
    for(int v = 0; v < m_camNum; v++)
    {
        int camid = m_camids[v];
        std::stringstream ss;
        ss << path << std::setw(2) << std::setfill('0') << camid << ".txt"; 
        std::ifstream inputstream; 
        inputstream.open(ss.str()); 
        if(!inputstream.is_open())
        {
            std::cout << "can not open save file " << ss.str() << std::endl; 
            return;
        }
        while(true)
        {
            if(inputstream.eof()) break; 
            double x,y; 
            inputstream >> x; 
            if(inputstream.eof()) break;
            inputstream >> y; 
            Eigen::Vector3d p;
            p(0) = x; p(1) = y; p(2) = 1; 
            tmp[v].push_back(p); 
        };
        inputstream.close(); 
    }

    for(int i = 0; i < tmp[0].size(); i++)
    {
        vector<Vec3> p_list; 
        for(int camid = 0; camid < m_camNum; camid++)
        {
            p_list.push_back(tmp[camid][i]); 
        }
        m_added.push_back(p_list); 
    }

    // ATTENTION!! must precompute all camera params 
    out_points_new.resize(m_added.size()); 
    for(int i = 0; i < m_added.size(); i++)
    {
        std::vector<Camera> visible_cams;
        // std::vector<Vec2> points2d; 
        std::vector<Vec3> points2d; 
        for(int camid = 0; camid < m_camNum; camid++)
        {
            if(m_added[i][camid](0) < 0) continue; 
            visible_cams.push_back(m_camsUndist[camid]);
            // Vec2 px = m_added[i][camid].segment<2>(0); 
            // points2d.push_back(px); 
            points2d.push_back(m_added[i][camid]); 
        }
        // Vec3 p3d = NViewDLT(visible_cams, points2d);
        Vec3 p3d = triangulate_ceres(visible_cams, points2d);
        out_points_new[i] = p3d; 
    }

    std::cout << "Have reload " << out_points_new.size() << " new points " << std::endl; 

    if(m_added.size() >= 3)
    {
        Vec3 p1 = out_points_new[0];
        Vec3 p2 = out_points_new[1]; 
        Vec3 p3 = out_points_new[2]; 
        // check pependicular
        Vec3 v1 = p2 - p1; 
        Vec3 v2 = p3 - p1; 
        double angle = std::acos(v1.dot(v2));
        std::cout << "In reality, v1.norm == 0.5m, p3.z == 0.6m" << std::endl; 
        std::cout << "v1:         " << v1.transpose() << std::endl; 
        std::cout << "v2:         " << v2.transpose() << std::endl; 
        std::cout << "v1.dot(v2): " << v1.dot(v2) << std::endl; 
        std::cout << "angle(v1v2):" << angle << std::endl;
        std::cout << "v1.norm():  " << v1.norm() << std::endl; 
        std::cout << "v2.norm():  " << v2.norm() << std::endl; 
        std::cout << "p3.z:       " << p3(2) << std::endl; 
    }
}

void Calibrator::interactive_mark()
{
    while(true)
    {
        std::vector<Vec3> marks; 
        marks.resize(m_camNum); 
        for(int i = 0; i < m_camNum; i++) marks[i] = {-1, -1, -1}; 

        cv::Mat output; 
        packImgBlock(m_imgsDraw, output); 
        cv::namedWindow("add_marker", cv::WINDOW_NORMAL); 
        cv::setMouseCallback("add_marker", CallBackFunc, &marks); 
        cv::imshow("add_marker", output); 
        int key = cv::waitKey(); 
        if(key == 27) 
        {
            break; 
        }
        else if(char(key) == 'n')
        {
            m_added.push_back(marks); 
            std::vector<Camera> visible_cams;
            // std::vector<Vec2> points2d; 
            std::vector<Vec3> points2d; 
            for(int camid = 0; camid < m_camNum; camid++)
            {
                std::cout << marks[camid].transpose() << std::endl;
                if(marks[camid](0) < 0) continue; 
                visible_cams.push_back(m_camsUndist[camid]);
                // Vec2 px = marks[camid].segment<2>(0); 
                // points2d.push_back(px); 
                points2d.push_back(marks[camid]); 
            }
            // Vec3 p3d = NViewDLT(visible_cams, points2d);
            Vec3 p3d = triangulate_ceres(visible_cams, points2d); 
            for(int camid = 0; camid < m_camNum; camid++)
            {
                if(marks[camid](0) < 0) continue; 
                marks[camid] = m_camsUndist[camid].inv_K * marks[camid]; 
            }
            
            ba.addMarker(marks, p3d); 
            ba.solve_again(); 
            out_points_new = ba.getAddedPoints(); 
            evaluate();
            draw_points(); 
        }
        else if(char(key) == 'd')
        {
            
        }
    };
}

int Calibrator::calib_pipeline()
{
	std::string marker_folder = folder+"/python/markers";
	std::string K_file = folder+"/python/data/newK.txt"; 
	readK(K_file); 
	readAllMarkers(marker_folder); 
	unprojectMarkers(); 

	std::cout << "data prepared. " << std::endl; 

	ba.initMarkers(m_camids, 42); 
	ba.readInit(folder); 
	ba.setObs(m_i_markers); 
	ba.solve_init_calib(true); 
	std::cout << "initial calibration done. " << std::endl; 

	out_points = ba.getPoints(); 
	out_rvecs = ba.getRvecs(); 
	out_tvecs = ba.getTvecs(); 
    out_ratio = ba.getRatio(); 

    double z = 0; 
    for(int i = 0; i < out_points.size(); i++) z += out_points[i](2); 
    z /= out_points.size(); 
    std::cout << "average floor height: " <<  z << std::endl; 

    // reset camera 
    for(int camid = 0; camid < m_camNum; camid++) 
    {
        m_camsUndist[camid].SetRT(out_rvecs[camid], out_tvecs[camid]);
        m_cams[camid].SetRT(out_rvecs[camid], out_tvecs[camid]); 
    }

    // evalute 
    evaluate(); 
    draw_points(); 

    // re-calib with addtional data 
    reload_added(); 
    if(m_added.size() > 0)
    {
        for(int i = 0; i < m_added.size(); i++)
        {
            vector<Vec3> marks; 
            marks.resize(m_camNum); 
            for(int camid = 0; camid < m_camNum; camid++)
                marks[camid] = m_camsUndist[camid].inv_K * m_added[i][camid]; 
            ba.addMarker(marks, out_points_new[i]); 
        }
        ba.solve_again();
        out_points = ba.getPoints();
        out_points_new = ba.getAddedPoints(); 
        out_rvecs = ba.getRvecs(); 
        out_tvecs = ba.getTvecs(); 
        for(int camid = 0; camid < m_camNum; camid++) 
        {
            m_camsUndist[camid].SetRT(out_rvecs[camid], out_tvecs[camid]);
            m_cams[camid].SetRT(out_rvecs[camid], out_tvecs[camid]); 
        }
    }
    evaluate();
    draw_points(); 
    //save_results("results"); 

    // interactive calib 
    interactive_mark(); 
    //save_added();
    //save_results("results"); 

	return 0; 
}

void Calibrator::read_results_rt(std::string result_folder)
{
    // save r and t
    for(int i = 0; i < m_camids.size(); i++)
    {
        std::stringstream ss; 
        ss << result_folder << "/" << std::setw(2) << std::setfill('0') << m_camids[i]<< ".txt"; 
        std::ifstream is;
        is.open(ss.str()); 
        if(!is.is_open())
        {
            std::cout << "error openning " << ss.str() << std::endl; 
            exit(-1); 
        }
        Eigen::Vector3d r_vec; 
        Eigen::Vector3d t_vec;
        for(int j = 0; j < 3; j++)
        {
            is >> r_vec(j); 
        }
        for(int j = 0; j < 3; j++)
        {
            is >> t_vec(j);
        }
        m_camsUndist[i].SetRT(r_vec, t_vec); 
        is.close(); 
    }
}

void Calibrator::test_epipolar()
{
    std::string marker_folder = folder+"/python/markers";
	std::string K_file = folder+"/python/data/newK.txt"; 
	readK(K_file); 
	readAllMarkers(marker_folder); 
    read_results_rt("results"); 

    test_epipolar_all(m_camsUndist, m_imgsUndist, m_markers);
}