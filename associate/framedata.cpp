#include "framedata.h" 
#include <json/json.h> 
#include <math.h> 
#include <algorithm>

#define DEBUG_A 

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
    m_keypointsDir = root["keypointsfolder"].asString(); 
    m_camDir       = root["camfolder"].asString(); 
    m_imgDir       = root["imgfolder"].asString(); 
    m_imgExtension = root["imgExtension"].asString(); 
    startid        = root["startid"].asInt(); 
    framenum       = root["framenum"].asInt(); 
    m_epi_thres    = root["epipolar_threshold"].asDouble(); 
    m_epi_type     = root["epipolartype"].asString(); 
    std::vector<int> camids; 
    for(auto const &c : root["camids"])
    {
        int id = c.asInt(); 
        camids.push_back(id); 
    }
    setCamIds(camids); 
    
    m_kpts_to_show.clear(); 
    for(auto const &c : root["keypoints_show"])
    {
        int id = c.asInt(); 
        m_kpts_to_show.push_back(id); 
    }
    
    m_keypoint_conf_thres.clear(); 
    for(auto const &c : root["keypoint_conf_thres"])
    {
        double conf = c.asDouble(); 
        m_keypoint_conf_thres.push_back(conf); 
    }
    instream.close(); 
}

void FrameData::readKeypoints()
{
    std::string jsonDir = m_keypointsDir;
    std::stringstream ss; 
    ss << jsonDir << "keypoints_" << std::setw(6) << std::setfill('0') << m_frameid << ".json";
    readKeypoint(ss.str()); 
}

void FrameData::readKeypoint(std::string jsonfile)
{
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

    dets.clear(); 

    // for(auto const &c : root["keypoints"])
    for(int idx = 0; idx < m_camNum; idx++)
    {
        int camid = m_camids[idx];
        Json::Value c = root[std::to_string(camid)]; 
        vector<vector<Eigen::Vector3d> > aframe; 
        for(int i =0;i<20;i++)
        {
            std::string key = std::to_string(i); 
            int id = 0; 
            vector<Eigen::Vector3d> kpts; 
            int point_num = c[key].size() / 3; 
            for(int pid = 0; pid < point_num; pid++)
            {
                Eigen::Vector3d v;
                for(int idx = 0; idx < 3; idx++)
                {
                    v(idx) = c[key][pid*3+idx].asDouble(); 
                }
                v(0) = v(0) / 1024.0;
                v(1) = v(1) / 512.0;
                if(v(2) > m_keypoint_conf_thres[i])
                    kpts.push_back(v); 
            }
            aframe.push_back(kpts);
        }
        dets.push_back(aframe); 
    }
    is.close(); 
}

void FrameData::undistKeypoints(const Camera& cam, const Camera& camnew, int imw, int imh)
{
    int camNum = dets.size(); 
    dets_undist = dets; 
    for(int camid = 0; camid < camNum; camid++)
    {
        int kpt_num = dets[camid].size(); 
        for(int kpt_id = 0; kpt_id < kpt_num; kpt_id++)
        {
            int pnum = dets[camid][kpt_id].size(); 
            if(pnum == 0) continue; 
            for(int pid = 0; pid < pnum; pid++)
            {
                dets[camid][kpt_id][pid](0) *= imw; 
                dets[camid][kpt_id][pid](1) *= imh;
            }
            my_undistort_points(dets[camid][kpt_id], dets_undist[camid][kpt_id], cam, camnew); 
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
        m_imgs.emplace_back(cv::imread(ss.str()));
    }
}

void FrameData::readCameras()
{
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
        Vec3 rvec, tvec; 
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
        Camera cam = getDefaultCameraRaw(); 
        cam.SetRT(rvec,  tvec); 
        Camera camUndist = getDefaultCameraUndist(); 
        camUndist.SetRT(rvec, tvec); 
        m_cams.push_back(cam); 
        m_camsUndist.push_back(camUndist); 
        camfile.close(); 
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
    readKeypoints(); 
    undistKeypoints(m_cams[0], m_camsUndist[0],m_imw, m_imh); 
    readImages(); 
    undistImgs(); 
}

cv::Mat FrameData::test()
{
    for(int i = 0; i < m_camNum; i++)
    {
        for(int kpt_id = 0; kpt_id < 20; kpt_id ++)
            my_draw_points(m_imgsUndist[i], dets_undist[i][kpt_id], m_CM[kpt_color_id[kpt_id]]); 
    }
    cv::Mat output; 
    packImgBlock(m_imgsUndist, output); 

    return output; 
}

void FrameData::checkEpipolar(int kpt_id)
{
    // repack keypoints 
    std::vector<std::vector<Vec3> > points; // points of same kpt type
    int totalPointNum = 0; 
    std::vector<int> view_ids;
    std::vector<int> cand_ids;
    for(int i = 0; i < m_camNum; i++)
    {
        points.push_back(dets_undist[i][kpt_id]);
        totalPointNum += dets_undist[i][kpt_id].size(); 
        for(int j = 0; j < dets_undist[i][kpt_id].size(); j++) 
        {
            view_ids.push_back(i); 
            cand_ids.push_back(j); 
        }
    }
    
    // construct graph 
    Eigen::MatrixXd G;
    G.resize(totalPointNum, totalPointNum);
    G.setZero(); 
    for(int index_i=0; index_i<totalPointNum; index_i++)
    {
        for(int index_j=0; index_j<totalPointNum; index_j++)
        {
            if(index_j == index_i)
            {
                G(index_i, index_j) = -1; 
                continue; 
            }
            int viewa = view_ids[index_i];
            int pida  = cand_ids[index_i];
            Vec3 pa = points[viewa][pida];

            int viewb = view_ids[index_j];
            int pidb  = cand_ids[index_j];
            Vec3 pb = points[viewb][pidb]; 
            if(viewa == viewb) 
            {
                G(index_i, index_j) = -1;
                continue; 
            }
            double dist = getEpipolarDistL2L(m_camsUndist[viewa], m_camsUndist[viewb], pa,pb);
            G(index_i, index_j) = dist;
        }
    }
    std::cout << G.block<10,10>(0,0) << std::endl; 

    // visualize 
    cv::Mat output; 
    packImgBlock(m_imgsUndist, output); 
    int cols = sqrt(m_camNum); 
    if(cols * cols < m_camNum) cols += 1; 
    for(int index_i = 0; index_i < totalPointNum; index_i++)
    {
        for(int index_j = 0; index_j < totalPointNum; index_j++)
        {
            if(index_i == index_j) continue; 
            if(G(index_i, index_j) < 0) continue; 
            double dist = G(index_i, index_j); 
            if(dist > 0.2) continue; 
            int linewidth = (0.6 - dist) * 10 + 2;
            // int linewidth = int(12 - dist); 
            int viewa = view_ids[index_i]; 
            int viewb = view_ids[index_j];
            int canda = cand_ids[index_i];
            int candb = cand_ids[index_j];
            Vec3 pa = points[viewa][canda];
            Vec3 pb = points[viewb][candb];
            int block_r_a = viewa / cols; 
            int block_c_a = viewa % cols; 
            int block_r_b = viewb / cols; 
            int block_c_b = viewb % cols;
            pa(0) += block_c_a * m_imw; 
            pa(1) += block_r_a * m_imh; 
            pb(0) += block_c_b * m_imw; 
            pb(1) += block_r_b * m_imh; 
            Eigen::Vector3i color = m_CM[kpt_id]; 
            my_draw_segment(output, pa,pb,color,linewidth); 
        }
    }

    cv::namedWindow("associate", cv::WINDOW_NORMAL); 
    cv::imshow("associate", output); 
    int key = cv::waitKey(); 
    cv::destroyAllWindows(); 
}


void FrameData::epipolarClustering(int kpt_id, vector<Vec3> & p3ds)
{
    // repack keypoints 
    std::vector<std::vector<Vec3> > points; // points of same kpt type
    int totalPointNum = 0; 
    std::vector<int> view_ids;
    std::vector<int> cand_ids;
    std::vector<std::pair<int,int>> table; 
    std::vector<std::vector<int> > invTable; 
    int index = 0; 
    for(int i = 0; i < m_camNum; i++)
    {
        points.push_back(dets_undist[i][kpt_id]);
        std::vector<int> subinvtable;
        subinvtable.resize(dets_undist[i][kpt_id].size()); 
        totalPointNum += dets_undist[i][kpt_id].size(); 
        for(int j = 0; j < dets_undist[i][kpt_id].size(); j++) 
        {
            view_ids.push_back(i); 
            cand_ids.push_back(j); 
            std::pair<int, int> T; 
            T.first = i; T.second = j; 
            table.push_back(T); 
            subinvtable[j] = index; 
            index ++; 
        }
        invTable.push_back(subinvtable);   
    }

    // construct graph 
    Eigen::MatrixXd G;
    G.resize(totalPointNum, totalPointNum);
    G.setZero(); 
    double (*distfunc)(const Camera&, const Camera&, const Eigen::Vector3d&, const Eigen::Vector3d&); 
    // distfunc = &getEpipolarDist; 
    if(m_epi_type == "p2l")
        distfunc = &getEpipolarDist; 
    else distfunc = &getEpipolarDistL2L;
    for(int index_i=0; index_i<totalPointNum; index_i++)
    {
        for(int index_j=0; index_j<totalPointNum; index_j++)
        {
            if(index_j == index_i)
            {
                G(index_i, index_j) = -1; 
                continue; 
            }
            int viewa = view_ids[index_i];
            int pida  = cand_ids[index_i];
            Vec3 pa = points[viewa][pida];

            int viewb = view_ids[index_j];
            int pidb  = cand_ids[index_j];
            Vec3 pb = points[viewb][pidb]; 
            if(viewa == viewb) 
            {
                G(index_i, index_j) = -1;
                continue; 
            }
            // double dist = getEpipolarDist(m_camsUndist[viewa], m_camsUndist[viewb], pa,pb);
            double dist = distfunc(m_camsUndist[viewa], m_camsUndist[viewb], pa,pb);
            G(index_i, index_j) = dist;
        }
    }
    // std::cout << G << std::endl; 
    // std::cout << "vertexnum: " << totalPointNum << std::endl; 

    // associate 
    ClusterClique CC; 
    CC.G = G; 
    CC.table = table; 
    CC.invTable = invTable; 
    CC.vertexNum = totalPointNum;
    if(kpt_id == 18)
    {
        CC.threshold = m_epi_thres * 5;
    }
    else 
    {
        CC.threshold = m_epi_thres;
    }
    CC.constructGraph(); 
    // CC.enumerateMaximalCliques();
    CC.enumerateBestCliques(); 
    // CC.printGraph();  
    // CC.printAllMC();
    // CC.printCliques(); 
    auto cliques = CC.cliques; 

    m_cliques[kpt_id] = cliques;
    m_tables[kpt_id] = table; 
    m_invTables[kpt_id] = invTable; 
    m_G[kpt_id] = G; 

    std::vector<Eigen::Vector3d> joints3d; 
    for(int cliqueIdx = 0; cliqueIdx < cliques.size(); cliqueIdx++)
    {
        std::vector<int> clique = cliques[cliqueIdx]; 
        if(clique.size() == 1) continue; 
        std::vector<Camera> cams_visible; 
        std::vector<Eigen::Vector3d> joints2d; 
        int clique_size = clique.size(); 
        for(int i = 0; i < clique_size; i++)
        {
            int vertex_idx = clique[i];
            int view = table[vertex_idx].first; 
            int cand = table[vertex_idx].second; 
            Vec3 p = dets_undist[view][kpt_id][cand];
            // p(2) = 1;
            joints2d.push_back(p); 
            cams_visible.push_back(m_camsUndist[view]);
        }
        Joint3DSolver solver; 
        solver.SetParam(cams_visible, joints2d); 
        solver.SetVerbose(false); 
        solver.Solve3D(); 
        Eigen::Vector3d X = solver.GetX(); 
        joints3d.push_back(X); 
    }

    p3ds = joints3d; 
}

void FrameData::compute3d()
{
    points3d.resize(20);
    for(int kpt_id = 0; kpt_id < 20; kpt_id++)
    {
        vector<Vec3> p3ds; 
        epipolarClustering(kpt_id, p3ds);
        points3d[kpt_id] = p3ds;  
    }
}

void FrameData::reproject()
{
    dets_reproj.resize(m_camNum); 
    for(int view = 0; view < m_camNum; view++)
    {
        dets_reproj[view].resize(20); 
        for(int kpt_id = 0; kpt_id < 20; kpt_id ++)
        {
            project(m_camsUndist[view], points3d[kpt_id], dets_reproj[view][kpt_id]); 
        }
    }
}

cv::Mat FrameData::visualize(int type, int Kid)
{
    vector<vector<vector<Vec3> > > data; 
    if(type==0) data = dets; 
    else if (type==1) data = dets_undist;
    else if (type==2) data = dets_reproj;
    else 
    {
        std::cout << "no such type "  << std::endl;
        exit(-1); 
    }

    std::vector<cv::Mat> imgdata;
    if(type==0) cloneImgs(m_imgs, imgdata); 
    else if(type==1 || type==2) cloneImgs(m_imgsUndist, imgdata);
    else{
        std::cout << "images not defined" << std::endl; 
        exit(-1); 
    }

    if(Kid < 0)
    {
        for(int view = 0; view < m_camNum; view++)
        {
            for(int i = 0; i < m_kpts_to_show.size(); i++)
            {
                int kpt_id = m_kpts_to_show[i];
                int color_id = kpt_color_id[kpt_id]; 
                my_draw_points(imgdata[view], data[view][kpt_id], m_CM[color_id]);
            }
        }
    }
    else
    {
        for(int view = 0; view < m_camNum; view++)
        {
            int colorid = kpt_color_id[Kid];
            
            for(int i = 0; i < data[view][Kid].size(); i++)
            {
                vector<Eigen::Vector3d> points; 
                points.push_back(data[view][Kid][i]); 
                my_draw_points(imgdata[view], points, m_CM[i]); 
            }
        }
    }
    
    cv::Mat packed; 
    packImgBlock(imgdata, packed); 
    return packed; 
}


cv::Mat FrameData::visualizeClique(int kpt_id)
{
    auto cliques = m_cliques[kpt_id]; 
    auto G = m_G[kpt_id]; 
    auto table = m_tables[kpt_id];
    auto invTable = m_invTables[kpt_id];

    cv::Mat output; 
    packImgBlock(m_imgsUndist, output); 
    int cols = sqrt(m_camNum); 
    if(cols*cols < m_camNum) cols+=1; 
    // visualize
    for(int cliqueIdx = 0; cliqueIdx < cliques.size(); cliqueIdx++)
    {
        if(cliques[cliqueIdx].size() == 1)
        {
            int node_i = cliques[cliqueIdx][0];
            int viewa = table[node_i].first;
            int canda = table[node_i].second;
            int block_r_a = viewa / cols; 
            int block_c_a = viewa % cols; 
            Vec3 pa = dets_undist[viewa][kpt_id][canda];
            pa(0) += block_c_a * m_imw; 
            pa(1) += block_r_a * m_imh; 
            Eigen::Vector3i color = m_CM[cliqueIdx];
            std::vector<Vec3> pas;
            pas.push_back(pa); 
            my_draw_points(output, pas, color, 14);
        }
        else{
        for(int cid = 0; cid < cliques[cliqueIdx].size(); cid++)
        {
            for(int cid2 = cid+1; cid2 < cliques[cliqueIdx].size(); cid2++)
            {
                int index_i = cliques[cliqueIdx][cid];
                int index_j = cliques[cliqueIdx][cid2]; 

                if(index_i == index_j) continue; 
                if(G(index_i, index_j) < 0) continue; 
                double dist = G(index_i, index_j); 
                // if(dist > 0.2) continue; 
                int linewidth = 10;
                // int linewidth = int(12 - dist); 
                int viewa = table[index_i].first; 
                int viewb = table[index_j].first;
                int canda = table[index_i].second;
                int candb = table[index_j].second;
                Vec3 pa = dets_undist[viewa][kpt_id][canda];
                Vec3 pb = dets_undist[viewb][kpt_id][candb];
                int block_r_a = viewa / cols; 
                int block_c_a = viewa % cols; 
                int block_r_b = viewb / cols; 
                int block_c_b = viewb % cols;
                pa(0) += block_c_a * m_imw; 
                pa(1) += block_r_a * m_imh; 
                pb(0) += block_c_b * m_imw; 
                pb(1) += block_r_b * m_imh; 
                Eigen::Vector3i color = m_CM[cliqueIdx]; 
                my_draw_segment(output, pa,pb,color,0, 14); 
            }
        } }
        // cv::namedWindow("assoc", cv::WINDOW_NORMAL); 
        // cv::imshow("assoc", output); 
        // int key = cv::waitKey(); 
        // if(key == 27) exit(0); 
    }

    return output; 
}