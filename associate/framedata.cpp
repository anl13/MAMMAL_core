#include "framedata.h" 
#include <json/json.h> 
#include <math.h> 
#include <algorithm>
#include "colorterminal.h"
#include "image_utils.h"

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
    m_keypointsDir = m_sequence + "/keypoints_hrnet/"; 
    m_imgDir       = m_sequence + "/images/";  
    m_boxDir       = m_sequence + "/boxes/"; 
    m_camDir       = root["camfolder"].asString(); 
    m_imgExtension = root["imgExtension"].asString(); 
    startid        = root["startid"].asInt(); 
    framenum       = root["framenum"].asInt(); 
    m_epi_thres    = root["epipolar_threshold"].asDouble(); 
    m_epi_type     = root["epipolartype"].asString(); 
    m_ransac_nms_thres = root["ransac_nms_thres"].asDouble(); 
    m_sigma        = root["sigma"].asDouble(); 
    m_boxExpandRatio = root["box_expand_ratio"].asDouble(); 
    m_pruneThreshold = root["prune_threshold"].asDouble(); 
    m_cliqueSizeThreshold = root["clique_size_threshold"].asInt(); 
    m_skelType     = root["skel_type"].asString(); 
    m_topo         = getSkelTopoByType(m_skelType); 

    std::vector<int> camids; 
    for(auto const &c : root["camids"])
    {
        int id = c.asInt(); 
        camids.push_back(id); 
    }
    setCamIds(camids); 

    instream.close(); 
}

void FrameData::readKeypoints() // load hrnet keypoints
{
    std::string jsonDir = m_keypointsDir;
    std::stringstream ss; 
    ss << jsonDir << std::setw(6) << std::setfill('0') << m_frameid << ".json";
    std::string jsonfile = ss.str(); 
    std::cout << jsonfile << std::endl; 

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

    m_dets.clear(); 

    for(int camid = 0; camid < m_camNum; camid++)
    {
        Json::Value c = root[std::to_string(m_camids[camid])]; 
        vector<vector<Eigen::Vector3d> > aframe; 
        int cand_num = c.size(); 
        for(int candid = 0; candid < cand_num; candid++)
        {
            if(candid >=4) break; 
            vector<Eigen::Vector3d> pig; 
            pig.resize(m_topo.joint_num); 
            for(int pid = 0; pid < m_topo.joint_num; pid++)
            {
                Eigen::Vector3d v;
                for(int idx = 0; idx < 3; idx++)
                {
                    v(idx) = c[candid][pid*3+idx].asDouble(); 
                }
                pig[pid] = v; 
            }
            aframe.push_back(pig);
        }
        m_dets.push_back(aframe); 
    }
    is.close(); 
}

void FrameData::readBoxes()
{
    std::string jsonDir = m_boxDir;
    std::stringstream ss; 
    ss << jsonDir << "/boxes_" << std::setw(6) << std::setfill('0') << m_frameid << ".json";
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
        std::vector<Eigen::Vector4d> bb; 
        for(int bid = 0; bid < boxnum; bid++)
        {
            if(bid >= 4) break; // remain only 4 top boxes 
            Json::Value box_jv = c[bid]; 
            Eigen::Vector4d B; 
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
            Eigen::Vector4d box = my_undistort_box(
                m_boxes_raw[cid][bid], m_cams[cid], m_camsUndist[cid]
            ); 
            m_boxes_processed[cid][bid] = expand_box(box, m_boxExpandRatio); 
        }
    }
}

void FrameData::undistKeypoints()
{
    int camNum = m_dets.size(); 
    m_dets_undist = m_dets; 
    for(int camid = 0; camid < camNum; camid++)
    {
        Camera cam = m_cams[camid]; 
        Camera camnew = m_camsUndist[camid]; 
        int candnum = m_dets[camid].size(); 
        for(int candid = 0; candid < candnum; candid++)
        {
            my_undistort_points(m_dets[camid][candid], m_dets_undist[camid][candid], cam, camnew); 
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
    undistKeypoints(); 
    readImages(); 
    undistImgs(); 
    readBoxes();
    processBoxes(); 
}



// void FrameData::reproject()
// {
//     m_dets_reproj.resize(m_camNum); 
//     for(int view = 0; view < m_camNum; view++)
//     {
//         m_dets_reproj[view].resize(20); 
//         for(int kpt_id = 0; kpt_id < 20; kpt_id ++)
//         {
//             project(m_camsUndist[view], m_points3d[kpt_id], m_dets_reproj[view][kpt_id]); 
//         }
//     }
//     // compare error 

// }

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
            m_projs[camid][id].resize(m_topo.joint_num, Eigen::Vector3d::Zero()); 
            for(int kpt_id = 0; kpt_id < m_topo.joint_num; kpt_id++)
            {
                if(m_skels3d[id][kpt_id].norm() == 0) continue; 
                Eigen::Vector3d p = m_skels3d[id][kpt_id]; 
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
        for(int k = 0; k < m_dets_undist[i].size(); k++)
        {
            drawSkel(imgdata[i], m_dets_undist[i][k], k); 
        }
        my_draw_boxes(imgdata[i], m_boxes_processed[i]); 
    }
    cv::Mat output; 
    packImgBlock(imgdata, output); 

    return output; 
}

cv::Mat FrameData::visualizeIdentity2D()
{
    std::vector<cv::Mat> imgdata;
    cloneImgs(m_imgsUndist, imgdata); 
    
    for(int id = 0; id < m_clusters.size(); id++)
    {
        for(int camid = 0; camid < m_camNum; camid++)
        {
            int candid = m_clusters[id][camid];
            if(candid < 0) continue; 
            drawSkel(imgdata[camid], m_dets_undist[camid][candid], id);
            my_draw_box(imgdata[camid], m_boxes_processed[camid][candid], m_CM[id]); 
        }
    }
    cv::Mat packed; 
    packImgBlock(imgdata, packed); 
    return packed;
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

// #include <json/writer.h> 
// void FrameData::writeSkeltoJson(std::string jsonfile)
// {
//     std::ofstream os;
//     os.open(jsonfile); 
//     if(!os.is_open())
//     {
//         std::cout << "file " << jsonfile << " cannot open" << std::endl; 
//         return; 
//     }

//     Json::Value root;
//     Json::Value pigs(Json::arrayValue);  
//     for(int index=0; index < 4; index++)
//     {
//         Json::Value pose(Json::arrayValue); 
//         for(int i = 0; i < 20; i++)
//         {
//             pose.append(Json::Value(m_skels[index].col(i)(0)) ); 
//             pose.append(Json::Value(m_skels[index].col(i)(1)) );
//             pose.append(Json::Value(m_skels[index].col(i)(2)) );
//             pose.append(Json::Value(m_skels[index].col(i)(3)) );   
//         }
//         pigs.append(pose); 
//     }
//     root["pigs"] = pigs;
    
//     // Json::StyledWriter stylewriter; 
//     // os << stylewriter.write(root); 
//     Json::StreamWriterBuilder builder;
//     builder["commentStyle"] = "None";
//     builder["indentation"] = "    "; 
//     std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter()); 
//     writer->write(root, &os); 
//     os.close(); 
// }

// void FrameData::readSkelfromJson(std::string jsonfile)
// {
//     Json::Value root;
//     Json::CharReaderBuilder rbuilder; 
//     std::string errs; 
//     std::ifstream instream(jsonfile); 
//     if(!instream.is_open())
//     {
//         std::cout << "can not open " << jsonfile << std::endl; 
//         exit(-1); 
//     }
//     bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs); 
//     if(!parsingSuccessful)
//     {
//         std::cout << "Fail to parse \n" << errs << std::endl; 
//         exit(-1); 
//     }

//     m_skels.clear(); 
//     for(auto const &pig: root["pigs"])
//     {
//         PIG_SKEL skel; 
//         for(int index=0; index < 20; index++)
//         {
//             double x = pig[index * 4 + 0].asDouble(); 
//             double y = pig[index * 4 + 1].asDouble();
//             double z = pig[index * 4 + 2].asDouble(); 
//             double w = pig[index * 4 + 3].asDouble(); 
//             Eigen::Vector4d vec(x,y,z,w);
//             skel.col(index) = vec; 
//         }
//         m_skels.push_back(skel);
//     }
//     instream.close(); 
//     std::cout << "read json done. " << std::endl; 
// }

cv::Mat FrameData::test()
{
    cv::Mat output = visualizeSkels2D(); 

    return output; 
}

void FrameData::drawSkel(cv::Mat& img, const vector<Eigen::Vector3d>& _skel2d, int colorid)
{
    Eigen::Vector3i color = m_CM[colorid];
    cv::Scalar cv_color(color(0), color(1), color(2)); 
    for(int i = 0; i < _skel2d.size(); i++)
    {
        cv::Point2d p(_skel2d[i](0), _skel2d[i](1)); 
        double conf = _skel2d[i](2); 
        if(conf < m_topo.kpt_conf_thresh[i]) continue; 
        cv::circle(img, p, 9, cv_color, -1); 
    }
    for(int k = 0; k < m_topo.bone_num; k++)
    {
        Eigen::Vector2i b = m_topo.bones[k]; 
        Eigen::Vector3d p1 = _skel2d[b(0)];
        Eigen::Vector3d p2 = _skel2d[b(1)]; 
        if(p1(2) < m_topo.kpt_conf_thresh[b(0)] || p2(2) < m_topo.kpt_conf_thresh[b(1)]) continue; 
        cv::Point2d p1_cv(p1(0), p1(1)); 
        cv::Point2d p2_cv(p2(0), p2(1)); 
        cv::line(img, p1_cv, p2_cv, cv_color, 4); 
    }
}