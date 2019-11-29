#include "matching.h" 

// #define DEBUG_MATCH

bool equal_concensus(const ConcensusData& data1, const ConcensusData& data2)
{
    if(data1.num != data2.num) return false;
    return my_equal(data1.ids, data2.ids); 
}

bool compare_concensus(ConcensusData data1, ConcensusData data2)
{
    if(data1.num < data2.num) return true; 
    if(data1.num > data2.num) return false; 
    // if(data1.metric < data2.metric) return true; 
    // if(data1.metric > data2.metric) return false; 
    for(int i = 0; i < data2.ids.size(); i++)
    {
        if(data1.ids[i] < data2.ids[i]) return true; 
        if(data1.ids[i] > data2.ids[i]) return false; 
    }
    return false; 
}

bool equal_concensus_list( std::vector<ConcensusData> data1,  std::vector<ConcensusData> data2)
{
    if(data1.size() != data2.size()) return false; 
    std::sort(data1.begin(), data1.end(), compare_concensus); 
    std::sort(data2.begin(), data2.end(), compare_concensus); 
    for(int i = 0; i < data1.size(); i++)
    {
        if(!equal_concensus(data1[i], data2[i])) return false; 
    }
    return true; 
}


/*
Epipolar Matching Functions 
AN Liang 20191129 Fri
*/

void EpipolarMatching::epipolarWholeBody(const Camera& cam1, const Camera& cam2, 
        const vector<Eigen::Vector3d>& pig1, const vector<Eigen::Vector3d>& pig2,
        double& avg_loss, int& matched_num)
{
    double (*distfunc)(const Camera&, const Camera&, const Eigen::Vector3d&, const Eigen::Vector3d&); 
    if(m_epi_type == "p2l")
        distfunc = &getEpipolarDist; 
    else if(m_epi_type == "l2l") 
        distfunc = &getEpipolarDistL2L;
    else 
    {
        std::cout << "Epipolar Type " << m_epi_type << " not implemented yet" << std::endl; 
    }
    double total_loss = 0; 
    int valid = 0; 
    for(int i = 0; i < m_topo.joint_num; i++)
    {
        Eigen::Vector3d v1 = pig1[i];
        Eigen::Vector3d v2 = pig2[i];
        if(v1(2) < m_topo.kpt_conf_thresh[i] || v2(2) < m_topo.kpt_conf_thresh[i])
        {
            continue; 
        }
        double epi_dist = distfunc(cam1, cam2, v1, v2); 
        total_loss += epi_dist;
        valid += 1; 
    }
    if(valid == 0) 
    {
        matched_num = 0; 
        avg_loss = -1; 
    }
    else
    {
        matched_num = valid; 
        avg_loss = total_loss / (valid); 
    }
}

void EpipolarMatching::epipolarSimilarity()
{
    // build table and invtable 
    m_total_detection_num = 0; 
    m_table.clear(); 
    m_inv_table.clear(); 
    int cam_num = m_cams.size(); 
    m_inv_table.resize(cam_num); 
    for(int i = 0; i < cam_num; i++)
    {
        int cand_num = m_dets[i].size();
        m_inv_table[i].resize(cand_num);  
        for(int j = 0; j < cand_num; j++)
        {
            std::pair<int,int> query_record;
            query_record.first = i;
            query_record.second= j; 
            m_table.push_back(query_record); 
            m_inv_table[i][j] = m_total_detection_num; 
            m_total_detection_num++; 
        }
    }

    // construct graph 
    m_G.resize(m_total_detection_num, m_total_detection_num); 
    m_G.setZero(); 
    
    for(int index_i=0; index_i<m_total_detection_num; index_i++)
    {
        for(int index_j=0; index_j<m_total_detection_num; index_j++)
        {
            // cornor case 
            if(index_i == index_j)
            {
                m_G(index_i, index_j) = -1; 
                continue;
            }
            int camid1 = m_table[index_i].first; 
            int candid1 = m_table[index_i].second; 
            int camid2 = m_table[index_j].first; 
            int candid2 = m_table[index_j].second;
            if(camid1 == camid2) 
            {
                m_G(index_i,index_j) = -1; 
                continue;
            }
            // main func
            double avg_loss; 
            int matched_num;  
            epipolarWholeBody(m_cams[camid1], m_cams[camid2], m_dets[camid1][candid1], m_dets[camid2][candid2],
                avg_loss, matched_num); 
            m_G(index_i, index_j) = avg_loss / pow(matched_num, 1); 
        }
    }

#ifdef DEBUG_MATCH
    std::cout << m_G << std::endl; 
#endif // DEBUG_MATCH
}

void EpipolarMatching::epipolarClustering()
{
    ClusterClique CC; 
    CC.G = m_G; 
    CC.table = m_table; 
    CC.invTable = m_inv_table; 
    CC.vertexNum = m_total_detection_num;
    CC.threshold = m_epi_thres;
    CC.constructAdjacentGraph(); 
    CC.enumerateBestCliques(); 
    // CC.printGraph();  
    // CC.printAllMC();
    // CC.printCliques(); 
    m_cliques = CC.cliques; 
    int cluster_num = m_cliques.size(); 
    m_clusters.resize(cluster_num); 
    for(int i = 0; i < cluster_num; i++)
    {
        m_clusters[i].resize(m_cams.size(), -1); 
        for(int j = 0; j < m_cliques[i].size(); j++)
        {
            int camid = m_table[ m_cliques[i][j] ].first; 
            int candid = m_table[ m_cliques[i][j] ].second; 
            m_clusters[i][camid] = candid; 
        }
    }
}

void EpipolarMatching::compute3dDirectly()
{
    m_skels3d.clear(); 
    for(int i = 0; i < m_clusters.size(); i++)
    {
        std::vector<Eigen::Vector3d> joints3d; 
        joints3d.resize(m_topo.joint_num, Eigen::Vector3d::Zero() ); 
        for(int kptid = 0; kptid < m_topo.joint_num; kptid++)
        {
            std::vector<Camera> cams_visible; 
            vector<Eigen::Vector3d> joints2d; // [associated cams]
            for(int camid = 0; camid < m_cams.size(); camid++)
            {
                if(m_clusters[i][camid] < 0) continue; 
                int candid = m_clusters[i][camid]; 
                if(m_dets[camid][candid][kptid](2) < m_topo.kpt_conf_thresh[kptid]) continue; 
                Eigen::Vector3d p = m_dets[camid][candid][kptid]; 
                joints2d.push_back(p);
                cams_visible.push_back(m_cams[camid]);
            }
            if(cams_visible.size() < 2) continue; 
            joints3d[kptid] = triangulate_ceres(cams_visible, joints2d); 
        }
        m_skels3d.push_back(joints3d); 
    }
}

void EpipolarMatching::match()
{
    epipolarSimilarity(); 
    epipolarClustering(); 
    // compute3dDirectly(); 
    compute3dRANSAC(); 
}

void EpipolarMatching::compute3dRANSAC()
{
    m_skels3d.clear(); 
    for(int i = 0; i < m_clusters.size(); i++)
    {
        std::vector<Eigen::Vector3d> joints3d; 
        joints3d.resize(m_topo.joint_num, Eigen::Vector3d::Zero()); 
        for(int kptid = 0; kptid < m_topo.joint_num; kptid++)
        {
            std::vector<Camera> cams_visible; 
            vector<Eigen::Vector3d> joints2d; // [associated cams]
            for(int camid = 0; camid < m_cams.size(); camid++)
            {
                if(m_clusters[i][camid] < 0) continue; 
                int candid = m_clusters[i][camid]; 
                if(m_dets[camid][candid][kptid](2) < m_topo.kpt_conf_thresh[kptid]) continue; 
                Eigen::Vector3d p = m_dets[camid][candid][kptid]; 
                joints2d.push_back(p);
                cams_visible.push_back(m_cams[camid]);
            }
            if(cams_visible.size() < 2) continue; 
            double sigma2 = 30; 
            if(m_topo.label_names[kptid] == "center") {
                sigma2 = 200; 
                std::cout << "sigma 2 " << sigma2 << std::endl; 
            }
            joints3d[kptid] = triangulate_ransac(cams_visible, joints2d, 10, sigma2); 
        }
        m_skels3d.push_back(joints3d); 
    }
}

Eigen::Vector3d triangulate_ransac(
    const vector<Camera>& _cams, const vector<Eigen::Vector3d>& _xs,
    double sigma, double sigma2
    )
{
    // build init concensus
    // generate all init proposals: 3 points from different views 
    int nodeNum = _cams.size(); 
    vector<vector<int> > init;  // store viewid in graph 
    for(int i = 0; i < nodeNum; i++)
    {
        for(int j = i+1; j < nodeNum; j++)
        {
            for(int k = j+1; k < nodeNum; k++)
            {
                vector<int> a_init_prop; 
                a_init_prop.push_back(i); 
                a_init_prop.push_back(j); 
                a_init_prop.push_back(k); 
                init.push_back(a_init_prop);
            }
        }
    }
    // compute init 3d joints (model kernels)
    std::vector<ConcensusData> init_concensus; 
    for(int i = 0; i < init.size(); i++)
    {
        ConcensusData data; 
        vector<Camera> cams_visible; 
        vector<Eigen::Vector3d> joints2d; 
        for(int j = 0; j < init[i].size(); j++)
        {
            int camid   = init[i][j];
            cams_visible.push_back(_cams[camid]); 
            Eigen::Vector3d p  = _xs[camid]; 
            joints2d.push_back(p); 
        }
        Eigen::Vector3d X = triangulate_ceres(cams_visible, joints2d); 
        
        vector<Eigen::Vector3d> reprojs;
        double max_err = 0;  
        std::vector<double> errs; 
        for(int j = 0; j < init[i].size(); j++)
        {
            Eigen::Vector3d x_local  = project(cams_visible[j], X); 
            reprojs.push_back(x_local); 
            Eigen::Vector2d err_vec = x_local.segment<2>(0) - joints2d[j].segment<2>(0); 
            double err = err_vec.norm(); 
            if(err > max_err) max_err = err;  
            errs.push_back(err); 
        }
        if(max_err > sigma) {
            continue; 
        }
        data.X = X; 
        data.cams = cams_visible;
        data.joints2d = joints2d; 
        data.errs = errs; 
        data.metric = max_err; 
        data.ids = init[i]; 
        data.num = joints2d.size(); 
        init_concensus.push_back(data); 
    }
    // concensus 
    int iter_num = 0; 
    std::vector<ConcensusData> concensus_0 = init_concensus; 
    std::vector<ConcensusData> concensus_1; 
    for(;;)
    {
        int kernel_num = concensus_0.size(); 
        for(int kid = 0; kid < kernel_num; kid++)
        {
            //// expand concensus kernel 
            ConcensusData initdata = concensus_0[kid];
            ConcensusData data; 
            Eigen::Vector3d X = initdata.X; 

            std::vector<int> concensus_ids; 
            std::vector<Camera> concensus_cameras; 
            std::vector<Eigen::Vector3d> concensus_2ds; 
            for(int cid = 0; cid < _cams.size(); cid++)
            {
                Camera cam = _cams[cid]; 
                Eigen::Vector3d proj = project(cam, X); 
                double dist;
                dist = (proj.segment<2>(0) - _xs[cid].segment<2>(0)).norm(); 
                if(dist > sigma2 )continue; // this view has no nearby detection
                concensus_cameras.push_back(cam); 
                concensus_2ds.push_back(_xs[cid]); 
                concensus_ids.push_back(cid); 
            }

            Joint3DSolver solver; 
            solver.SetInit(X); 
            solver.SetParam(concensus_cameras, concensus_2ds); 
            solver.SetVerbose(false); 
            solver.Solve3D(); 
            Eigen::Vector3d con_3d = solver.GetX(); 
            std::vector<double> errs; 

            // check expansion is valid or not  
            double max_err = 0; 
            for(int c = 0; c < concensus_2ds.size(); c++)
            {
                Eigen::Vector3d proj = project(concensus_cameras[c], con_3d); 
                Eigen::Vector2d err_vec = proj.segment<2>(0) - concensus_2ds[c].segment<2>(0); 
                double err = err_vec.norm(); 
                if(max_err < err) max_err = err; 
                errs.push_back(err); 
            }
            if(max_err > sigma2) 
            {
                continue; 
            }
            // update concensus
            data.X = con_3d; 
            data.joints2d = concensus_2ds;
            data.ids = concensus_ids; 
            data.errs = errs; 
            data.num = concensus_2ds.size(); 
            data.metric = max_err; 
            data.cams = concensus_cameras; 
            concensus_1.push_back(data); 
        }
        // clean repeated ones 
        std::sort(concensus_1.begin(), concensus_1.end(), compare_concensus); 
        std::vector<ConcensusData> concensus_tmp;
        for(int i = 0; i < concensus_1.size(); i++)
        {
            if(i==0) concensus_tmp.push_back(concensus_1[i]); 
            else
            {
                bool is_repeat = equal_concensus(concensus_1[i], concensus_1[i-1]);

                if(is_repeat) continue; 
                else concensus_tmp.push_back(concensus_1[i]); 
            }
        }
        concensus_1 = concensus_tmp; 

        // check if converge? 
        bool is_converge = equal_concensus_list(concensus_0, concensus_1); 
        if(is_converge) break; 
        else {
            concensus_0 = concensus_1;
            concensus_1.clear(); 
            iter_num ++; 
        }
        if(iter_num > 10)
        {
            std::cout << RED_TEXT("Warning: too many iters for ransac") << std::endl; 
        }
    }
    // std::cout << "final concensus num: " << concensus_1.size() << std::endl; 
    if(concensus_1.size() == 0) return Eigen::Vector3d::Zero(); 
    return concensus_1[0].X; 
}