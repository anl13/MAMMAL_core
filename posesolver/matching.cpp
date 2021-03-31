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
        const vector<Eigen::Vector3f>& pig1, const vector<Eigen::Vector3f>& pig2,
        double& avg_loss, int& matched_num)
{
    float (*distfunc)(const Camera&, const Camera&, const Eigen::Vector3f&, const Eigen::Vector3f&); 
    double dist_thres = 0; 
    if(m_epi_type == "p2l")
    {
        distfunc = &getEpipolarDist; 
        dist_thres = 50; 
    }
    else if(m_epi_type == "l2l") 
    {
        distfunc = &getEpipolarDistL2L;
        dist_thres = 0.02;
    }
    else 
    {
        std::cout << "Epipolar Type " << m_epi_type << " not implemented yet" << std::endl; 
		exit(-1); 
    }
    double total_loss = 0; 
    int valid = 0; 
    matched_num = 0; 
    for(int i = 0; i < m_topo.joint_num; i++)
    {
        Eigen::Vector3f v1 = pig1[i];
        Eigen::Vector3f v2 = pig2[i];
        if(v1(2) < m_topo.kpt_conf_thresh[i] || v2(2) < m_topo.kpt_conf_thresh[i])
        {
            continue; 
        }
        float epi_dist = distfunc(cam1, cam2, v1, v2); 
		if (i == 20)
		{
			total_loss += epi_dist * 1; 
			valid += 1; 
		}
		else
		{
			total_loss += epi_dist;
			valid += 1;
		}
        if(epi_dist < dist_thres) 
        {matched_num +=1;}

    }
    if(valid == 0) 
    {
        matched_num = -1;
        /* 
        20191225: AN Liang
        we can't tell whether these two belong to the same pig. 
        so only add a suitable penalty, not cut the edges. 
        */ 
        avg_loss = 500;  
    }
    else
    { 
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
            epipolarWholeBody(m_cams[camid1], m_cams[camid2], 
                m_dets[camid1][candid1].keypoints, m_dets[camid2][candid2].keypoints,
                avg_loss, matched_num); 
#ifdef DEBUG_MATCH
            std::cout << "matched num: " << matched_num << std::endl; 
#endif 
            double penalty_base = 0; 
            if(matched_num == 0) 
            {
                m_G(index_i, index_j) = -1; 
            }
            else 
            {
                if(matched_num < 0) penalty_base = 1; 
                else penalty_base = matched_num;  
                m_G(index_i, index_j) = avg_loss / pow(penalty_base, 1.0); 
            }
        }
    }
}

void EpipolarMatching::epipolarClustering()
{
    ClusterClique CC; 
    CC.G = m_G.cast<double>(); 
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

void EpipolarMatching::match()
{
    epipolarSimilarity(); 
    epipolarClustering(); 
    compute3dRANSAC(); 
#ifdef DEBUG_MATCH
    std::cout << "cliques: " << m_cliques.size() << std::endl;
#endif 
}

void EpipolarMatching::compute3dRANSAC()
{
    m_skels3d.clear(); 
    for(int i = 0; i < m_clusters.size(); i++)
    {
        std::vector<Eigen::Vector3f> joints3d; 
        joints3d.resize(m_topo.joint_num, Eigen::Vector3f::Zero()); 
        for(int kptid = 0; kptid < m_topo.joint_num; kptid++)
        {
            std::vector<Camera> cams_visible; 
            vector<Eigen::Vector3f> joints2d; // [associated cams]
            for(int camid = 0; camid < m_cams.size(); camid++)
            {
                if(m_clusters[i][camid] < 0) continue; 
                int candid = m_clusters[i][camid]; 
                if(m_dets[camid][candid].keypoints[kptid](2) < m_topo.kpt_conf_thresh[kptid]) continue; 
                Eigen::Vector3f p = m_dets[camid][candid].keypoints[kptid]; 
                joints2d.push_back(p);
                cams_visible.push_back(m_cams[camid]);
            }
            if(cams_visible.size() < 2) continue; 
            if(cams_visible.size() >= 5)
            {
                joints3d[kptid] = triangulate_ceres(cams_visible, joints2d).cast<float>(); 
            }
            else 
            {
                if(m_topo.label_names[kptid] == "center")
                { 
                    joints3d[kptid] = triangulate_ransac(cams_visible, joints2d, 45, 90);
                }
                else
                {
                    joints3d[kptid] = triangulate_ransac(cams_visible, joints2d, 30, 60);
                } 
            }
            double max_err = 0;  
            double mean_err = 0; 
            for(int j = 0; j < joints2d.size(); j++)
            {
                Eigen::Vector3f x_local  = project(cams_visible[j], joints3d[kptid]); 
                Eigen::Vector2f err_vec = x_local.segment<2>(0) - joints2d[j].segment<2>(0); 
                double err = err_vec.norm(); 
                if(err > max_err) max_err = err;  
                mean_err += err; 
            }
            mean_err /= joints2d.size(); 
#ifdef DEBUG_MATCH
            std::cout << "id: " << i << ", kpt " << kptid << " : " << max_err << "; " << mean_err << std::endl; 
#endif 
        }
        m_skels3d.push_back(joints3d); 
    }
}

Eigen::Vector3f triangulate_ransac(
    const vector<Camera>& _cams, const vector<Eigen::Vector3f>& _xs, double sigma1, double sigma2)
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
        vector<Eigen::Vector3f> joints2d; 
        for(int j = 0; j < init[i].size(); j++)
        {
            int camid   = init[i][j];
            cams_visible.push_back(_cams[camid]); 
            Eigen::Vector3f p  = _xs[camid]; 
            joints2d.push_back(p); 
        }
        Eigen::Vector3f X = triangulate_ceres(cams_visible, joints2d).cast<float>(); 
        
        vector<Eigen::Vector3f> reprojs;
        float max_err = 0;
        float mean_err = 0; 
        std::vector<float> errs; 
        for(int j = 0; j < init[i].size(); j++)
        {
            Eigen::Vector3f x_local  = project(cams_visible[j], X); 
            reprojs.push_back(x_local); 
            Eigen::Vector2f err_vec = x_local.segment<2>(0) - joints2d[j].segment<2>(0); 
            float err = err_vec.norm(); 
            if(err > max_err) max_err = err;  
            mean_err += err; 
            errs.push_back(err); 
        }
        mean_err /= errs.size(); 
        if(max_err > sigma1 * 2 || mean_err > sigma1) {
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
    // concensus iteratively. 
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
            Eigen::Vector3f X = initdata.X; 

            std::vector<int> concensus_ids; 
            std::vector<Camera> concensus_cameras; 
            std::vector<Eigen::Vector3f> concensus_2ds; 
            for(int cid = 0; cid < _cams.size(); cid++)
            {
                Camera cam = _cams[cid]; 
                Eigen::Vector3f proj = project(cam, X); 
                float dist;
                dist = (proj.segment<2>(0) - _xs[cid].segment<2>(0)).norm(); 
                if(dist > sigma2 )continue; // this view has no nearby detection
                concensus_cameras.push_back(cam); 
                concensus_2ds.push_back(_xs[cid]); 
                concensus_ids.push_back(cid); 
            }

            Joint3DSolver solver; 
            solver.SetInit(X.cast<double>()); 
			std::vector<Eigen::Vector3d> con2ds_d(concensus_2ds.size()); 
			for (int k = 0; k < concensus_2ds.size(); k++)con2ds_d[k] = concensus_2ds[k].cast<double>(); 
            solver.SetParam(concensus_cameras, con2ds_d); 
            solver.SetVerbose(false); 
            solver.Solve3D(); 
            Eigen::Vector3d con_3d = solver.GetX(); 
			Eigen::Vector3f con_3f = con_3d.cast<float>(); 
            std::vector<float> errs; 

            // check expansion is valid or not  
            double max_err = 0; 
            double mean_err = 0; 
            for(int c = 0; c < concensus_2ds.size(); c++)
            {
                Eigen::Vector3f proj = project(concensus_cameras[c], con_3f); 
                Eigen::Vector2f err_vec = proj.segment<2>(0) - concensus_2ds[c].segment<2>(0); 
                double err = err_vec.norm(); 
                if(max_err < err) max_err = err; 
                errs.push_back(err);
                mean_err += err;  
            }
            mean_err /= errs.size();
            if(max_err > sigma2 * 2 || mean_err > sigma2) 
            {
                continue; 
            }
            // update concensus
            data.X = con_3f; 
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
    if(concensus_1.size() == 0) return Eigen::Vector3f::Zero(); 
    return concensus_1[0].X; 
}

void EpipolarMatching::truncate(int _clusternum)
{
    assert(_clusternum>=0); 
    vector<vector<int> > trunc_cliques; 
    vector<vector<int> > trunc_clusters; 
    vector<vector<Eigen::Vector3f> > trunc_skels; 
    for(int i = 0; i < _clusternum && i < m_skels3d.size(); i++)
    {
        trunc_cliques.push_back(m_cliques[i]);
        trunc_clusters.push_back(m_clusters[i]); 
        trunc_skels.push_back(m_skels3d[i]); 
    }
    m_clusters = trunc_clusters;
    m_cliques = trunc_cliques;
    m_skels3d = trunc_skels; 
}

void EpipolarMatching::match_by_tracking()
{
	// init clusters 
    int cluster_num = m_pignum; 
    int camNum = m_cams.size(); 
    assert(m_skels_t_1.size() == cluster_num); 
	project_all();

    // create similarity matrix m_G
	trackingSimilarity();
	// solve m_G
	trackingClustering(); 
}

void EpipolarMatching::project_all()
{
	int camNum = m_cams.size(); 
	int pointNum = m_topo.joint_num;
	m_skels_proj_t_1.resize(camNum);
	for (int camid = 0; camid < camNum; camid++)
	{
		m_skels_proj_t_1[camid].resize(m_pignum); 
		for (int j = 0; j < m_pignum; j++)
		{
			m_skels_proj_t_1[camid][j].resize(pointNum);
			for (int jid = 0; jid < pointNum; jid++)
			{
				Eigen::Vector3f j3d = m_skels_t_1[j][jid];
				if (j3d.norm() > 0) m_skels_proj_t_1[camid][j][jid] = project(m_cams[camid], j3d);
				else
				{
					m_skels_proj_t_1[camid][j][jid] = Eigen::Vector3f::Zero();
				}
			}
		}
	}
}

void EpipolarMatching::projectLoss(
	int pig_id, 
    int camid, 
	const DetInstance& det, 
	double& mean_err, double& matched_num)
{
    mean_err = 0; 
    double validnum = 0; 
    double dist_thres = 100; 
    matched_num = 0;
    for(int kptid = 0; kptid < m_topo.joint_num; kptid++)
    {
        Eigen::Vector3f j2d = det.keypoints[kptid];
        if(j2d(2) >= m_topo.kpt_conf_thresh[kptid]) continue; 
        Eigen::Vector3f j_proj = m_skels_proj_t_1[camid][pig_id][kptid]; 
        if(j_proj.norm() > 0) 
        {
            float dist = (j_proj.segment<2>(0) - j2d.segment<2>(0)).norm();
            mean_err += dist; 
            validnum += 1; 
            if(dist < dist_thres) matched_num += 1; 
        }
    }

    if(validnum == 0)
    {
        mean_err = -1; 
        matched_num = 0; 
    }
    else 
    {
        mean_err /= validnum;
    }
}


void EpipolarMatching::trackingSimilarity()
{
    // build table and invtable 
    m_total_detection_num = 0; 
    m_table.clear(); 
    m_inv_table.clear(); 
    int cam_num = m_cams.size(); 
    m_inv_table.resize(cam_num+1); 
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
	for (int i = 0; i < m_pignum; i++)
	{
		m_inv_table[cam_num].push_back(m_total_detection_num + i);
		m_table.push_back({ cam_num, i });
	}

    // construct graph 
	int graph_size = m_total_detection_num + m_pignum; // m_pignum is cluster num 
    m_G.resize(graph_size, graph_size); 
    m_G.setZero(); 
    
	// matching block 
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
            epipolarWholeBody(m_cams[camid1], m_cams[camid2], 
                m_dets[camid1][candid1].keypoints, m_dets[camid2][candid2].keypoints,
                avg_loss, matched_num); 
            m_G(index_i, index_j) = avg_loss / pow(matched_num, 1.0); 
        }
    }

	// tracking block 
	for (int i = 0; i < m_pignum; i++)
	{
		for (int index_i = 0; index_i < m_total_detection_num; index_i++)
		{
			int camid = m_table[index_i].first; 
			int candid = m_table[index_i].second; 
			double mean_err, matched_num;
			projectLoss(i, camid, m_dets[camid][candid], mean_err, matched_num);
			m_G(index_i, i+m_total_detection_num) = mean_err / 25; 
			m_G(i+m_total_detection_num, index_i) = mean_err / 25; 
			//std::cout << "proj err: " << i << ": " << index_i << ": " << mean_err << std::endl;
		}
	}
	//std::cout << "threshold: " << m_epi_thres << std::endl; 

	for (int i = 0; i < m_pignum; i++)
	{
		for (int j = 0; j < m_pignum; j++)
		{
			m_G(i + m_total_detection_num, j + m_total_detection_num) = -1; 
			m_G(j + m_total_detection_num, i + m_total_detection_num) = -1;
		}
	}
}

void EpipolarMatching::trackingClustering()
{
	ClusterClique CC;
	CC.G = m_G.cast<double>();
	CC.table = m_table;
	CC.invTable = m_inv_table;
	CC.vertexNum = m_total_detection_num + m_pignum;
	CC.threshold = m_epi_thres;
	CC.constructAdjacentGraph();
	CC.enumerateBestCliques();
	CC.printGraph();  
	// CC.printAllMC();
	// CC.printCliques(); 
	m_cliques = CC.cliques;
	int cluster_num = m_pignum; 
	std::vector<std::vector<int> > m_cliques_resort;
	m_cliques_resort.resize(m_pignum); 

	for (int cluster_id = 0; cluster_id < m_pignum; cluster_id++)
	{
		for (int i = 0; i < m_cliques.size(); i++)
		{
			if (in_list(cluster_id + m_total_detection_num, m_cliques[i]))
			{
				m_cliques_resort[cluster_id] = m_cliques[i];
			}
		}
	}
	m_clusters.resize(m_pignum);
	for (int i = 0; i < m_pignum; i++)
	{
		m_clusters[i].resize(m_cams.size(), -1);
		for (int j = 0; j < m_cliques_resort[i].size(); j++)
		{
			int node_id = m_cliques_resort[i][j];
			if (node_id >= m_total_detection_num) continue;
			int camid = m_table[node_id].first;
			int candid = m_table[node_id].second;
			m_clusters[i][camid] = candid;
		}
	}
}