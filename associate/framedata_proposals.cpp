#include "framedata.h" 
#include "colorterminal.h" 

/*
pre-operation: epopolarSimilarity; 
*/

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

void FrameData::ransacProposals()
{
#define DEBUG_RANSAC
    m_concensus.clear(); 
    double sigma = 10 ; // 10 pixel ransac error metric(reprojection error) 
    
    double gamma = m_epi_thres; // epipolar distance threshold, default 25 pixels. 
    double delta = 0.01; //non-maxima suppression threshold
    std::cout << "m_epi_thre " << m_epi_thres << std::endl;  
    
    for(int kpt_id = 0; kpt_id < m_kptNum; kpt_id++)
    {
        if(kpt_id != 18) 
        {
            std::vector<ConcensusData> data; 
            m_concensus.push_back(data); 
            continue; 
        }
        double sigma2 = m_keypoint_proposal_thres[kpt_id]; 
        // std::cout << RED_TEXT("kpt_id: ") << kpt_id << std::endl; 
        auto  table = m_tables[kpt_id];
        auto  invTable    = m_invTables[kpt_id]; 
        const Eigen::MatrixXd G                  = m_G[kpt_id]; 
        const vector<vector<Vec3> > points       = m_points[kpt_id];  // camnum, candnum

        const int nodeNum = table.size(); // graph node num 

        // generate all init proposals: 3 points from different views 
        vector<vector<int> > init;  // store node id in graph 
        for(int i = 0; i < nodeNum; i++)
        {
            for(int j = i+1; j < nodeNum; j++)
            {
                if(G(i,j) < 0) continue; 
                if(G(i,j) > gamma) continue;
                for(int k = j+1; k < nodeNum; k++)
                {
                    if(G(i,k) < 0 || G(j,k) < 0) continue; 
                    if(G(i,k) > gamma || G(j,k) > gamma) continue; 
                    vector<int> a_init_prop; 
                    a_init_prop.push_back(i); 
                    a_init_prop.push_back(j); 
                    a_init_prop.push_back(k); 
                    init.push_back(a_init_prop); 
                }
            }
        }

        // std::cout << "kpt " << kpt_id << " init " << init.size() << std::endl; 
        // compute init 3d joints (model kernels)
        std::vector<ConcensusData> init_concensus; 
        for(int i = 0; i < init.size(); i++)
        {
            ConcensusData data; 
            vector<Camera> cams_visible; 
            vector<Eigen::Vector3d> joints2d; 
            for(int j = 0; j < init[i].size(); j++)
            {
                int pid   = init[i][j];
                int camid = table[pid].first; 
                int cand  = table[pid].second; 
                Camera cam = m_camsUndist[camid]; 
                cams_visible.push_back(m_camsUndist[camid]); 
                Eigen::Vector3d p  = m_dets_undist[camid][kpt_id][cand]; 
                joints2d.push_back(p); 
            }
            Joint3DSolver solver; 
            solver.SetParam(cams_visible, joints2d); 
            solver.SetVerbose(false); 
            solver.Solve3D(); 

            Eigen::Vector3d X = solver.GetX(); 
            
            vector<Eigen::Vector3d> reprojs;
            double max_err = 0;  
            std::vector<double> errs; 
            for(int j = 0; j < init[i].size(); j++)
            {
                Eigen::Vector3d x_local  = project(cams_visible[j], X); 
                reprojs.push_back(x_local); 
                Eigen::Vector2d err_vec = x_local.segment<2>(0) - joints2d[j].segment<2>(0); 
                double err = err_vec.norm(); 
                // std::cout << YELLOW_TEXT("err: ") << err.norm() << std::endl;
                if(err > max_err) max_err = err;  
                errs.push_back(err); 
            }
            if(max_err > sigma) continue; 
            data.X = X; 
            data.cams = cams_visible;
            data.joints2d = joints2d; 
            data.errs = errs; 
            data.metric = max_err; 
            data.ids = init[i]; 
            data.num = joints2d.size(); 
            init_concensus.push_back(data); 
        }

#ifdef DEBUG_RANSAC
        std::cout << YELLOW_TEXT("init concensus num: ") << init_concensus.size() << std::endl; 
#endif 
        // concensus 
        int iter_num = 0; 
        std::vector<ConcensusData> concensus_0 = init_concensus; 
        std::vector<ConcensusData> concensus_1; 
        for(;;)
        {
#ifdef DEBUG_RANSAC
            std::cout << RED_TEXT("current iter: ") << iter_num << std::endl; 
#endif 

            int kernel_num = concensus_0.size(); 
#ifdef DEBUG_RANSAC
            std::cout << "kernel num : " << kernel_num << std::endl; 
#endif 
            for(int kid = 0; kid < kernel_num; kid++)
            {
                ConcensusData initdata = concensus_0[kid];
                ConcensusData data; 
                Eigen::Vector3d X = initdata.X; 

                std::vector<int> concensus_ids; 
                std::vector<Camera> concensus_cameras; 
                std::vector<Eigen::Vector3d> concensus_2ds; 
                for(int cid = 0; cid < m_camNum; cid++)
                {
                    Camera cam = m_camsUndist[cid]; 
                    Eigen::Vector3d proj = project(cam, X); 
                    std::vector<double> dists; 
                    int cand_num = points[cid].size(); 
                    if(cand_num == 0) continue; // this view has no valid detection 
                    dists.resize(cand_num); 
                    for(int j = 0; j < cand_num; j++)
                    {
                        Eigen::Vector2d err_vec = proj.segment<2>(0) - points[cid][j].segment<2>(0); 
                        dists[j] = err_vec.norm(); 
                    }
                    int min_id = 0; 
                    double min_dist = dists[0]; 
                    for(int j = 0; j < cand_num; j++)
                    {
                        if(dists[j] < min_dist)
                        {
                            min_dist = dists[j]; 
                            min_id = j; 
                        }
                    }
                    if(min_dist > sigma2 )continue; // this view has no nearby detection
                    concensus_cameras.push_back(cam); 
                    concensus_2ds.push_back(points[cid][min_id]); 
                    int node_id = invTable[cid][min_id]; 
                    concensus_ids.push_back(node_id); 
                }

                Joint3DSolver solver; 
                solver.SetInit(X); 
                solver.SetParam(concensus_cameras, concensus_2ds); 
                solver.SetVerbose(false); 
                solver.Solve3D(); 
                Eigen::Vector3d con_3d = solver.GetX(); 
                std::vector<double> errs; 

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
                    std::cout << "max err: " << max_err << std::endl; 
                    continue; 
                }

                data.X = con_3d; 
                data.joints2d = concensus_2ds;
                data.ids = concensus_ids; 
                data.errs = errs; 
                data.num = concensus_2ds.size(); 
                data.metric = max_err; 
                data.cams = concensus_cameras; 

                concensus_1.push_back(data); 
            }

#ifdef DEBUG_RANSAC
            //             
            std::cout << YELLOW_TEXT("before clean ") << concensus_1.size() << std::endl; 
            for(int i = 0; i < concensus_1.size(); i++)
            {
                ConcensusData data = concensus_1[i];
                std::cout << "concensus " << i << " : "; 
                for(int j = 0; j < data.num; j++)
                {
                    std::cout << data.ids[j] << " ";
                }
                std::cout << "metric: " << data.metric << std::endl; 
            }
#endif 
            // clean repeated ones 
            
            std::sort(concensus_1.begin(), concensus_1.end(), compare_concensus); 
            std::vector<ConcensusData> concensus_tmp;
            for(int i = 0; i < concensus_1.size(); i++)
            {
                if(i==0) concensus_tmp.push_back(concensus_1[i]); 
                else
                {
                    bool is_repeat = equal_concensus(concensus_1[i], concensus_1[i-1]);
                    std::cout << is_repeat << " ";
                    if(is_repeat) continue; 
                    else concensus_tmp.push_back(concensus_1[i]); 
                }
            }
            std::cout << std::endl; 
            concensus_1 =  concensus_tmp; 

#ifdef DEBUG_RANSAC
            std::cout << YELLOW_TEXT("after clean ") << concensus_1.size() << std::endl;  
            // visualize concensus_1 
            for(int i = 0; i < concensus_1.size(); i++)
            {
                ConcensusData data = concensus_1[i];
                std::cout << "concensus " << i << " : "; 
                for(int j = 0; j < data.num; j++)
                {
                    std::cout << data.ids[j] << " ";
                }
                std::cout << "metric: " << data.metric << std::endl; 
            }
#endif 

            // check if converge? 
            bool is_converge = equal_concensus_list(concensus_0, concensus_1); 
            if(is_converge) break; 
            else {
                concensus_0 = concensus_1;
                concensus_1.clear(); 
                iter_num ++; 
            }
        }
#ifdef DEBUG_RANSAC
        std::cout << "converge in " << iter_num << " iterations" << std::endl; 
#endif 
        // non-maxima suppression 
        int c_num = concensus_1.size(); 
        Eigen::MatrixXd E = Eigen::MatrixXd::Zero(c_num, c_num); 
        for(int i = 0; i < c_num; i++)
        {
            for(int j = 0; j < c_num; j++)
            {
                if(i == j) E(i,j) = -1; 
                else
                {
                    double e = (concensus_1[i].X - concensus_1[j].X).norm(); 
                    E(i,j) = e; 
                }
            }
        }
        std::cout << "Euclidean Dist: " << std::endl << E << std::endl; 
        m_concensus.push_back(concensus_1); 

    }
}


void FrameData::projectProposals()
{
    std::vector<Eigen::Vector3i> CM2; 
    getColorMap("jet", CM2); 
    std::vector<cv::Mat> imgs;
    cloneImgs(m_imgsUndist, imgs); 
    for(int kpt_id = 18; kpt_id < 19; kpt_id++)
    {
        auto concensus = m_concensus[kpt_id]; 
        for(int i = 0; i < concensus.size(); i++)
        {
            for(int cid = 0; cid < concensus[i].ids.size(); cid++)
            {
                int pid = concensus[i].ids[cid];
                int camid = m_tables[kpt_id][pid].first;
                Eigen::Vector3d p2d = project(concensus[i].cams[cid], concensus[i].X); 
                my_draw_point(imgs[camid], p2d, m_CM[i], 10); 
            }
        }
    }

    cv::Mat output; 
    packImgBlock(imgs, output); 
    cv::namedWindow("concensus", cv::WINDOW_NORMAL); 
    cv::imshow("concensus", output); 
    int key = cv::waitKey(); 
}