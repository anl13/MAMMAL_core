#include "framedata.h" 
#include "colorterminal.h" 

/*
pre-operation: epopolarSimilarity; 

*/
void FrameData::ransacProposals()
{
    m_proposals.clear(); // store final 3d proposals 
    m_proposal_errs.clear(); 
    m_proposal_concensus_num.clear(); 
    double sigma = 10; // 10 pixel ransac error metric(reprojection error) 
    double gamma = m_epi_thres; // epipolar distance threshold, default 25 pixels. 
    std::cout << "m_epi_thre " << m_epi_thres << std::endl;  
    
    for(int kpt_id = 0; kpt_id < m_kptNum; kpt_id++)
    {
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
        std::vector<Eigen::Vector3d> kernels; // ransac kernels 
        std::vector<double> kernel_metrics;  // ransac error (max 2d projection err)
        for(int i = 0; i < init.size(); i++)
        {
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
            for(int j = 0; j < init[i].size(); j++)
            {
                Eigen::Vector3d x_local  = project(cams_visible[j], X); 
                reprojs.push_back(x_local); 
                Eigen::Vector2d err_vec = x_local.segment<2>(0) - joints2d[j].segment<2>(0); 
                double err = err_vec.norm(); 
                // std::cout << YELLOW_TEXT("err: ") << err.norm() << std::endl;
                if(err > max_err) max_err = err;  
            }
            if(max_err > sigma) continue; 
            kernel_metrics.push_back(max_err); 
            kernels.push_back(X); 
        }

        // concensus 
        std::vector<std::vector<int> > concensus_all; 
        std::vector<std::vector<Camera> > concensus_cameras_all; 
        std::vector<std::vector<Eigen::Vector3d> > concensus_2ds_all; 
        std::vector<Eigen::Vector3d>  concensus_kernels; 
        std::vector<double>  concensus_metric; 
        std::vector<int>     concensus_num; 
        int kernel_num = kernels.size(); 
        for(int kid = 0; kid < kernel_num; kid++)
        {
            std::vector<int> concensus_ids; 
            Eigen::Vector3d X = kernels[kid];
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
                if(min_dist > sigma )continue; // this view has no nearby detection
                concensus_cameras.push_back(cam); 
                concensus_2ds.push_back(points[cid][min_id]); 
                int node_id = invTable[cid][min_id]; 
                concensus_ids.push_back(node_id); 
            }
            concensus_2ds_all.push_back(concensus_2ds); 
            concensus_cameras_all.push_back(concensus_cameras); 
            concensus_all.push_back(concensus_ids); 
            Eigen::Vector3d con_3d = triangulate_ceres(concensus_cameras, concensus_2ds); 
            concensus_kernels.push_back(con_3d); 

            double max_err = 0; 
            for(int c = 0; c < concensus_2ds.size(); c++)
            {
                Eigen::Vector3d proj = project(concensus_cameras[c], con_3d); 
                Eigen::Vector2d err_vec = proj.segment<2>(0) - concensus_2ds[c].segment<2>(0); 
                double err = err_vec.norm(); 
                if(max_err < err) max_err = err; 
            }
            concensus_metric.push_back(max_err); 
            concensus_num.push_back(concensus_2ds.size()); 
        }

        std::vector<Eigen::Vector3d> empty_vec; 
        std::vector<double> empty_double; 
        std::vector<int> empty_int; 
        if(kpt_id == 18)
        {
            m_proposals.push_back(concensus_kernels); 
            m_proposal_errs.push_back(concensus_metric); 
            m_proposal_concensus_num.push_back(concensus_num); 
        }
        else 
        {
            m_proposals.push_back(empty_vec); 
            m_proposal_errs.push_back(empty_double); 
            m_proposal_concensus_num.push_back(empty_int); 
        }
    }

}


void FrameData::projectProposals()
{

}