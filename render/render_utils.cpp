#include "render_utils.h" 

void GetBallsAndSticks(
    const PIG_SKEL& joints, 
    const std::vector<Eigen::Vector2i>& bones, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks 
    )
{
    balls.clear(); 
    sticks.clear(); 
    for(int i = 0; i < joints.cols(); i++)
    {
        Eigen::Vector4d p = joints.col(i); 
        if(p(3) > 0) // joint is valid 
        {
            Eigen::Vector3f pf = p.segment<3>(0).cast<float>();
            pf(0) = pf(0); 
            pf(1) = pf(1); 
            pf(2) = pf(2);
            balls.push_back(pf);
        }
    }
    
    for(int i = 0; i < bones.size(); i++)
    {
        int sid = bones[i](0);
        int eid = bones[i](1); 
        Eigen::Vector4d ps = joints.col(sid); 
        Eigen::Vector4d pe = joints.col(eid); 

        if(ps(3) > 0 && pe(3) > 0)
        {
            std::pair<Eigen::Vector3f, Eigen::Vector3f> stick = 
            {
                ps.segment<3>(0).cast<float>(), 
                pe.segment<3>(0).cast<float>()
            }; 
            sticks.push_back(stick); 
        }
    }
}

void GetBallsAndSticks(
    const Eigen::Matrix3Xf& joints, 
    const Eigen::VectorXi& parents, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f> > &sticks
)
{
    balls.clear(); 
    sticks.clear(); 
    for(int i = 0; i < joints.cols(); i++)
    {
        Eigen::Vector3f p = joints.col(i); 
        balls.push_back(p); 
    }

    for(int i = 0; i < parents.size(); i++)
    {
        if(parents(i) == -1) continue; 
        int parent = parents(i); 
        Eigen::Vector3f s = joints.col(parent); 
        Eigen::Vector3f e = joints.col(i); 
        std::pair<Eigen::Vector3f, Eigen::Vector3f> stick = {s,e}; 
        sticks.push_back(stick); 
    }
}


std::vector<Eigen::Vector3f> getColorMapEigen(std::string cm_type, bool reorder)
{
    std::vector<Eigen::Vector3i> cm; 
    getColorMap(cm_type, cm); 
    std::vector<Eigen::Vector3f> cm_float;
    cm_float.resize(cm.size()); 
    for(int i = 0; i < cm.size(); i++)
    {
        Eigen::Vector3f c;
        if(reorder)
        {
            c(0) = float(cm[i](2)); 
            c(1) = float(cm[i](1)); 
            c(2) = float(cm[i](0));
        }
        else
        {
            c(0) = float(cm[i](0)); 
            c(1) = float(cm[i](1)); 
            c(2) = float(cm[i](2));
        }
        c = c / 255.0f; 
        cm_float[i] = c;  
    }
    return cm_float; 
}

void GetBalls(
    const vector<vector<ConcensusData> > & data, 
    const vector<int>& m_kpt_color_id, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<float> & sizes, 
    std::vector<int>& color_ids
)
{
    balls.clear(); 
    color_ids.clear(); 
    sizes.clear(); 
    double basic_size = 0.005f; 
    for(int kpt_id = 0; kpt_id < data.size(); kpt_id++)
    {
        for(int p = 0; p < data[kpt_id].size(); p++)
        {
            ConcensusData d = data[kpt_id][p];
            Vec3 point = d.X; 
            balls.push_back(point.cast<float>()); 
            color_ids.push_back(m_kpt_color_id[kpt_id]); 
            // std::cout << "metric proposal " << p << " : " << metric[kpt_id][p] << std::endl; 
            float s = (float)(std::min( 0.02/d.metric, basic_size) ); 
            float ratio = d.num / 5.0f;
            sizes.push_back(s * ratio); 
        }
    }
}