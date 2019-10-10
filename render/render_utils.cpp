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


std::vector<Eigen::Vector3f> getColorMapEigen()
{
    std::vector<Eigen::Vector3i> cm; 
    getColorMap("anliang", cm); 
    std::vector<Eigen::Vector3f> cm_float;
    cm_float.resize(cm.size()); 
    for(int i = 0; i < cm.size(); i++)
    {
        Eigen::Vector3f c;
        c(0) = float(cm[i](2)); 
        c(1) = float(cm[i](1)); 
        c(2) = float(cm[i](0));
        c = c / 255.0f; 
        cm_float[i] = c;  
    }
    return cm_float; 
}