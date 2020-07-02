#include "render_utils.h" 

void GetBallsAndSticks(
    const vector<Eigen::Vector3d>& joints, 
    const std::vector<Eigen::Vector2i>& bones, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks 
    )
{
    balls.clear(); 
    sticks.clear(); 
    for(int i = 0; i < joints.size(); i++)
    {
        Eigen::Vector3d p = joints[i]; 
        if(p.norm() > 0) // joint is valid 
        {
            Eigen::Vector3f pf = p.cast<float>();
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
        Eigen::Vector3d ps = joints[sid]; 
        Eigen::Vector3d pe = joints[eid]; 

        if(ps.norm() > 0 && pe.norm() > 0)
        {
            std::pair<Eigen::Vector3f, Eigen::Vector3f> stick = 
            {
                ps.cast<float>(), 
                pe.cast<float>()
            }; 
            sticks.push_back(stick); 
        }
    }
}

void GetBallsAndSticks(
	const Eigen::MatrixXd& joints,
	const std::vector<Eigen::Vector2i>& bones,
	std::vector<Eigen::Vector3f>& balls,
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks
)
{
	balls.clear();
	sticks.clear();
	for (int i = 0; i < joints.cols(); i++)
	{
		Eigen::Vector3d p = joints.col(i);
		if (p.norm() > 0) // joint is valid 
		{
			Eigen::Vector3f pf = p.cast<float>();
			balls.push_back(pf);
		}
	}

	for (int i = 0; i < bones.size(); i++)
	{
		int sid = bones[i](0);
		int eid = bones[i](1);
		Eigen::Vector3d ps = joints.col(sid);
		Eigen::Vector3d pe = joints.col(eid);

		if (ps.norm() > 0 && pe.norm() > 0)
		{
			std::pair<Eigen::Vector3f, Eigen::Vector3f> stick =
			{
				ps.cast<float>(),
				pe.cast<float>()
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