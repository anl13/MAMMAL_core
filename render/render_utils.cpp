#include "render_utils.h" 

void GetBallsAndSticks(
    const vector<Eigen::Vector3f>& joints, 
    const std::vector<Eigen::Vector2i>& bones, 
    std::vector<Eigen::Vector3f>& balls, 
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks 
    )
{
    balls.clear(); 
    sticks.clear(); 
    for(int i = 0; i < joints.size(); i++)
    {
        Eigen::Vector3f p = joints[i]; 
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
        Eigen::Vector3f ps = joints[sid]; 
        Eigen::Vector3f pe = joints[eid]; 

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
	const vector<Eigen::Vector3f>& joints,
	const std::vector<int>& parents,
	std::vector<Eigen::Vector3f>& balls,
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& sticks
)
{
	balls.clear();
	sticks.clear();
	balls = joints; 

	for (int i = 0; i < parents.size(); i++)
	{
		if (parents[i] == -1) continue;
		int parent = parents[i];
		Eigen::Vector3f s = joints[parent];
		Eigen::Vector3f e = joints[i];
		std::pair<Eigen::Vector3f, Eigen::Vector3f> stick = { s,e };
		sticks.push_back(stick);
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

void readObjectWithColor(
	std::string filename,
	std::vector<Eigen::Vector3f>& vertices,
	std::vector<Eigen::Vector3f>& color,
	std::vector<Eigen::Vector3u>& faces
)
{
	std::ifstream reader(filename); 
	if (!reader.is_open())
	{
		std::cout << "[Mesh::Load] can not open the file" << std::endl;
		return;
	}

	vertices.clear(); 
	color.clear(); 
	faces.clear(); 
	float v1, v2, v3;
	float c1, c2, c3; 
	int v1_index, v2_index, v3_index; 
	char ch;
	while (!reader.eof())
	{
		std::string tempstr;
		reader >> tempstr;
		if (tempstr == "v")
		{
			reader >> v1 >> v2 >> v3;
			reader >> c1 >> c2 >> c3;
			Eigen::Vector3f temp_v((float)v1, (float)v2, (float)v3);
			Eigen::Vector3f color_v(c1 / 255, c2 / 255, c3 / 255);
			vertices.push_back(temp_v);
			color.push_back(color_v);
		}
		else if (tempstr == "f")
		{
			reader >> v1_index >> v2_index >> v3_index;
			Eigen::Vector3u f(v1_index - 1, v2_index - 1, v3_index - 1); 
			faces.push_back(f); 
		}
		else
		{
			// nothing 
		}
	}
}