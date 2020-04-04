#include "obj_reader.h"


void OBJReader::read(std::string filename)
{
	std::fstream reader;
	reader.open(filename.c_str(), std::ios::in);
	std::string tempstr;
	if (!reader.is_open())
	{
		std::cout << "can not open the file" << std::endl;
		return;
	}
	double v1, v2, v3;
	double vn1, vn2, vn3;
	double t1, t2;
	int v1_index, t1_index, n1_index,
		v2_index, t2_index, n2_index,
		v3_index, t3_index, n3_index;
	std::vector<std::pair<int, int> > vtpairs;
	char ch;
	int i = 0;
	while (!reader.eof())
	{
		reader >> tempstr;
		if (tempstr == "v")
		{
			reader >> v1 >> v2 >> v3;
			Eigen::Vector3d temp_v(v1, v2, v3);
			vertices.push_back(temp_v);
		}
		else if (tempstr == "vn")
		{
		}
		else if (tempstr == "vt")
		{
			reader >> t1 >> t2;
			Eigen::Vector2d temp_vt(t1, t2);
			textures.push_back(temp_vt);
		}
		else if (tempstr == "f")
		{
			
			std::string v_str_1, v_str_2, v_str_3;
			reader >> v_str_1 >> v_str_2 >> v_str_3;
			split_face_str(v_str_1, v1_index, t1_index, n1_index);
			split_face_str(v_str_2, v2_index, t2_index, n2_index);
			split_face_str(v_str_3, v3_index, t3_index, n3_index);
			faces_v.push_back(Eigen::Vector3u(v1_index-1, v2_index-1, v3_index-1));
			if (t1_index > 0 && t2_index > 0 && t3_index > 0)
			{
				faces_t.push_back(Eigen::Vector3u(t1_index - 1, t2_index - 1, t3_index - 1));
				vtpairs.push_back({ t1_index - 1,v1_index - 1 });
				vtpairs.push_back({ t2_index - 1,v2_index - 1 });
				vtpairs.push_back({ t3_index - 1,v3_index - 1 });
			}
		}
		else
		{

		}
		//std::cout << i << ": " << tempstr << std::endl;
		//ch = getch();
		//if (ch == 'e')break;
	}

	vertices_eigen.resize(3, vertices.size()); 
	for(int i = 0; i < vertices.size(); i++) vertices_eigen.col(i) = vertices[i]; 
	faces_v_eigen.resize(3, faces_v.size()); 
	for(int i = 0; i < faces_v.size(); i++) faces_v_eigen.col(i) = faces_v[i].cast<unsigned int>(); 
	if (textures.size() > 0)
	{
		textures_eigen.resize(2, textures.size());
		for (int i = 0; i < textures.size(); i++) textures_eigen.col(i) = textures[i];
		tex_to_vert.resize(textures.size(),-1); 

		std::cout << "vtpairs.size:" << vtpairs.size() << std::endl;
		for (int i = 0; i < vtpairs.size(); i++)
		{
			int t = vtpairs[i].first;
			int v = vtpairs[i].second;
			if (tex_to_vert[t] >= 0)
			{
				if (tex_to_vert[t] != v)
				{
					std::cout << "tex to vert mapping error ! " << std::endl; 
					std::cout << "old: " << tex_to_vert[t] << "," << v << ": ";
					Eigen::Vector3d a = vertices[tex_to_vert[t]];
					Eigen::Vector3d b = vertices[v];
					float dist = (a - b).norm();
					std::cout << "dist: " << dist << std::endl;
				}
			}
			else
			{
				tex_to_vert[t] = v;
			}
		}
	}
	std::cout << "[OBJReader]vertices number:" << vertices.size() << std::endl;
	std::cout << "[OBJReader]textures number:" << textures.size() << std::endl;
	std::cout << "[OBJReader]faces v size   :" << faces_v.size() << std::endl; 
}

void OBJReader::split_face_str(std::string str, int &i1, int &i2, int &i3)
{
	std::vector<std::string> strs;
	boost::split(strs, str, boost::is_any_of("/"));
	if (strs.size() == 3){
		if(strs[0]!="")
			i1 = stoi(strs[0]);
		else i1 = -1;
		if(strs[1]!="")
			i2 = stoi(strs[1]);
		else i2 = -1; 
		if(strs[2]!="")
			i3 = stoi(strs[2]);
		else i3 = -1;
	}
	else
	{
		std::cout << "split error: " << str << std::endl;
	}
}
float OBJReader::calcTriangleArea(double u1, double v1,
	double u2, double v2,
	double u3, double v3)
{
	double result = u1*(v2 - v3) + u2*(v3 - v1) + u3*(v1 - v2);
	return result > 0 ? result : -result;
}


void OBJReader::write(std::string filename)
{
	std::ofstream f(filename);
	for (int i = 0; i < vertices_eigen.cols(); i++)
	{
		f << "v " << vertices_eigen(0, i) << " " << vertices_eigen(1, i) << " " << vertices_eigen(2, i) << std::endl;
	}

	for (int i = 0; i < faces_v_eigen.cols(); i++)
	{
		f << "f " << faces_v_eigen(0, i) + 1 << " " << faces_v_eigen(1, i) + 1 << " " << faces_v_eigen(2, i) + 1 << std::endl;
	}
	f.close();
}