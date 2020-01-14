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
	float v1, v2, v3;
	float vn1, vn2, vn3;
	float t1, t2;
	int v1_index, t1_index, n1_index,
		v2_index, t2_index, n2_index,
		v3_index, t3_index, n3_index;

	char ch;
	int i = 0;
	while (!reader.eof())
	{
		reader >> tempstr;
		if (tempstr == "v")
		{
			reader >> v1 >> v2 >> v3;
			Eigen::Vector3f temp_v(v1, v2, v3);
			vertices.push_back(temp_v);
		}
		else if (tempstr == "vn")
		{
            //// AN Liang 201912
			// reader >> vn1 >> vn2 >> vn3;
			// Eigen::Vector3f temp_vn(vn1, vn2, vn3);
			// normals.push_back(temp_vn);
		}
		else if (tempstr == "vt")
		{
			reader >> t1 >> t2;
			Eigen::Vector2f temp_vt(t1, t2);
			textures.push_back(temp_vt);
		}
		else if (tempstr == "f")
		{
			
			std::string v_str_1, v_str_2, v_str_3;
			reader >> v_str_1 >> v_str_2 >> v_str_3;
			split_face_str(v_str_1, v1_index, t1_index, n1_index);
			split_face_str(v_str_2, v2_index, t2_index, n2_index);
			split_face_str(v_str_3, v3_index, t3_index, n3_index);
			faces_v.push_back(Eigen::Vector3i(v1_index-1, v2_index-1, v3_index-1));
			faces_t.push_back(Eigen::Vector3i(t1_index-1, t2_index-1, t3_index-1));
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
	std::cout << "[OBJReader]vertices number:" << vertices.size() << std::endl;
	std::cout << "[OBJReader]textures number:" << textures.size() << std::endl;
	std::cout << "[OBJReader]faces v size   :" << faces_v.size() << std::endl; 
}

void OBJReader::split_face_str(std::string str, int &i1, int &i2, int &i3)
{
	std::vector<std::string> strs;
	boost::split(strs, str, boost::is_any_of("/"));
	if (strs.size() == 3){
		i1 = stoi(strs[0]);
		i2 = stoi(strs[1]);
		i3 = stoi(strs[2]);
	}
	else
	{
		std::cout << "split error: " << str << std::endl;
	}
}
float OBJReader::calcTriangleArea(float u1, float v1,
	float u2, float v2,
	float u3, float v3)
{
	float result = u1*(v2 - v3) + u2*(v3 - v1) + u3*(v1 - v2);
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