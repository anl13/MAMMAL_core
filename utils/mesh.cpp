#include "mesh.h"

void MeshEigen::Deform(const Eigen::Vector3f &xyzScale)
{
	vertices.row(0) = vertices.row(0) * xyzScale(0);
	vertices.row(1) = vertices.row(1) * xyzScale(1);
	vertices.row(2) = vertices.row(2) * xyzScale(2);
}

void Mesh::CalcNormal()
{
	normals_vec.resize(vertex_num, Eigen::Vector3f::Zero()); 
	for (int fIdx = 0; fIdx < faces_v_vec.size(); fIdx++) {
		const Eigen::Vector3u face = faces_v_vec[fIdx];
		//if (face(0) >= vertex_num || face(1) >= vertex_num || face(2) >= vertex_num)
		//{
		//	std::cout << "vertex_num: " << vertex_num << std::endl; 
		//	std::cout << "face: " << face.transpose() << std::endl; 
		//	system("pause"); 
		//	exit(-1);
		//}
		Eigen::Vector3f normal = ((vertices_vec[face.x()] - vertices_vec[face.y()]).cross(
			vertices_vec[face.y()] - vertices_vec[face.z()])).normalized();

		normals_vec[face.x()] += normal;
		normals_vec[face.y()] += normal;
		normals_vec[face.z()] += normal;
	}
	for (int i = 0; i < vertex_num; i++) normals_vec[i].normalize(); 
}

void Mesh::SplitFaceStr(std::string &str, int &i1, int &i2, int &i3)
{
	i1 = -1; 
	i2 = -1; 
	i2 = -1; 
	std::vector<std::string> strs;
	boost::split(strs, str, boost::is_any_of("/"));
	if (strs.size() >= 1 && strs[0] != "") i1 = stoi(strs[0]); 
	if (strs.size() >= 2 && strs[1] != "") i2 = stoi(strs[1]); 
	if (strs.size() == 3 && strs[2] != "") i3 = stoi(strs[2]); 

	if(strs.size() < 1 || strs.size() > 3)
	{
		std::cout << "split error: " << str << std::endl;
	}
}

void Mesh::Load(const std::string& filename)
{
	std::vector<std::string> strs;
	boost::split(strs, filename, boost::is_any_of("."));
	if (strs[strs.size() - 1] != "obj")
	{
		std::cout << "[Mesh::Load] Attention! Not a obj file. " << std::endl; 
		return; 
	}

	vertices_vec.clear(); 
	normals_vec.clear(); 
	textures_vec.clear(); 
	faces_v_vec.clear(); 
	faces_t_vec.clear(); 

	std::fstream reader;
	reader.open(filename.c_str(), std::ios::in);
	
	if (!reader.is_open())
	{
		std::cout << "[Mesh::Load] can not open the file" << std::endl;
		return;
	}
	float v1, v2, v3;
	float vn1, vn2, vn3;
	float t1, t2;
	int v1_index, t1_index, n1_index,
		v2_index, t2_index, n2_index,
		v3_index, t3_index, n3_index;
	std::vector<std::pair<int, int> > vtpairs;
	char ch;
	while (!reader.eof())
	{
		std::string tempstr;
		reader >> tempstr;
		if (tempstr == "v")
		{
			reader >> v1 >> v2 >> v3;
			Eigen::Vector3f temp_v((float)v1, (float)v2, (float)v3);
			vertices_vec.push_back(temp_v);
		}
		else if (tempstr == "vn")
		{
			reader >> vn1 >> vn2 >> vn3; 
			Eigen::Vector3f temp_vn((float)vn1, (float)vn2, (float)vn3);
			normals_vec.push_back(temp_vn); 
		}
		else if (tempstr == "vt")
		{
			reader >> t1 >> t2;
			Eigen::Vector2f temp_vt(t1, t2);
			textures_vec.push_back(temp_vt);
		}
		else if (tempstr == "f")
		{
			std::string v_str_1, v_str_2, v_str_3;
			reader >> v_str_1 >> v_str_2 >> v_str_3;
			SplitFaceStr(v_str_1, v1_index, t1_index, n1_index);
			SplitFaceStr(v_str_2, v2_index, t2_index, n2_index);
			SplitFaceStr(v_str_3, v3_index, t3_index, n3_index);
			if (v1_index <= 0 || v2_index <= 0 || v3_index <= 0)
			{
				std::cout << "v_str: " << v_str_1 << "  " << v_str_2 << "  " << v_str_3 << std::endl; 
			}
			faces_v_vec.push_back(Eigen::Vector3u(v1_index - 1, v2_index - 1, v3_index - 1));
			if (t1_index > 0 && t2_index > 0 && t3_index > 0)
			{
				faces_t_vec.push_back(Eigen::Vector3u(t1_index - 1, t2_index - 1, t3_index - 1));
			}
		}
		else
		{
			// nothing 
		}
	}

	vertex_num = vertices_vec.size();
	texture_num = textures_vec.size(); 
	face_num = faces_v_vec.size(); 

	CalcNormal(); 
}

void Mesh::Save(const std::string &filename) const
{
	std::ofstream f(filename);
	for (int i = 0; i < vertices_vec.size(); i++)
	{
		f << "v " << vertices_vec[i].transpose() << std::endl;
	}

	if (texture_num > 0)
	{
		for (int i = 0; i < texture_num; i++)
			f << "vt " << textures_vec[i].transpose() << std::endl; 
	}

	for (int i = 0; i < faces_v_vec.size(); i++)
	{
		if(texture_num == 0)
			f << "f " << faces_v_vec[i](0) + 1 << " " << faces_v_vec[i](1) + 1 << " " << faces_v_vec[i](2) + 1 << std::endl;
		else
		{
			f << "f " << faces_v_vec[i](0) + 1 <<"/"<<faces_t_vec[i](0)+1 << " "
				<< faces_v_vec[i](1) + 1 << "/" << faces_t_vec[i](1)+1 << " " \
				<< faces_v_vec[i](2) + 1 << "/" << faces_t_vec[i](2)+1 << std::endl;
		}
	}
	f.close();
}

void Mesh::GetMeshEigen(MeshEigen &_m)
{
	_m.vertices.resize(3, vertex_num); 
	for (int i = 0; i < vertex_num; i++) _m.vertices.col(i) = vertices_vec[i]; 

	_m.normals.resize(3, vertex_num); 
	for (int i = 0; i < vertex_num; i++) _m.normals.col(i) = normals_vec[i]; 

	_m.faces.resize(3, face_num); 
	for (int i = 0; i < face_num; i++) _m.faces.col(i) = faces_v_vec[i]; 

	_m.textures.resize(2, texture_num); 
	for (int i = 0; i < texture_num; i++) _m.textures.col(i) = textures_vec[i]; 

}

void Mesh::GetMeshFloat4(MeshFloat4 &_m)
{
	_m.vertices.resize(vertex_num); 
	for (int i = 0; i < vertex_num; i++) _m.vertices[i] = make_float4(vertices_vec[i].x(),
		vertices_vec[i].y(), vertices_vec[i].z(), 1.0f); 

	_m.normals.resize(vertex_num); 
	for (int i = 0; i < vertex_num; i++) _m.normals[i] = make_float4(normals_vec[i].x(),
		normals_vec[i].y(), normals_vec[i].z(), 1.0f);
	
	_m.indices.resize(3 * face_num); 
	for (int i = 0; i < face_num; i++)
	{
		_m.indices[3 * i + 0] = faces_v_vec[i].x(); 
		_m.indices[3 * i + 1] = faces_v_vec[i].y();
		_m.indices[3 * i + 2] = faces_v_vec[i].z();
	}
}

