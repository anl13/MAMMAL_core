#include "mesh.h"

MeshEigen::MeshEigen(const Mesh& _mesh)
{
	vertices.resize(3, _mesh.vertex_num);
#pragma omp parallel for 
	for (int i = 0; i < _mesh.vertex_num; i++) vertices.col(i) = _mesh.vertices_vec[i];

	normals.resize(3, _mesh.vertex_num);
#pragma omp parallel for 
	for (int i = 0; i < _mesh.vertex_num; i++) normals.col(i) = _mesh.normals_vec[i];

	faces.resize(3, _mesh.face_num);
#pragma omp parallel for 
	for (int i = 0; i < _mesh.face_num; i++) faces.col(i) = _mesh.faces_v_vec[i];

	textures.resize(2, _mesh.texture_num);
#pragma omp parallel for 
	for (int i = 0; i < _mesh.texture_num; i++) textures.col(i) = _mesh.textures_vec[i];
}

void MeshEigen::Deform(const Eigen::Vector3f &xyzScale)
{
	vertices.row(0) = vertices.row(0) * xyzScale(0);
	vertices.row(1) = vertices.row(1) * xyzScale(1);
	vertices.row(2) = vertices.row(2) * xyzScale(2);
}

MeshFloat4::MeshFloat4(const Mesh& _mesh)
{
	LoadFromMesh(_mesh);
}

void MeshFloat4::LoadFromMesh(const Mesh& _mesh)
{
	vertices.resize(_mesh.vertex_num);
	for (int i = 0; i < _mesh.vertex_num; i++) vertices[i] = make_float4(_mesh.vertices_vec[i].x(),
		_mesh.vertices_vec[i].y(), _mesh.vertices_vec[i].z(), 1.0f);

	normals.resize(_mesh.vertex_num);
	for (int i = 0; i < _mesh.vertex_num; i++) normals[i] = make_float4(_mesh.normals_vec[i].x(),
		_mesh.normals_vec[i].y(), _mesh.normals_vec[i].z(), 1.0f);

	indices.resize(3 * _mesh.face_num);
	for (int i = 0; i < _mesh.face_num; i++)
	{
		indices[3 * i + 0] = _mesh.faces_v_vec[i].x();
		indices[3 * i + 1] = _mesh.faces_v_vec[i].y();
		indices[3 * i + 2] = _mesh.faces_v_vec[i].z();
	}
	textures.resize(_mesh.texture_num);
	for (int i = 0; i < _mesh.texture_num; i++)
	{
		textures[i].x = _mesh.textures_vec[i](0);
		textures[i].y = _mesh.textures_vec[i](1);
	}
}

void Mesh::CalcNormal()
{
	normals_vec.resize(vertex_num, Eigen::Vector3f::Zero()); 
	for (int fIdx = 0; fIdx < faces_v_vec.size(); fIdx++) {
		const Eigen::Vector3u face = faces_v_vec[fIdx];
		Eigen::Vector3f normal = ((vertices_vec[face.x()] - vertices_vec[face.y()]).cross(
			vertices_vec[face.y()] - vertices_vec[face.z()]));
		float area = normal.norm();  
		float area_sq = area * area; 

		normals_vec[face.x()] += normal / area_sq;
		normals_vec[face.y()] += normal / area_sq;
		normals_vec[face.z()] += normal / area_sq;
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

void Mesh::Load(const std::string& filename, bool isReadTex, bool isCalcNormal)
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
	vertices_vec_t.clear(); 

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
			if (!isReadTex) continue;
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
			if (!isReadTex) continue; 
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

	if(isCalcNormal)
		CalcNormal(); 

	// map texture 
	if (isReadTex && texture_num > 0)
	{
		vertices_vec_t.resize(texture_num);
		tex2vert.resize(texture_num, -1);
		normals_vec_t.resize(texture_num); 
		for (int i = 0; i < face_num; i++)
		{
			Eigen::Vector3u fv = faces_v_vec[i];
			Eigen::Vector3u ft = faces_t_vec[i];
			for (int k = 0; k < 3; k++)
			{
				if (tex2vert[ft(k)] < 0)
				{
					tex2vert[ft(k)] = fv[k];
				}
				else
				{
					if (tex2vert[ft(k)] != fv[k])
					{
						std::cout << "error: " << ft(k) << " --> " << tex2vert[ft(k)] << " by " << fv[k] << std::endl;
					}
				}
			}
		}
		for (int i = 0; i < texture_num; i++)
		{
			if (tex2vert[i] < 0)
			{
				std::cout << "false tex2vert map. " << std::endl; 
				continue; 
			}
			vertices_vec_t[i] = vertices_vec[tex2vert[i]];
			normals_vec_t[i] = normals_vec[tex2vert[i]];
		}
	}
}

void Mesh::ReMapTexture()
{
	// map texture 
	if (texture_num > 0)
	{
		vertices_vec_t.resize(texture_num);
		normals_vec_t.resize(texture_num);
		for (int i = 0; i < texture_num; i++)
		{
			if (tex2vert[i] < 0)
			{
				std::cout << "false tex2vert map. " << std::endl;
				continue;
			}
			vertices_vec_t[i] = vertices_vec[tex2vert[i]];
			normals_vec_t[i] = normals_vec[tex2vert[i]];
		}
	}
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


void MeshEigen::CalcNormal()
{
	normals.resize(3, vertices.cols());
	normals.setZero();
	for (int fIdx = 0; fIdx < faces.cols(); fIdx++) {
		const Eigen::Vector3u face = faces.col(fIdx);
		Eigen::Vector3f normal = ((vertices.col(face.x()) - vertices.col(face.y())).cross(
			vertices.col(face.y()) - vertices.col(face.z())));
		float area = normal.norm(); 
		area = area * area;
		normals.col(face.x()) += normal / area;
		normals.col(face.y()) += normal / area;
		normals.col(face.z()) += normal / area;
	}
	for (int i = 0; i < normals.cols(); i++) normals.col(i).normalize();
}

void composeMesh(Mesh& mesh1, const Mesh& mesh2)
{
	int vertexnum1 = mesh1.vertex_num;
	int vertexnum2 = mesh2.vertex_num;
	int facenum1 = mesh1.face_num;
	int facenum2 = mesh2.face_num;
	mesh1.vertices_vec.insert(mesh1.vertices_vec.end(), mesh2.vertices_vec.begin(), mesh2.vertices_vec.end());
	mesh1.normals_vec.insert(mesh1.normals_vec.end(), mesh2.normals_vec.begin(), mesh2.normals_vec.end());
	mesh1.faces_v_vec.insert(mesh1.faces_v_vec.end(), mesh2.faces_v_vec.begin(), mesh2.faces_v_vec.end());
	for (int i = facenum1; i < facenum1 + facenum2; i++)
	{
		mesh1.faces_v_vec[i](0) += vertexnum1;
		mesh1.faces_v_vec[i](1) += vertexnum1;
		mesh1.faces_v_vec[i](2) += vertexnum1;
	}

	mesh1.vertex_num = mesh1.vertices_vec.size(); 
	mesh1.face_num = mesh1.faces_v_vec.size(); 
}

void Mesh::flip(int axis)
{
	for (int i = 0; i < vertex_num; i++)
	{
		vertices_vec[i](axis) *= -1; 
	}
	for (int i = 0; i < face_num; i++)
	{
		int a = faces_v_vec[i](0);
		faces_v_vec[i](0) = faces_v_vec[i](1);
		faces_v_vec[i](1) = a; 
	}
}