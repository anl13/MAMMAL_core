#include "dataconverter.h"

void convert3CTo4C(
	const Model& in_m3c,
	ObjModel& out_m4c
)
{
	out_m4c.vertices.clear(); 
	out_m4c.normals.clear();
	out_m4c.indices.clear();
	
	out_m4c.vertices.resize(in_m3c.vertices.cols());
#pragma omp parallel for
	for (int i = 0; i < in_m3c.vertices.cols(); i++)
	{
		out_m4c.vertices[i].x = in_m3c.vertices(0, i);
		out_m4c.vertices[i].y = in_m3c.vertices(1, i);
		out_m4c.vertices[i].z = in_m3c.vertices(2, i);
		out_m4c.vertices[i].w = 1.0f;
	}

	out_m4c.normals.resize(in_m3c.normals.cols());
#pragma omp parallel for 
	for (int i = 0; i < in_m3c.normals.cols(); i++)
	{
		out_m4c.normals[i].x = in_m3c.normals(0, i);
		out_m4c.normals[i].y = in_m3c.normals(1, i);
		out_m4c.normals[i].z = in_m3c.normals(2, i);
		out_m4c.normals[i].w = 1.0f;
	}

	out_m4c.indices.resize(in_m3c.faces.cols()*3);
#pragma omp parallel for 
	for (int i = 0; i < in_m3c.faces.cols(); i++)
	{
		out_m4c.indices[3 * i + 0] = in_m3c.faces(0, i);
		out_m4c.indices[3 * i + 1] = in_m3c.faces(1, i);
		out_m4c.indices[3 * i + 2] = in_m3c.faces(2, i);
	}
}

void convert4CTo3C(
	const ObjModel& in_m4c,
	Model& out_m3c
)
{
	// TODO
	std::cout << "TODO! convert4CTo3C not implemented" << std::endl;
}

nanogui::Matrix4f eigen2nanoM4f(const Eigen::Matrix4f& mat)
{
	nanogui::Matrix4f M; 
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			M.m[j][i] = mat(i, j);
		}
	}
	return M; 
}

Eigen::Matrix4f nano2eigenM4f(const nanogui::Matrix4f& mat)
{
	Eigen::Matrix4f M; 
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			M(i, j) = mat.m[j][i];
		}
	}
	return M; 
}