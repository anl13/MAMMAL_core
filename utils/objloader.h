#ifndef OBJLOADER_H
#define OBJLOADER_H
#include <Eigen/Eigen>
#include <cuda_runtime_api.h>

struct ObjModel
{
	std::vector<float4> vertices;
	std::vector<float4> normals;
	std::vector<unsigned int> indices;
};

bool loadOBJ(
	const char * path,
	std::vector<float4> & out_vertices,
	std::vector<float4> & out_normals,
	std::vector<unsigned int> &out_indices);


#endif