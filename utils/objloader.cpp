#include "objloader.h"
#include <vector_functions.hpp>
#include <vector>
#include <stdio.h>
#include <string>
#include <cstring>


bool loadOBJ(
	const char * path,
	std::vector<float4> & out_vertices,
	std::vector<float4> & out_normals,
	std::vector<unsigned int> & out_indices) 
{
	printf("Loading OBJ file %s...\n", path);

	std::vector<unsigned int> vertexIndices;
	std::vector<float4> temp_vertices;
	std::vector<float4> temp_normals;


	FILE * file = fopen(path, "r");
	if (file == NULL) {
		printf("Impossible to open the file !\n");
		return false;
	}

	while (1) {

		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.

		// else : parse lineHeader
		if (strcmp(lineHeader, "v") == 0) {
			float4 vertex = make_float4(0,0,0,1);
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			temp_vertices.push_back(vertex);
		}
		else if (strcmp(lineHeader, "vn") == 0) {
			float4 normal = make_float4(0,0,0,1);
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			temp_normals.push_back(normal);
		}
		else if (strcmp(lineHeader, "f") == 0) {
			unsigned int vertexIndex[3];
			int matches = fscanf(file, "%d %d %d\n", &vertexIndex[0], &vertexIndex[1], &vertexIndex[2]);
			if (matches != 3) {
				printf("File can't be read by this simple parser :-( Try exporting with other options\n");
				return false;
			}
			vertexIndices.push_back(vertexIndex[0] - 1);
			vertexIndices.push_back(vertexIndex[1] - 1);
			vertexIndices.push_back(vertexIndex[2] - 1);
		}
		else {
			// Probably a comment, eat up the rest of the line
			char stupidBuffer[1000];
			fgets(stupidBuffer, 1000, file);
		}
	}

	out_vertices = temp_vertices;
	out_normals = temp_normals;
	out_indices = vertexIndices;

	return true;
}
