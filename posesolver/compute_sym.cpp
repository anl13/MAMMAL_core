#include "main.h"
#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 

#include "../utils/colorterminal.h" 
#include "../utils/obj_reader.h"
#include "../utils/timer_util.h"
#include "../articulation/pigmodel.h"
#include "../articulation/pigsolver.h"
#include "../associate/framedata.h"
#include "../utils/volume.h"
#include "../utils/model.h"
#include "../utils/dataconverter.h" 
#include "../nanorender/NanoRenderer.h"
#include <vector_functions.hpp>
#include "../utils/timer.hpp" 
#include "main.h"


#define RUN_SEQ
#define VIS 
#define DEBUG_VIS
//#define LOAD_STATE
#define SHAPE_SOLVER
//#define VOLUME

using std::vector;

int ComputeSymmetry()
{
	std::string pig_config = "D:/Projects/animal_calib/smal/pigmodel_config.json";
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	SkelTopology topo = getSkelTopoByType("UNIV");

	PigModel pig(pig_config); 
	
	OBJReader obj; 
	obj.vertices_eigen = pig.GetVertices();
	obj.faces_v_eigen = pig.GetFacesVert(); 

	OBJReader obj2; 
	obj2 = obj; 
	obj2.vertices_eigen.row(0) = -obj2.vertices_eigen.row(0);
	obj2.faces_v_eigen.row(0) = obj.faces_v_eigen.row(1);
	obj2.faces_v_eigen.row(1) = obj.faces_v_eigen.row(0);
	obj2.write("D:/Projects/animal_calib/data/pig_model_noeye/flip.obj");
	
	int vertexNum = pig.GetVertexNum(); 
	auto parts = pig.GetBodyPart(); 

	int N = 3; // symmetric degree
	Eigen::MatrixXi sym(N, vertexNum); 
	Eigen::MatrixXd symweights(N, vertexNum);

	sym.setConstant(-1);
	symweights.setConstant(-1);

	int M = 8; // nearest neibhour to search 

	std::shared_ptr<const KDTree<double> > m_tarTree = std::make_shared<KDTree<double>>(obj2.vertices_eigen);
	for (int i = 0; i < vertexNum; i++)
	{
		if (parts[i] == TAIL) continue; 
		Eigen::Vector3d v = obj.vertices_eigen.col(i);
		std::vector<std::pair<double, size_t> > nbors = m_tarTree->KNNSearch(v, M);
		int n = 0;
		for (int k = 0; k < M; k++)
		{
			if (n == N)break; 
			double dist = nbors[k].first; 
			int id = nbors[k].second; 
			if (id < 0)
			{
				std::cout << "i:" << i <<  " id : " << id << std::endl; 
			}
			if (parts[id] != TAIL)
			{
				sym(n, i) = id; 
				symweights(n, i) = 0.01 / (dist + 0.00001); 
				n++;
			}
			if (k == M-1 && n < N)
			{
				std::cout << "no enough valid neibour " << i << ", " << n << std::endl;
			}
		}
	}

	//for (int i = 0; i < vertexNum; i++)
	//{
	//	if (symweights(0, i) >= 0)
	//	{
	//		symweights.col(i).normalize();
	//	}
	//}

	std::string sym_file = "D:/Projects/animal_calib/data/pig_model_noeye/sym.txt";
	std::ofstream sym_stream(sym_file);
	for (int i = 0; i < vertexNum; i++)
	{
		for (int k = 0; k < N; k++)
		{
			sym_stream << sym(k, i) << " ";
		}
		sym_stream << std::endl;
	}
	std::string symweight_file = "D:/Projects/animal_calib/data/pig_model_noeye/symweights.txt";
	std::ofstream symw_stream(symweight_file); 
	for (int i = 0; i < vertexNum; i++)
	{
		for (int k = 0; k < N; k++)
		{
			symw_stream << symweights(k, i) << " ";
		}
		symw_stream << std::endl;
	}
	sym_stream.close(); 
	symw_stream.close(); 


	/// render
	NanoRenderer renderer;
	std::vector<float4> colormap = getColorMapFloat4("jet");
	int colornum = colormap.size();

	Model model1;
	model1.vertices = pig.GetVertices();
	model1.faces = pig.GetFacesVert();
	model1.CalcNormal();
	ObjModel m4c;
	convert3CTo4C(model1, m4c);

	std::vector<float4> color1;
	std::vector<float4> color2;
	color1.resize(model1.vertices.cols());
	color2.resize(model1.vertices.cols());

	for (int i = 0; i < vertexNum; i++)
	{
		color1[i] = colormap[i%colornum];

		int id = sym(0, i);
		if (id < 0) color2[i] = make_float4(1.0f, 1.f, 1.f, 1.f);
		color2[i] = colormap[id%colornum];
	}

	renderer.Init(1920, 1080, 1340.f, 1340.f, 960, 540, -2.0f);
	renderer.ClearRenderObjects();
	auto smal_model = renderer.CreateRenderObject("smal_model", vs_phong_vertex_color, fs_phong_vertex_color);
	smal_model->SetIndices(m4c.indices);
	smal_model->SetBuffer("positions", m4c.vertices);
	smal_model->SetBuffer("normals", m4c.normals);
	smal_model->SetBuffer("incolor", color2);

	int frameIdx = 0;
	while (!renderer.ShouldClose())
	{
		renderer.Draw();
		++frameIdx;
	}

	//system("pause");
	return 0;
}