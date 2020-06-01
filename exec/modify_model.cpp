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
#include "../smal/pigmodel.h"
#include "../smal/pigsolver.h"
#include "../associate/framedata.h"
#include "../utils/volume.h"
#include "../utils/model.h"
#include "../utils/dataconverter.h" 
#include "../nanorender/NanoRenderer.h"
#include <vector_functions.hpp>
#include "../utils/timer.hpp" 
#include "main.h"

#include "../utils/node_graph.h"

// what is necessary

void PigModel::remapmodel()
{

	// targets
	Eigen::MatrixXd vertices; 
	Eigen::Matrix3Xu faces; 
	std::vector<BODY_PART> body_parts; 
	Eigen::MatrixXd joints;
	Eigen::MatrixXd skin_weights; 

	std::vector<std::pair<int, int> > joint_pair =
	{//{newmodel, oldmodel}
		{0,0},{1,1},{2,2},{3,3},{4,4},{5,5},{6,6},{7,7},{8,8},
	{9,20},{10,21},{11,22},{12,23},{13,26},{14,27},{15,28},{16,29},
	{17,38},{18,39},{19,40},{20,41},{21,31},{22,11},{23,12},
	{24,13},{25,14},{26,15},{27,16},{28,18}
	};


	// construct mapping
	Model model("C:\\Users\\Liang\\Documents\\maya\\projects\\default\\scenes\\pigmodel_node2.obj");
	int newVertexNum = model.vertices.cols();
	int newJointNum = 29;
	std::vector<int> jointmap;
	std::vector<int> jointmaprev;
	jointmap.resize(newJointNum);
	jointmaprev.resize(m_jointNum, -1);

	std::vector<int> vertexmap;
	vertexmap.resize(newVertexNum);
	std::vector<int> vertexmaprev(m_vertexNum, -1);
	for (int i = 0; i < newVertexNum; i++)
	{
		Eigen::Vector3d x = model.vertices.col(i);
		for (int j = 0; j < m_vertexNum; j++)
		{
			Eigen::Vector3d y = m_verticesOrigin.col(j);
			double dist = (x - y).norm();
			if (dist < 0.00001)
			{
				vertexmap[i] = j;
				vertexmaprev[j] = i;
			}
		}
	}

	for (int i = 0; i < joint_pair.size(); i++)
	{
		jointmap[joint_pair[i].first] = joint_pair[i].second; 
		jointmaprev[joint_pair[i].second] = joint_pair[i].first; 
	}

	// compute model 
	vertices = model.vertices; 
	faces = model.faces; 
	joints.resize(3, newJointNum);
	for (int i = 0; i < newJointNum; i++)
	{
		joints.col(i) = m_jointsOrigin.col(jointmap[i]);
	}

	body_parts.resize(newVertexNum);
	for (int i = 0; i < newVertexNum; i++)
	{
		body_parts[i] = m_bodyParts[vertexmap[i]];
	}

	skin_weights.resize(newJointNum, newVertexNum);
	skin_weights.setZero();
	auto weights = m_lbsweights;
	weights.row(23) += weights.row(24);
	weights.row(23) += weights.row(25);
	weights.row(8) += weights.row(9);
	weights.row(8) += weights.row(10);
	weights.row(16) += weights.row(17);
	weights.row(18) += weights.row(19);
	weights.row(31) += weights.row(32); 
	weights.row(31) += weights.row(33);
	weights.row(29) += weights.row(30);
	weights.row(41) += weights.row(42);
	for (int i = 0; i < m_jointNum; i++)
	{
		int jid = jointmaprev[i];
		if (jid < 0) continue; 
		for (int j = 0; j < m_vertexNum; j++)
		{
			int vid = vertexmaprev[j];
			if (vid < 0) continue; 
			if (weights(i, j) == 0) continue; 
			skin_weights(jid, vid) = weights(i, j);
		}
	}


	for (int i = 0; i < newVertexNum; i++)
	{
		double sum = skin_weights.col(i).sum();
		if (sum == 0)
		{
			std::cout << "fatal error: " << i << std::endl; 
			std::cout << "body_part: " << body_parts[i] << std::endl; 

		}
		else 
			skin_weights.col(i) = skin_weights.col(i) / sum; 
	}

	// save 
	std::string folder = "D:/Projects/animal_calib/data/pig_model_noeye2/";
	std::ofstream os_v(folder + "vertices.txt");
	os_v << model.vertices.transpose();
	os_v.close();

	std::ofstream os_fv(folder + "faces_vert.txt");
	os_fv << model.faces.transpose();
	os_fv.close();

	std::ofstream os_weight(folder + "skinning_weights.txt");
	for (int i = 0; i < newJointNum; i++)
	{
		for (int j = 0; j < newVertexNum; j++)
		{
			if (skin_weights(i, j) > 0)
			{
				os_weight << i << " " << j << " " << skin_weights(i, j) << std::endl;
			}
		}
	}
	os_weight.close(); 

	std::ofstream os_joint(folder + "t_pose_joints.txt");
	os_joint << joints.transpose() << std::endl; 
	os_joint.close(); 

	/*
	std::vector<int> t_to_v;
	std::vector<int> t_map;
	std::vector<int> t_rev(m_texNum, -1);
	for (int i = 0; i < m_texNum; i++)
	{
	int left = m_texToVert[i];
	if (reverse[left] < 0) continue;
	t_rev[i] = t_map.size();
	t_map.push_back(i);
	}
	int tnum = t_map.size();
	t_to_v.resize(tnum, -1);
	int count = vnum;
	for (int i = 0; i < tnum; i++)
	{
	int t1 = t_map[i];
	int v1 = m_texToVert[t1];
	int n1 = reverse[v1];
	t_to_v[i] = n1;
	}
	std::ofstream os_t_to_v(folder + "tex_to_vert.txt");
	for (int i = 0; i < tnum; i++)os_t_to_v << t_to_v[i] << "\n";
	os_t_to_v.close();
	std::ofstream os_tex(folder + "textures.txt");
	for (int i = 0; i < tnum; i++)
	{
	int tn = t_map[i];
	Eigen::Vector2d tex = m_texcoords.col(tn);
	os_tex << tex(0) << " " << tex(1) << "\n";
	}
	os_tex.close();
	std::ofstream os_ft(folder + "faces_tex.txt");
	for (int i = 0; i < m_faceNum; i++)
	{
	int t1 = m_facesTex(0, i);
	int t2 = m_facesTex(1, i);
	int t3 = m_facesTex(2, i);
	if (t_rev[t1] < 0 || t_rev[t2] < 0 || t_rev[t3] < 0)continue;
	os_ft << t_rev[t1] << " " << t_rev[t2] << " " << t_rev[t3] << "\n";
	}
	os_ft.close();
	*/

	//for (int i = 0; i < m_vertexNum; i++)
	//{
	//	if (reverse[i] < 0)continue;
	//	Eigen::VectorXd x = m_lbsweights.col(i);
	//	double w = x.sum();
	//	if (w > 0)std::cout << w << std::endl; 
	//}


	// use new model as node graph for old model 
	NodeGraphGenerator nodegen; 
	Model raw;
	raw.vertices = m_verticesOrigin;
	raw.faces = m_facesVert; 
	raw.CalcNormal();
	nodegen.model = raw; 
	nodegen.CalcGeodesic();
	nodegen.nodeIdx.resize(vertexmap.size());
	for (int i = 0; i < vertexmap.size(); i++)
		nodegen.nodeIdx(i) = vertexmap[i];
	nodegen.GenKnn();
	nodegen.nodeNet = Eigen::MatrixXi::Constant(4, vertexmap.size(), -1);
	
	std::vector<std::map<int, double>> nbor(newVertexNum);
	for (int fIdx = 0; fIdx < model.faces.cols(); fIdx++) {
		for (int i = 0; i < 3; i++) {
			const int va = model.faces(i, fIdx);
			const int vb = model.faces((i + 1) % 3, fIdx);
			const double norm = (model.vertices.col(va) - model.vertices.col(vb)).norm();
			nbor[va].insert(std::make_pair(vb, norm));
			nbor[vb].insert(std::make_pair(va, norm));
		}
	}

	for (int i = 0; i < newVertexNum; i++)
	{
		std::vector<std::pair<double, int>> diss; 
		for (const auto& nb : nbor[i])
		{
			diss.push_back({ nb.second, nb.first });
		}
		std::sort(diss.begin(), diss.end());
		if (diss.size() < 4)
		{
			std::cout << "no enough nbor . " << i 
				<< " : " << diss.size() << std::endl; 
		}
		for (int k = 0; k < 4; k++)
		{
			if (k >= diss.size()) nodegen.nodeNet(k, i) = -1; 
			else nodegen.nodeNet(k, i) = diss[k].second; 
		}
	}
	nodegen.Save(m_folder + "/node_graph.txt");
}

int modify_model()
{
	std::string config = "D:/Projects/animal_calib/smal/pigmodel_config.json";
	PigModel model(config);

	model.remapmodel();

	system("pause");
	return 0; 
}