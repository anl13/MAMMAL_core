#pragma once
#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include "mesh.h"

struct NodeGraph
{
	Eigen::VectorXi nodeIdx;
	Eigen::MatrixXi nodeNet;
	Eigen::MatrixXi knn;
	Eigen::MatrixXf weight;

	NodeGraph() {};
	NodeGraph(const std::string& folder) { Load(folder); }
	void Load(const std::string& filename);
	void Save(const std::string& filename) const;
};


struct NodeGraphGenerator : public NodeGraph
{
	Eigen::VectorXf Dijkstra(const int& start, const Eigen::MatrixXf& w)const;
	void Generate(const Mesh& _model);
	void CalcGeodesic();
	void SampleNode();
	void GenKnn();
	void GenNodeNet();
	void VisualizeNodeNet(const std::string& filename) const;
	void VisualizeKnn(const std::string& filename) const;
	void LoadGeodesic(const std::string& filename);
	void SaveGeodesic(const std::string& filename) const;

	// param for pigmodel
	//int netDegree = 4;
	//int knnSize = 4;
	//double nodeSpacing = 0.08f;
	//double cutRate = 20.f;
	// param for smal notail
	int netDegree = 4;
	int knnSize = 8;
	float nodeSpacing = 0.05f;
	float cutRate = 5.f;
	Mesh model;
	std::vector<std::map<int, float>> geodesic;
};
