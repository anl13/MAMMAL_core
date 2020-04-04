#pragma once
#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include "model.h"


struct NodeGraph
{
	Eigen::VectorXi nodeIdx;
	Eigen::MatrixXi nodeNet;
	Eigen::MatrixXi knn;
	Eigen::MatrixXd weight;

	NodeGraph() {};
	NodeGraph(const std::string& folder) { Load(folder); }
	void Load(const std::string& filename);
	void Save(const std::string& filename) const;
};


struct NodeGraphGenerator : public NodeGraph
{
	Eigen::VectorXd Dijkstra(const int& start, const Eigen::MatrixXd& w)const;
	void Generate(const Model& _model);
	void CalcGeodesic();
	void SampleNode();
	void GenKnn();
	void GenNodeNet();
	void VisualizeNodeNet(const std::string& filename) const;
	void VisualizeKnn(const std::string& filename) const;
	void LoadGeodesic(const std::string& filename);
	void SaveGeodesic(const std::string& filename) const;

	int netDegree = 4;
	int knnSize = 4;
	double nodeSpacing = 0.08f;
	double cutRate = 20.f;
	Model model;
	std::vector<std::map<int, double>> geodesic;
};
