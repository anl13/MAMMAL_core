#include "node_graph.h"
#include "math_utils.h"
#include <set>
#include <fstream>

//#define DEBUG_NODE

void NodeGraph::Load(const std::string& filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		std::cerr << "cannot open " << filename << std::endl;
		std::abort();
	}

	// load nodes
	int nodeSize, vertexSize, netDegree, knnSize;
	ifs >> nodeSize >> vertexSize >> netDegree >> knnSize;
	nodeIdx.resize(nodeSize);
	for (int i = 0; i < nodeSize; i++)
		ifs >> nodeIdx[i];

	// load neighborCnt
	nodeNet.resize(netDegree, nodeSize);
	for (int i = 0; i < nodeSize; i++)
		for (int j = 0; j < netDegree; j++)
			ifs >> nodeNet(j, i);

	// load knn and weights
	knn.resize(knnSize, vertexSize);
	weight.resize(knnSize, vertexSize);
	for (int vi = 0; vi < vertexSize; vi++)
		for (int ki = 0; ki < knnSize; ki++)
			ifs >> knn(ki, vi) >> weight(ki, vi);

	ifs.close();
}


void NodeGraph::Save(const std::string& filename) const
{
	std::ofstream ofs(filename);

	ofs << nodeIdx.size() << "\t" << knn.cols() << "\t" << nodeNet.rows() << "\t" << knn.rows() << std::endl;
	for (int i = 0; i < nodeIdx.size(); i++)
		ofs << nodeIdx[i] << std::endl;
	ofs << std::endl;

	// load neighborCnt
	for (int i = 0; i < nodeIdx.size(); i++) {
		for (int j = 0; j < nodeNet.rows(); j++)
			ofs << nodeNet(j, i) << "\t";
		ofs << std::endl;
	}
	ofs << std::endl;

	// load knn and weights
	for (int vi = 0; vi < knn.cols(); vi++) {
		for (int ki = 0; ki < knn.rows(); ki++)
			ofs << knn(ki, vi) << "\t" << weight(ki, vi) << std::endl;
		ofs << std::endl;
	}

	ofs.close();
}


Eigen::VectorXd NodeGraphGenerator::Dijkstra(const int& start, const Eigen::MatrixXd& w) const
{
	const int n = (int)w.rows();
	Eigen::VectorXd dist = w.col(start);
	Eigen::VectorXi visited = Eigen::VectorXi::Zero(n);

	dist(start) = 0;
	visited(start) = 1;

	// add n+1 vertices
	for (int cnt = 1; cnt < n; cnt++) {
		// find the vertex unvisited nearest to start vertex
		int k = -1;
		double dmin = DBL_MAX;
		for (int i = 0; i < n; i++)
			if (!visited(i) && dist(i) < dmin) {
				dmin = dist(i);
				k = i;
			}

		if (k == -1)
			break;

		visited(k) = 1;
		// update
		for (int i = 0; i < n; i++) {
			if (visited[i] || w(k, i) == DBL_MAX)
				continue;
			dist[i] = std::min(dist[i], dist(k) + w(k, i));
		}
	}
	return dist;
}


void NodeGraphGenerator::CalcGeodesic()
{
	const int n = (int)model.vertices.cols();
	geodesic.clear();
	geodesic.resize(n);

#ifdef DEBUG_NODE
	double max_nbor_dist = 0;
	double min_nbor_dist = 1000;
#endif 
	// generate graph
	std::vector<std::map<int, double>> nbor(n);
	for (int fIdx = 0; fIdx < model.faces.cols(); fIdx++) {
		for (int i = 0; i < 3; i++) {
			const int va = model.faces(i, fIdx);
			const int vb = model.faces((i + 1) % 3, fIdx);
			const double norm = (model.vertices.col(va) - model.vertices.col(vb)).norm();
			nbor[va].insert(std::make_pair(vb, norm));
			nbor[vb].insert(std::make_pair(va, norm));
#ifdef DEBUG_NODE
			if (norm < min_nbor_dist) min_nbor_dist = norm;
			if (norm > max_nbor_dist) max_nbor_dist = norm;
#endif 
		}
	}

#ifdef DEBUG_NODE
	std::cout << "max nbor dist: " << max_nbor_dist << std::endl;
	std::cout << "min nbor dist: " << min_nbor_dist << std::endl;
	int min_nbor_n = 1000;
	int max_nbor_n = -1;
	for (int i = 0; i < n; i++)
	{
		int num = nbor[i].size();
		if (num > max_nbor_n) max_nbor_n = num;
		if (num < min_nbor_n) min_nbor_n = num;
	}
	std::cout << "max nbor n : " << max_nbor_n << std::endl;
	std::cout << "min nbor n : " << min_nbor_n << std::endl;
#endif 

	// calc geodesic distance
	const double cutSpacing = cutRate * nodeSpacing;
#pragma omp parallel for
	for (int srcIdx = 0; srcIdx < n; srcIdx++) {
		std::vector<int> tarIdxMap;
		int start = -1;
		for (int tarIdx = 0; tarIdx < n; tarIdx++) {
			if (tarIdx == srcIdx)
				start = int(tarIdxMap.size());
			if ((model.vertices.col(tarIdx) - model.vertices.col(srcIdx)).norm() < cutSpacing)
				tarIdxMap.emplace_back(tarIdx);
		}

		Eigen::MatrixXd w = Eigen::MatrixXd::Constant(tarIdxMap.size(), tarIdxMap.size(), DBL_MAX);
		for (int i = 0; i < w.rows(); i++) {
			for (const auto& nb : nbor[tarIdxMap[i]]) {
				auto iter = std::find(tarIdxMap.begin(), tarIdxMap.end(), nb.first);
				if (iter != tarIdxMap.end())
					w(i, iter - tarIdxMap.begin()) = nb.second;
			}
		}

		Eigen::VectorXd dist = Dijkstra(start, w);

		for (int i = 0; i < tarIdxMap.size(); i++)
			if (dist[i] != DBL_MAX)
				geodesic[srcIdx].insert(std::make_pair(tarIdxMap[i], dist[i]));
	}
}


void NodeGraphGenerator::SampleNode()
{
	Eigen::VectorXi valid = Eigen::VectorXi::Ones(geodesic.size());

	for (int vIdx = 0; vIdx < geodesic.size(); vIdx++)
		if (valid[vIdx])
			for (const auto& pair : geodesic[vIdx])
				if (valid(pair.first) && pair.first != vIdx && pair.second < nodeSpacing)
					valid[pair.first] = 0;

	nodeIdx.resize((valid.array() > 0).count());
	for (int i = 0, j = 0; i < geodesic.size(); i++)
		if (valid(i))
			nodeIdx[j++] = i;
}


void NodeGraphGenerator::GenKnn()
{
	knn = Eigen::MatrixXi::Constant(knnSize, geodesic.size(), -1);
	weight = Eigen::MatrixXd::Zero(knnSize, geodesic.size());

	for (int vIdx = 0; vIdx < geodesic.size(); vIdx++) {
		std::vector<std::pair<double, int>> dist;
		for (int ni = 0; ni < nodeIdx.size(); ni++) {
			if (nodeIdx[ni] == vIdx)
				continue;
			auto iter = geodesic[vIdx].find(nodeIdx[ni]);
			if (iter != geodesic[vIdx].end() && iter->second != DBL_MAX)
				dist.emplace_back(std::make_pair(iter->second, ni));
		}

		if (dist.size() < knnSize)
		{
			std::cout << "vid: " << vIdx << std::endl;
			std::cerr << "knn dist size: " << dist.size() << " geodesic is not enough, should turn up cut rate" << std::endl;
		}
		std::sort(dist.begin(), dist.end());
		double sum = 0;
		for (int i = 0; i < dist.size() && i < knnSize; i++)
			sum += 1.0 / dist[i].first;

		for (int i = 0; i < dist.size() && i < knnSize; i++) {
			knn(i, vIdx) = dist[i].second;
			weight(i, vIdx) = 1.0 / sum / dist[i].first;
		}
	}
}


void NodeGraphGenerator::GenNodeNet()
{
	nodeNet = Eigen::MatrixXi::Constant(netDegree, nodeIdx.size(), -1);
	for (int ni = 0; ni < nodeIdx.size(); ni++) {
		std::vector<std::pair<double, int>> dist;
		for (int nj = 0; nj < nodeIdx.size(); nj++)
			if (nj != ni) {
				auto iter = geodesic[nodeIdx[ni]].find(nodeIdx[nj]);
				if (iter != geodesic[nodeIdx[ni]].end() && iter->second != DBL_MAX)
					dist.emplace_back(std::make_pair(iter->second, nj));
				//dist.emplace_back(std::make_pair(
				//(model.vertices.col(nodeIdx[ni])
				//	- model.vertices.col(nodeIdx[nj])).norm(), nj));
			}

		if (dist.size() < netDegree)
		{
			std::cerr << "net dist size: " << dist.size() << " geodesic is not enough, should turn up cut rate" << std::endl;
		}

		std::sort(dist.begin(), dist.end());
		for (int i = 0; i < dist.size() && i < netDegree; i++)
			nodeNet(i, ni) = dist[i].second;
	}
}


void NodeGraphGenerator::Generate(const Model& _model)
{
	model = _model;
	CalcGeodesic();
	std::cout << "done calc geodesci. " << std::endl;
	SampleNode();
	std::cout << "done sampling node" << std::endl;
	GenKnn();
	std::cout << "done generating knn" << std::endl;
	GenNodeNet();
	std::cout << "done generate node net" << std::endl;
}


void NodeGraphGenerator::LoadGeodesic(const std::string& filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open()) {
		std::cerr << "cannot open " << filename << std::endl;
		std::abort();
	}

	int m, n;
	std::pair<int, double> pair;
	ifs >> n;
	geodesic.resize(n);
	for (int i = 0; i < n; i++) {
		ifs >> m;
		for (int j = 0; j < m; j++) {
			ifs >> pair.first >> pair.second;
			geodesic[i].insert(pair);
		}
	}
	ifs.close();
}


void NodeGraphGenerator::SaveGeodesic(const std::string& filename) const
{
	std::ofstream ofs(filename);
	ofs << geodesic.size() << std::endl;
	for (const auto& geo : geodesic) {
		ofs << geo.size() << "\t";
		for (const auto& pair : geo)
			ofs << pair.first << "\t" << pair.second << "\t";
		ofs << std::endl;
	}
	ofs.close();
}


void NodeGraphGenerator::VisualizeNodeNet(const std::string& filename) const
{
	std::ofstream fs(filename);
	for (int i = 0; i < nodeIdx.size(); i++)
		fs << "v " << model.vertices(0, nodeIdx[i]) << " " <<
		model.vertices(1, nodeIdx[i]) << " " << model.vertices(2, nodeIdx[i]) << std::endl;

	for (int i = 0; i < nodeNet.cols(); i++)
		for (int j = 0; j < nodeNet.rows(); j++)
			if (nodeNet(j, i) != -1)
				fs << "l " << i + 1 << " " << nodeNet(j, i) + 1 << std::endl;
	fs.close();
}


void NodeGraphGenerator::VisualizeKnn(const std::string& filename) const
{
	std::ofstream fs(filename);
	for (int i = 0; i < knn.cols(); i++)
		fs << "v " << model.vertices(0, i) << " " << model.vertices(1, i) << " " << model.vertices(2, i) << std::endl;

	for (int i = 0; i < knn.cols(); i++)
		for (int j = 0; j < knn.rows(); j++)
			if (knn(j, i) != -1)
				fs << "l " << i + 1 << " " << nodeIdx[knn(j, i)] + 1 << std::endl;
	fs.close();
}
