#pragma once
#include <Eigen/Core>
#include "nanoflann.hpp"
#include <memory>

template <typename T>
class KDTree
{
public:
	KDTree(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& _mat)
		: KDTree(std::make_shared<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(_mat)) {}

	KDTree(std::shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _mat)
		:m_mat(_mat), m_adaptor(m_mat),m_tree(int(m_mat->rows()), m_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams())
	{
		m_tree.buildIndex();
	}

	~KDTree() = default;
	KDTree(const KDTree& _) = delete;
	KDTree& operator=(const KDTree& _) = delete;

	std::vector<std::pair<T, size_t>> KNNSearch(const Eigen::Matrix<T, Eigen::Dynamic,1>& p, const int& cnt) const
	{
		nanoflann::KNNResultSet<T> resultSet(cnt);
		std::vector<T> dists(cnt);
		std::vector<size_t> ids(cnt);
		resultSet.init(&ids[0], &dists[0]);
		m_tree.findNeighbors(resultSet, p.data(), nanoflann::SearchParams());
		std::vector<std::pair<T, size_t>> result(cnt);
		for (int i = 0; i < cnt; i++)
			result[i] = std::pair<T, size_t>(std::sqrt(dists[i]), ids[i]);
		return result;
	}

private:
	template <typename T>
	struct EigenAdaptor
	{
		std::shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> mat;

		EigenAdaptor(std::shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _mat) :mat(_mat) {}
		inline size_t kdtree_get_point_count() const { return mat->cols(); }
		inline double kdtree_get_pt(const size_t idx, const size_t d) const { return (*mat)(d, idx); }

		template <class BBOX>
		bool kdtree_get_bbox(BBOX&) const { return false; }
	};

	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<T, EigenAdaptor<T>>, EigenAdaptor<T>> KDTreeType;
	std::shared_ptr<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> m_mat;
	EigenAdaptor<T> m_adaptor;
	KDTreeType m_tree;
};

