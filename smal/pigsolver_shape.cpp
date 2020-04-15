#include "pigsolver.h"
#include "../utils/image_utils.h"

void PigSolver::feedData(
	const ROIdescripter& _roi,
	const BodyState& _body
)
{
	m_rois.push_back(_roi); 
	m_bodies.push_back(_body);
}

void PigSolver::feedRender(
	const cv::Mat& _render
)
{
	m_renders.push_back(_render);
}

void PigSolver::iterateStep(int iter)
{
	double D = 0.001; // 1 mm per iter 
	// prepare chamfer 
	vector<cv::Mat> currentChamfers; 
	for (int i = 0; i < m_renders.size(); i++)
	{
		cv::Mat chamfer = get_dist_trans(m_renders[i]);
		currentChamfers.push_back(chamfer); 
	}

	cv::Mat packChamfer;
	packImgBlock(currentChamfers, packChamfer); 
	packChamfer = reverseChamfer(packChamfer);
	cv::Mat visPackChamfer = vis_float_image(packChamfer); 
	std::stringstream ss;
	ss << "E:/debug_pig2/chamfer/source_" << iter << ".png";
	cv::imwrite(ss.str(), visPackChamfer); 

	// currently, assu me only one frame used 
	//m_poseParam = m_bodies[0].pose;
	//m_translation = m_bodies[0].trans;
	//UpdateVertices();
	// compute gradient
	Eigen::VectorXd gradient; 
	gradient.resize(m_vertexNum); 
	gradient.setZero(); 

	cv::Mat packTargetChamfer; 
	std::vector<cv::Mat> targets; 
	for (int oid = 0; oid < m_rois.size(); oid++)
	{
		ROIdescripter& roi = m_rois[oid];
		cv::Mat&       chamfer = currentChamfers[oid];
		targets.push_back(roi.chamfer);
		for (int i = 0; i < m_vertexNum; i++)
		{
			if (m_bodyParts[i] != MAIN_BODY) {
				continue; // only optimize main body
			}
			double g = 0; 
			Eigen::Vector3d p = m_verticesFinal.col(i);
			float f = queryPixel(chamfer, p, roi.cam);
			float t = roi.queryChamfer(p); 
			if (t < 1)
			{
				g = D; 
			}
			else
			{
				//if (f < 5 && f < t)
				//{
				//	g = -D/2; 
				//}
			}
			gradient(i) += g; 
		}
	}
	packImgBlock(targets, packTargetChamfer); 
	packTargetChamfer = reverseChamfer(packTargetChamfer); 
	std::stringstream ss1; 
	ss1 << "E:/debug_pig/chamfer/target_" << iter << ".png";
	cv::Mat visTarget = vis_float_image(packTargetChamfer); 
	cv::imwrite(ss1.str(), visTarget); 

	m_deform -= gradient; 

	double r = gradient.norm(); 
	double r1 = m_deform.norm();

	std::cout << "r: " << r << std::endl; 
	std::cout << "d: " << r1 << std::endl;
}

void PigSolver::NaiveNodeDeformStep(int iter)
{
	double D = 0.001; // 1 mm per iter 
					  // prepare chamfer 
	vector<cv::Mat> currentChamfers;
	for (int i = 0; i < m_renders.size(); i++)
	{
		cv::Mat chamfer = get_dist_trans(m_renders[i]);
		currentChamfers.push_back(chamfer);
	}
	// compute gradient
	UpdateNormals(); 
	std::vector<cv::Mat> targets;
	for (int oid = 0; oid < m_rois.size(); oid++)
	{
		ROIdescripter& roi = m_rois[oid];
		cv::Mat&       chamfer = currentChamfers[oid];
		targets.push_back(roi.chamfer);
		for (int i = 0; i < mp_nodeGraph->nodeIdx.size(); i++)
		{
			int vid = mp_nodeGraph->nodeIdx[i];
			if (m_bodyParts[vid] != MAIN_BODY) {
				continue; // only optimize main body
			}
			double g = 0;
			Eigen::Vector3d p = m_verticesFinal.col(vid);
			Eigen::Vector3d n = m_normalShaped.col(vid);
			float f = queryPixel(chamfer, p, roi.cam);
			float t = roi.queryChamfer(p);
			if (t < 1)
			{
				m_warpField.block<3, 1>(0, 4 * i + 3) -= n*D;
			}
			else
			{
				//if (f < 5 && f < t)
				//{
				//	g = -D/2; 
				//}
			}
		}
	}
}

void PigSolver::clearData()
{
	m_rois.clear(); 
	m_bodies.clear(); 
	m_renders.clear(); 
}

void PigSolver::optimizeShapeToBoneLength(int maxIter, double terminal)
{
	Eigen::MatrixXd JSkel = CalcShapeJacobiToSkel();
	std::vector<std::pair<int, int> > rigid_bones =
	{
		{ 0,1 },{ 0,2 },{ 1,2 },{ 1,3 },{ 2,4 },{ 3,4 },
	{ 5,6 },{ 5,7 },{ 7,9 },{ 6,8 },{ 8,10 },{ 20,21 },
	{ 21,11 },{ 21,12 },{ 11,12 },{ 11,13 },{ 13,15 },
	{ 12,14 },{ 14,16 }
	};
	Eigen::MatrixXd ATA(m_shapeNum, m_shapeNum);
	Eigen::VectorXd ATb(m_shapeNum);
	CalcZ(); 

	for (int iter = 0; iter < maxIter; iter++) 
	{
		Eigen::MatrixXd tpose = getRegressedSkelTPose(); 
		ATA.setZero();
		ATb.setZero();
		for (int i = 1; i < rigid_bones.size(); i++) 
		{
			int a = rigid_bones[i].first;
			int b = rigid_bones[i].second;
			if (Z.col(a).norm() < 0.00001 || Z.col(b).norm() < 0.00001) continue;
			if (tpose.col(a).norm() < 0.00001 || tpose.col(b).norm() < 0.00001)continue;
			Eigen::Vector3d unit_vec = tpose.col(a) - tpose.col(b); 
			unit_vec.normalize();
			double length = (Z.col(a) - Z.col(b)).norm();
			Eigen::Vector3d source = unit_vec * length; 
			Eigen::Vector3d target = Z.col(a) - Z.col(b);
			if (length > 0.5) continue; 
			double ww = length;
			Eigen::Vector3d residual = - source + target;
			Eigen::MatrixXd jacobi = JSkel.middleRows(3 * a, 3) - JSkel.middleRows(3*b,3);
			ATA += ww * jacobi.transpose() * jacobi;
			ATb += ww * jacobi.transpose() * residual;
		}

		// prior
		ATA += 0.01 * Eigen::MatrixXd::Identity(m_shapeNum,m_shapeNum);
		ATb -= 0.01 * m_shapeParam;

		Eigen::VectorXd delta = ATA.ldlt().solve(ATb);
		m_shapeParam += delta;

		if (delta.norm() < terminal)
			break;
	}
}

void PigSolver::CalcSmthTerm(
	Eigen::SparseMatrix<double>& ATA,
	Eigen::VectorXd& ATb
)
{

}

void PigSolver::CalcVertJacobiNode()
{

}
