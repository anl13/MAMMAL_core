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
	double D = 0.001; // 1 cm per iter 
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
	ss << "E:/debug_pig/chamfer/source_" << iter << ".png";
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
			if (m_bodyParts[i] == OTHERS) {
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

void PigSolver::clearData()
{
	m_rois.clear(); 
	m_bodies.clear(); 
	m_renders.clear(); 

}