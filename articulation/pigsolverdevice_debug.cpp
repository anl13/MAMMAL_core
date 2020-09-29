#include "pigsolverdevice.h"

void PigSolverDevice::debug_source_visualize(std::string folder, int frameid)
{
	std::vector<Eigen::Vector3i> m_CM;
	getColorMap("anliang_rgb", m_CM); 
	std::vector<cv::Mat> m_imgdet; 
	cloneImgs(m_rawimgs, m_imgdet); 
	for (int i = 0; i < m_source.view_ids.size(); i++)
	{
		int camid = m_source.view_ids[i];
		drawSkelDebug(m_imgdet[camid], m_source.dets[i].keypoints, m_skelTopo);
		my_draw_box(m_imgdet[camid], m_source.dets[i].box, m_CM[m_pig_id]);
		//my_draw_mask(m_imgsDetect[camid], m_matched[id].dets[i].mask, m_CM[id], 0.5);
	}


	std::vector<cv::Mat> crop_list;
	for (int i = 0; i < m_source.dets.size(); i++)
	{
		Eigen::Vector4f box = m_source.dets[i].box;
		int camid = m_source.view_ids[i];
		cv::Mat raw_img = m_imgdet[camid];
		cv::Rect2i roi(box[0], box[1], box[2] - box[0], box[3] - box[1]);
		cv::Mat img = raw_img(roi);
		cv::Mat img2 = resizeAndPadding(img, 256, 256);
		crop_list.push_back(img2);
	}
	cv::Mat output;
	if (crop_list.size() > 0)
	{
		packImgBlock(crop_list, output);
		std::stringstream ss;
		ss << folder << "/fitting/" << m_pig_id << "/det_" << std::setw(6) << std::setfill('0') << frameid << ".png";
		cv::imwrite(ss.str(), output);
		//cv::imshow("test", output); 
		//cv::waitKey(); 
	}
	return;
}