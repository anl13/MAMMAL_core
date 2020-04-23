#include "framedata.h"


vector<ROIdescripter> FrameData::getROI(int id)
{
	//assert(id >= 0 && id <= 3);
	// This function is run after solving single
	// frame pose 
	vector<ROIdescripter> rois; 
	for (int view = 0; view < m_matched[id].view_ids.size(); view++)
	{
		int camid = m_matched[id].view_ids[view];
		ROIdescripter roi;
		roi.setId(id);
		roi.setT(m_frameid);
		roi.setCam(m_camsUndist[camid]);
		roi.mask_list = m_matched[id].dets[view].mask;
		roi.mask_norm = m_matched[id].dets[view].mask_norm;
		cv::Mat chamfer, mask; 
		getChamferMap(id, view, chamfer, mask);
		//roi.setData(chamfer, mask, m_matched[id].dets[view].box);
		roi.chamfer = chamfer; 
		roi.mask = mask; 
		roi.box = m_matched[id].dets[view].box;
		rois.push_back(roi);
		//cv::namedWindow("debug_mask", cv::WINDOW_NORMAL); 
		//cv::imshow("debug_mask", mask); 
		//cv::Mat chamfer_vis = vis_float_image(chamfer); 
		//cv::namedWindow("debug_chamfer", cv::WINDOW_NORMAL); 
		//cv::imshow("debug_chamfer", chamfer_vis); 
		//int key = cv::waitKey(); 
		//if (key == 27) exit(-1); 
	}
	return rois; 
}
