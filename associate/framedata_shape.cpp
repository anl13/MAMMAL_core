#include "framedata.h"

vector<ROIdescripter> FrameData::getROI(int id)
{
	// assert(id >= 0 && id <= 3);
	// This function is run after solving single
	// frame pose 
	vector<cv::Mat> masks = drawMask();
	vector<ROIdescripter> rois; 
	for (int view = 0; view < m_matched[id].view_ids.size(); view++)
	{
		int camid = m_matched[id].view_ids[view];
		ROIdescripter roi;
		roi.setId(id);
		roi.setT(m_frameid);
		roi.viewid = camid;
		roi.setCam(m_camsUndist[camid]);
		roi.mask_list = m_matched[id].dets[view].mask;
		roi.mask_norm = m_matched[id].dets[view].mask_norm;
		roi.keypoints = m_matched[id].dets[view].keypoints;
		cv::Mat mask;
		mask.create(cv::Size(m_imw, m_imh), CV_8UC1);
		my_draw_mask_gray(mask,
			m_matched[id].dets[view].mask, 255);
		roi.area = cv::countNonZero(mask);

		roi.chamfer = computeSDF2d(mask); 
		roi.mask = masks[camid]; 
		roi.box = m_matched[id].dets[view].box;
		roi.undist_mask = m_undist_mask; // valid area for image distortion 
		roi.scene_mask = m_scene_masks[camid];
		roi.pid = id;
		roi.idcode = 1 << id; 
		roi.valid = roi.keypointsMaskOverlay(); 
		computeGradient(roi.chamfer, roi.gradx, roi.grady);
		rois.push_back(roi);
		
	}
	// debug 
	//cv::Mat mask = masks[0];
	//mask = mask * 128;
	//cv::imshow("mask", mask);
	//cv::waitKey();
	//exit(-1);
	return rois; 
}

void FrameData::extractFG()
{
	m_foreground.resize(m_camNum);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		cv::Mat& bg = m_backgrounds[camid];
		cv::Mat& full = m_imgsUndist[camid];

		m_foreground[camid] = my_background_substraction(full,bg);
	}
}