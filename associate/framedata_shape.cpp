#include "framedata.h"
#include "../utils/timer_util.h"

void FrameData::getROI(std::vector<ROIdescripter>& rois, int id)
{
	std::vector<cv::Mat> masks = drawMask();
	rois.resize(m_matched[id].view_ids.size()); 
	for (int view = 0; view < m_matched[id].view_ids.size(); view++)
	{
		int camid = m_matched[id].view_ids[view];
		rois[view].setId(id);
		rois[view].setT(m_frameid);
		rois[view].viewid = camid;
		rois[view].setCam(m_camsUndist[camid]);
		rois[view].mask_list = m_matched[id].dets[view].mask;
		rois[view].mask_norm = m_matched[id].dets[view].mask_norm;
		rois[view].keypoints = m_matched[id].dets[view].keypoints;
		cv::Mat mask;
		mask.create(cv::Size(m_imw, m_imh), CV_8UC1);
		my_draw_mask_gray(mask,
			m_matched[id].dets[view].mask, 255);
		rois[view].area = cv::countNonZero(mask);

		rois[view].chamfer = computeSDF2d(mask);

		rois[view].mask = masks[camid];
		rois[view].box = m_matched[id].dets[view].box;
		rois[view].undist_mask = m_undist_mask; // valid area for image distortion 
		rois[view].scene_mask = m_scene_masks[camid];
		rois[view].pid = id;
		rois[view].idcode = 1 << id;
		rois[view].valid = rois[view].keypointsMaskOverlay();
		computeGradient(rois[view].chamfer, rois[view].gradx, rois[view].grady);
	}
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