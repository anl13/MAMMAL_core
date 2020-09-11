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

void FrameData::setConstDataToSolver(int id)
{
	assert((id >= 0 && id <= 3)); 

	if (mp_bodysolverdevice[id] == nullptr)
	{
		std::cout << "solver is empty! " << std::endl; 
		exit(-1); 
	}

	if (!mp_bodysolverdevice[id]->init_backgrounds)
	{
		for (int i = 0; i < 10; i++)
			cudaMemcpy(mp_bodysolverdevice[id]->d_const_scene_mask[i], m_scene_masks[i].data,
				1920 * 1080 * sizeof(uchar), cudaMemcpyHostToDevice);
		cudaMemcpy(mp_bodysolverdevice[id]->d_const_distort_mask, m_undist_mask.data,
			1920 * 1080 * sizeof(uchar), cudaMemcpyHostToDevice); 
		mp_bodysolverdevice[id]->init_backgrounds = true; 
	}
	mp_bodysolverdevice[id]->m_pig_id = id; 
	mp_bodysolverdevice[id]->m_det_masks = drawMask(); 

}
