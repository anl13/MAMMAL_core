/*
This file contains the overall pipeline control for 
our DARKOV system. 

maintainer: AN Liang. (al17@mails.tsinghua.edu.cn) 
date: 2020/11/08
*/
#include "framesolver.h"

#include "../utils/timer_util.h"

void FrameSolver::DARKOV_Step0_topdownassoc(bool isLoad)
{
	if (isLoad)
		load_clusters(); 
	else
	{
		if (m_frameid == m_startid)
			matching_by_tracking();
		else
			pureTracking(); 
	}

	save_clusters(); 
}

void FrameSolver::DARKOV_Step1_setsource()  // set source data to solvers 
{
	m_skels3d.resize(4); 
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]);
		mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist;
		mp_bodysolverdevice[i]->globalAlign();
		setConstDataToSolver(i);
		std::cout << "pig " << i << "  scale: " << mp_bodysolverdevice[i]->GetScale() << std::endl;
	}

}

void FrameSolver::DARKOV_Step2_loadanchor() // only load and set anchor id, without any fitting or align 
{
	std::string anchor_folder = m_anchor_folder;
	std::stringstream ss;
	ss << anchor_folder << "/anchor_" << std::setw(6) << std::setfill('0') << m_frameid <<
		".txt";
	std::ifstream infile(ss.str());
	for (int i = 0; i < 4; i++)
	{
		infile >> mp_bodysolverdevice[i]->m_anchor_id;
	}
	infile.close();
}

void FrameSolver::DARKOV_Step2_searchanchor()
{
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->searchAnchorSpace(); 
	}
}

void FrameSolver::DARKOV_Step2_optimanchor()
{
	// align given anchor Rotation and Translation 
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->optimizeAnchor(mp_bodysolverdevice[i]->m_anchor_id); 
	}
}

void FrameSolver::DARKOV_Step3_reassoc_type2() // type2 contains three small steps: find tracked, assign untracked, solve mix-up
{
	reAssocWithoutTracked();
}

void FrameSolver::DARKOV_Step3_reassoc_type1()
{
	reAssocProcessStep1();
}

void FrameSolver::DARKOV_Step4_fitrawsource()  // fit model to raw source
{
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->m_isReAssoc = false; 
	}
	if (m_solve_sil_iters > 0)
	{
		optimizeSilWithAnchor(m_solve_sil_iters); 
	}
}

void FrameSolver::DARKOV_Step4_fitreassoc()  // fit model to reassociated keypoints and silhouettes
{
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->m_isReAssoc = true; 
	}
	if (m_solve_sil_iters > 0)
	{
		optimizeSilWithAnchor(m_solve_sil_iters); 
	}
}

void FrameSolver::DARKOV_Step5_postprocess()  // some postprocessing step 
{
	for (int i = 0; i < 4; i++)
	{
		mp_bodysolverdevice[i]->postProcessing();
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host(); 
	}
	m_last_matched = m_matched; 
}