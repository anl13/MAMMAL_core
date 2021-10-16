/*
This file contains the overall pipeline control for 
our DARKOV system. 

maintainer: AN Liang. (al17@mails.tsinghua.edu.cn) 
date: 2020/11/08
*/
#include "framesolver.h"

#include "../utils/timer_util.h"

//#define DEBUG_SIL


void FrameSolver::DARKOV_Step0_topdownassoc(bool isLoad)
{
	if (isLoad)
		load_clusters(); 
	else
	{
		if (m_frameid == m_startid)
			matching_by_tracking();
		else
		{
			pureTracking();
			updateTrackConf(); 
			restart(); 
		}
	}

	save_clusters(); 
}

void FrameSolver::DARKOV_Step1_setsource()  // set source data to solvers 
{
	m_skels3d.resize(m_pignum); 
	setConstDataToSolver();

#ifdef USE_SIFT
	readSIFTandTrack(); 
#endif

	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->setSource(m_matched[i]);
		mp_bodysolverdevice[i]->m_rawimgs = m_imgsUndist;
		if (!m_use_init_pose)
		{
			mp_bodysolverdevice[i]->globalAlign();
		}
#ifdef USE_SIFT
		mp_bodysolverdevice[i]->m_siftCorrs = m_siftCorrs[i];
#endif
	}
}

void FrameSolver::DARKOV_Step2_loadanchor() // only load and set anchor id, without any fitting or align 
{
	std::string anchor_folder = m_anchor_folder;
	std::stringstream ss;
	ss << anchor_folder << "/anchor_" << std::setw(6) << std::setfill('0') << m_frameid <<
		".txt";
	std::ifstream infile(ss.str());
	for (int i = 0; i < m_pignum; i++)
	{
		infile >> mp_bodysolverdevice[i]->m_anchor_id;
	}
	infile.close();
}

void FrameSolver::DARKOV_Step2_searchanchor(int pid)
{
	mp_bodysolverdevice[pid]->searchAnchorSpace(); 
}

void FrameSolver::DARKOV_Step2_optimanchor(int pid)
{
	// align given anchor Rotation and Translation 
	mp_bodysolverdevice[pid]->optimizeAnchor(mp_bodysolverdevice[pid]->m_anchor_id); 
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
#ifdef DEBUG_SIL
	renderInteractDepth(true); 
	cv::Mat output;
	packImgBlock(m_interMask, output);
	std::stringstream ss;
	ss << m_result_folder << "/debug/mask_" << m_frameid << "_anchor_" << ".png";
	cv::imwrite(ss.str(), output);
#endif 
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->m_isReAssoc = false; 
		mp_bodysolverdevice[i]->generateDataForSilSolver();
	}

	//std::vector<std::vector<int> > hierarachy = 
	//{
	//	{0,1,2,3,4,21,22,23},    // main body
	//	//{21,22,23},     // neck
	//	{13,14,15,16},  // left front leg
	//	{5,6,7,8},      // right front leg
	//	{54,55,56,57},  // left back leg
	//	{38,39,40,41}   // right back leg
	//};

	//std::vector<std::vector<int> > hierarachy =
	//{
	//	{0,1,2,3,4, 21,22,23},    // main body
	//	{13,14,15,16, 5,6,7,8,54,55,56,57,38,39,40,41}
	//};

	std::vector<std::vector<int> > hierarachy =
	{
		{0,1,2,3,4, 21,22,23,13,14,15,16, 5,6,7,8,54,55,56,57,38,39,40,41}
	};

	TimerUtil::Timer<std::chrono::microseconds> tt;
	tt.Start();

	for (int k = 0; k < hierarachy.size(); k++)
	{
		for (int pid = 0; pid < m_pignum; pid++)
		{
			mp_bodysolverdevice[pid]->m_currentHierarchy = hierarachy[k];
		}
		if (m_solve_sil_iters > 0)
		{
#ifdef DEBUG_SIL 
			cv::Mat output;
			packImgBlock(m_interMask, output);
			std::stringstream ss;
			ss << m_result_folder << "/debug/mask_" << m_frameid << "_before_optim_" << ".png";
			cv::imwrite(ss.str(), output);
			for (int pid = 0; pid < 4; pid++)
			{
				std::cout << "pig " << pid << " : " << std::endl << mp_bodysolverdevice[pid]->GetScale() << std::endl; 
			}
#endif 
			optimizeSilWithAnchor(m_solve_sil_iters,0);
		}
	}
	float micro = tt.Elapsed();	
	std::cout << "raw fit: " << micro / 1000.0 << " ms; " 
		<< micro / 1000. / (hierarachy.size()) << " ms per hierarchy" 
		<< std::endl; 
}

void FrameSolver::DARKOV_Step4_fitreassoc()  // fit model to reassociated keypoints and silhouettes
{
	if (!m_use_reassoc) return; 
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->m_isReAssoc = true; 
	}
	std::vector<std::vector<int> > hierarachy =
	{
		{0,1,2,3,4, 21,22,23,13,14,15,16, 5,6,7,8,54,55,56,57,38,39,40,41}
	};

	TimerUtil::Timer<std::chrono::microseconds> tt;
	tt.Start();

	for (int k = 0; k < hierarachy.size(); k++)
	{
		for (int pid = 0; pid < m_pignum; pid++)
		{
			mp_bodysolverdevice[pid]->m_currentHierarchy = hierarachy[k];
		}
		if (m_solve_sil_iters > 0)
		{
			optimizeSilWithAnchor(m_solve_sil_iters_2nd_phase, m_solve_sil_iters);
		}
	}

	float micro = tt.Elapsed();
	std::cout << "rea fit: " << micro / 1000.0 << " ms; "
		<< micro / 1000. / (hierarachy.size()) << " ms per hierarchy"
		<< std::endl;
}

void FrameSolver::DARKOV_Step5_postprocess()  // some postprocessing step 
{
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->postProcessing();
		m_skels3d[i] = mp_bodysolverdevice[i]->getRegressedSkel_host(); 
	}
	m_last_matched = m_matched; 

#ifdef USE_SIFT
	//TimerUtil::Timer<std::chrono::microseconds> tt;
	//tt.Start();
	buildSIFTMapToSurface(); 
	//std::cout << "buildSIFTMapToSurface(): " << tt.Elapsed() / 1000.0 << " ms" << std::endl; 
#endif 
}

void FrameSolver::DirectTriangulation()
{
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->optimizeTri(); 
		m_skels3d[i] = mp_bodysolverdevice[i]->getSkel3D();
	}
	m_last_matched = m_matched; 
}