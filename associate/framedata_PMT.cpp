#include "framedata.h"
#include "matching.h"
#include "tracking.h" 

void FrameData::matching()
{
    EpipolarMatching m_matcher; 
    m_matcher.set_cams(m_camsUndist); 
    m_matcher.set_dets(m_dets_undist); 
    m_matcher.set_epi_thres(m_epi_thres); 
    m_matcher.set_epi_type(m_epi_type); 
    m_matcher.set_topo(m_topo); 
    m_matcher.match(); // main match func 
    m_matcher.truncate(4); // retain only 4 clusters 
    m_matcher.get_clusters(m_clusters); 
    m_matcher.get_skels3d(m_skels3d); 
}

void FrameData::tracking()
{
    if(m_frameid == m_startid) {
        m_skels3d_last = m_skels3d; 
        return; 
    }
    NaiveTracker m_tracker; 
    m_tracker.set_skels_curr(m_skels3d); 
    m_tracker.set_skels_last(m_skels3d_last); 
    m_tracker.track(); 
    m_skels3d = m_tracker.get_skels_curr_track(); 

    m_skels3d_last = m_skels3d; 
}