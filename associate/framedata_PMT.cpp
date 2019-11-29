#include "framedata.h"

#include "parsing.h"

void FrameData::matching()
{
    m_matcher.set_cams(m_camsUndist); 
    m_matcher.set_dets(m_dets_undist); 
    m_matcher.set_epi_thres(m_epi_thres); 
    m_matcher.set_epi_type(m_epi_type); 
    m_matcher.set_topo(m_topo); 
    m_matcher.match(); // main match func 
    m_matcher.get_clusters(m_clusters); 
    m_matcher.get_skels3d(m_skels3d); 
}