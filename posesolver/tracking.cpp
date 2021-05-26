#include "tracking.h" 

 #define DEBUG_TRACK

void NaiveTracker::track(
	const vector<MatchedInstance>& cands
)
{
    int last_num = m_skels_last.size(); 
	int curr_num = cands.size(); 
	if (last_num == 1 && curr_num == 1)
	{
		m_map.resize(1);
		m_map[0] = 0; 
		return; 
	}

    Eigen::MatrixXf G(last_num, curr_num); 
    for(int i = 0; i < last_num; i++)
    {
        for(int j = 0; j < curr_num; j++)
        {
            G(i,j) = distBetween3DSkelAnd2DDet(m_skels_last[i], cands[j], m_cameras, m_topo); 
        }
    }
#ifdef DEBUG_TRACK
	std::cout << "NaiveTracker::track(): " << std::endl; 
    std::cout << G << std::endl; 
#endif 
    std::vector<int> assign = solveHungarian(G); 
    m_map = assign; 
}