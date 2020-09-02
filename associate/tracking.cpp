#include "tracking.h" 

 #define DEBUG_TRACK

double distBetweenSkel3D(const vector<Eigen::Vector3f>& S1, const vector<Eigen::Vector3f>& S2)
{
    if(S1.size()==0 || S2.size()==0) return 10000; 
    int overlay = 0; 
    float total_dist = 0; 
    for(int i = 0; i < S1.size(); i++)
    {
        if(S1[i].norm() < 0.00001 || S2[i].norm() < 0.00001) continue; 
        Eigen::Vector3f vec = S1[i] - S2[i];
        total_dist += vec.norm(); 
        overlay += 1; 
    }
    if(overlay == 0) return 10000; 
    return total_dist / pow(overlay, 2.0); 

}

void NaiveTracker::track()
{
    int last_num = m_skels_last.size(); 
    int curr_num = m_skels_curr.size(); 

    Eigen::MatrixXf G(last_num, curr_num); 
    for(int i = 0; i < last_num; i++)
    {
        for(int j = 0; j < curr_num; j++)
        {
            G(i,j) = distBetweenSkel3D(m_skels_last[i], m_skels_curr[j]); 
        }
    }
#ifdef DEBUG_TRACK
    std::cout << G << std::endl; 
#endif 
    std::vector<int> assign = solveHungarian(G); 
    m_map = assign; 
    m_skels_curr_track.clear(); 
    for(int i = 0; i < last_num; i++)
    {
        int id = assign[i]; 
        if(id > -1) 
        {
            m_skels_curr_track.push_back(m_skels_curr[id]); 
        }
    }


}