#include "tracking.h" 

 #define DEBUG_TRACK

float distBetweenSkel3D(const vector<Eigen::Vector3f>& S1, const vector<Eigen::Vector3f>& S2)
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
	// 2020 09 18: change 2.0 to 1.0
    return total_dist / pow(overlay, 1.0f); 

}

float distBetween3DSkelAnd2DDet(const vector<Eigen::Vector3f>& skel3d,
	const MatchedInstance& det, const vector<Camera>& cams, const SkelTopology& topo
)
{
	float dist_all_views = 0; 
	for (int view = 0; view < det.view_ids.size(); view++)
	{
		int camid = det.view_ids[view];
		Camera cam = cams[camid];
		std::vector<Eigen::Vector3f> points2d; 
		project(cam, skel3d, points2d); 
		int valid_num = 0; 
		float total_dist = 0; 
		for (int i = 0; i < points2d.size(); i++)
		{
			if (det.dets[view].keypoints[i](2) < topo.kpt_conf_thresh[i]) continue; 
			Eigen::Vector3f diff = points2d[i] - det.dets[view].keypoints[i];
			total_dist += diff.norm(); 
			valid_num += 1; 
		}
		dist_all_views += total_dist / valid_num;
	}
	dist_all_views /= det.view_ids.size(); 
	return dist_all_views; 
}

void NaiveTracker::track(
	const vector<MatchedInstance>& cands
)
{
    int last_num = m_skels_last.size(); 
	int curr_num = cands.size(); 

	assert(last_num == 4); 
	assert(curr_num == 4); 

    Eigen::MatrixXf G(last_num, curr_num); 
    for(int i = 0; i < last_num; i++)
    {
        for(int j = 0; j < curr_num; j++)
        {
            G(i,j) = distBetween3DSkelAnd2DDet(m_skels_last[i], cands[j], m_cameras, m_topo); 
        }
    }
//#ifdef DEBUG_TRACK
//	std::cout << "NaiveTracker::track(): " << std::endl; 
//    std::cout << G << std::endl; 
//#endif 
    std::vector<int> assign = solveHungarian(G); 
    m_map = assign; 
}