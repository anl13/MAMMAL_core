#pragma once 
#include <vector>
#include <string> 
#include <Eigen/Eigen>
#include <iostream> 
#include "../utils/hungarian.h"
#include "../utils/math_utils.h"
#include "../utils/image_utils.h" 
#include "../utils/geometry.h" 
#include "../utils/skel.h"

using std::vector; 

class NaiveTracker
{
public: 
    inline void set_skels_last(const vector<vector<Eigen::Vector3f> >& _skel) {
        m_skels_last = _skel; 
    }
    //inline void set_skels_curr(const vector<vector<Eigen::Vector3f> >& _skel) {
    //    m_skels_curr = _skel; 
    //}
    std::vector<int> get_map(){return m_map; }
    //inline vector<vector<Eigen::Vector3f> > get_skels_curr_track(){return m_skels_curr_track;}
    void track(const vector<MatchedInstance>& cands);     


    vector<vector<Eigen::Vector3f> > m_skels_last; 
    //vector<vector<Eigen::Vector3f> > m_skels_curr; 
    //vector<vector<Eigen::Vector3f> > m_skels_curr_track; 
    vector<int> m_map; 
	vector<Camera> m_cameras;
	SkelTopology m_topo; 
};