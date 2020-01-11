#pragma once 
#include <vector>
#include <string> 
#include <Eigen/Eigen>
#include <iostream> 
#include "../utils/Hungarian.h"
#include "../utils/math_utils.h"

using std::vector; 

class NaiveTracker
{
public: 
    inline void set_skels_last(const vector<vector<Eigen::Vector3d> >& _skel) {
        m_skels_last = _skel; 
    }
    inline void set_skels_curr(const vector<vector<Eigen::Vector3d> >& _skel) {
        m_skels_curr = _skel; 
    }
    inline vector<int> get_map() {return m_map;}
    inline vector<vector<Eigen::Vector3d> > get_skels_curr_track(){return m_skels_curr_track;}
    void track();     
private: 
    vector<vector<Eigen::Vector3d> > m_skels_last; 
    vector<vector<Eigen::Vector3d> > m_skels_curr; 
    vector<vector<Eigen::Vector3d> > m_skels_curr_track; 
    vector<int> m_map;
};