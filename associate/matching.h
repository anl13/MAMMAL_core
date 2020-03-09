#pragma once 

#include <iostream> 
#include <fstream> 
#include <iomanip>
#include <Eigen/Eigen> 
#include <json/json.h> 
#include <vector> 
#include <opencv2/opencv.hpp> 
#include "../utils/math_utils.h"
#include "../utils/camera.h"
#include "../utils/image_utils.h"
#include "../utils/geometry.h" 
#include "../utils/Hungarian.h"
#include "skel.h" 
#include "clusterclique.h"
#include "../smal/smal_jsolver.h"

using std::vector; 

// cross view matching by RANSAC (Point Level for robust triangulation)
struct ConcensusData{
    ConcensusData() {
        num = 0; 
        metric = 0; 
        X = Eigen::Vector3d::Zero(); 
    }
    std::vector<Camera> cams; 
    std::vector<int> ids; 
    std::vector<Eigen::Vector3d> joints2d; 
    Eigen::Vector3d X; 
    int num; 
    std::vector<double> errs; 
    double metric; 
}; 

bool equal_concensus(const ConcensusData& data1, const ConcensusData& data2); 
bool equal_concensus_list(std::vector<ConcensusData> data1, std::vector<ConcensusData> data2); 
bool compare_concensus(ConcensusData data1, ConcensusData data2);

// two kinds of nms needed: 
// 1. remove repeated detection 
// 2. remove repeated 3D skeletons
class EpipolarMatching
{
public: 
    void set_topo(const SkelTopology& _topo){m_topo = _topo;}
    void set_dets(const vector<vector<DetInstance > >& _dets){m_dets = _dets;}
    void set_cams(const vector<Camera>& _cams){m_cams = _cams;}
    void set_epi_type(std::string _epi_type){m_epi_type = _epi_type;}
    void set_epi_thres(double _epi_thres){m_epi_thres = _epi_thres;}
	void set_skels_t_1(vector<vector<Eigen::Vector3d> > _skels_t_1) { m_skels_t_1 = _skels_t_1; }
    void get_clusters(vector<vector<int> > &_clusters){_clusters=m_clusters;}
    void get_skels3d(vector<vector<Eigen::Vector3d> >&_skels3d){
        _skels3d.clear(); _skels3d=m_skels3d;}
    
	// single view matching 
    void match(); 
    void truncate(int _clusternum); 

	// spatio-temporal matching
    void match_by_tracking(); 

private: 
	/// t
    void epipolarSimilarity(); 
    void epipolarClustering();
    void compute3dRANSAC(); 
    void epipolarWholeBody(const Camera& cam1, const Camera& cam2, 
        const vector<Eigen::Vector3d>& pig1, const vector<Eigen::Vector3d>& pig2,
        double &avg_loss, int &matched_num);
    // input data 
    SkelTopology                              m_topo;
    vector<vector<DetInstance> >              m_dets; //[camid, candid]
    vector<Camera>                            m_cams; 
    std::string                               m_epi_type; 
    double                                    m_epi_thres; 
    // variables for clique clustering
    Eigen::MatrixXd                   m_G; 
    std::vector<std::pair<int,int> >  m_table;     // <camid, candid>
    std::vector<std::vector<int>>     m_inv_table; // inverse table. 
    int                               m_total_detection_num;
    // cluster output 
    std::vector< std::vector<int> >   m_cliques; // [cliqueid, vertexid] 
    vector<vector<int> >              m_clusters; // [clusterid, camid]
    vector<vector<Eigen::Vector3d> >  m_skels3d;  // [clusterid, jointnum]

    /// t-1 
    vector<vector<Eigen::Vector3d> >  m_skels_t_1; 
	vector<vector<DetInstance> >      m_dets_t_1; 
	vector<vector<int> >              m_clusters_t_1; 
	vector < vector<vector<Eigen::Vector3d> > > m_skels_proj_t_1; 
    void trackingSimilarity(); 
    void trackingClustering(); 
	void projectLoss(int pig_id, int camid,
        const DetInstance& det,double& mean_err, double& matched_num); 
	void project_all(); 
}; 

Eigen::Vector3d triangulate_ransac(
    const vector<Camera>& cams, const vector<Eigen::Vector3d>& xs,
    double sigma1=20, double sigma2=50); 





