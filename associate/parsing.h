#pragma once 

#include <iostream> 
#include <fstream> 
#include <iomanip>
#include <Eigen/Eigen> 
#include <json/json.h> 
#include <vector> 
#include <opencv2/opencv.hpp> 
#include "math_utils.h"
#include "camera.h"
#include "image_utils.h"
#include "geometry.h" 
#include "clusterclique.h"
#include "Hungarian.h"
#include "skel.h" 

#include "matching.h" 
#include "clusterclique.h"

/*
20191127: such parsing method fails to discriminate pigs 
who are very close to each other. 

Description: This method project all 3D concensus proposals (joint) to each view. 
If projections of two points of different type are in a same bbox, 
their connnection confidence increase by 1. 
Then, for the whole connection graph defined on all 3D concensus proposals, 
we extract maximal subgraph, which corresponds to a parsing. 
Such method could be seen as a kind of clustering, which works well for non-close interaction 
case or non-severe-occlusion case. 
*/
class ParsingPool
{
public: 
    // input 
    std::vector<std::vector<ConcensusData> > rawData; // [kpt_id, cand_id]
    std::vector<std::vector<Eigen::Vector4d> > boxes; // [camid, cand_id]
    std::vector<Camera> camsUndist; 
    double pruneThreshold; 
    int cliqueSizeThreshold; 
    std::vector<PIG_SKEL> skels; 
    // output 

    // methods 
    void constructGraph(); // output m_G 
    void solveGraph();     // output cliques 
    void mappingToSkels(); // output skels 
    void test(); 

private: 
    Eigen::MatrixXd m_G; 
    std::vector<std::pair<int,int> > m_table;  // vertex table; [partite, candidate]
    std::vector<std::vector<int>> m_invTable; // inverse table. 
    int m_vertexNum; 
    std::vector< std::vector<int> > m_cliques; 
};

// return: 
//         0: projections of x and y not in the same box
//         1: in the same box 
int checkBothInBox(Eigen::Vector3d x, Eigen::Vector3d y, Eigen::Vector4d box, Camera cam); 
