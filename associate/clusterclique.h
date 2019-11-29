/*
This file implement assocation method with MCE(maximal clique enumeration) method. 
Author: AN Liang 
Date  : 2018-12-04
Recently modified: 2019-09-19
Reference:  https://github.com/darrenstrash/quick-cliques 
*/
#pragma once 

#include <iostream> 
#include <iomanip> 
#include <vector> 
#include <Eigen/Eigen>
#include "mce/Algorithm.h" 
#include "mce/AdjacencyListAlgorithm.h" 

/*
Cluster on K-partite graph. 
*/ 
class ClusterClique
{
public: 
    ClusterClique(); 
    void constructAdjacentGraph(); 
    Eigen::MatrixXd G; // distance graph 
    std::vector<std::pair<int,int>> table;  // vertex table; [partite, candidate]
    std::vector<std::vector<int>> invTable; // inverse table. 
    int vertexNum; 
    double threshold; 
    std::vector< std::vector<int> > cliques;  

    void enumerateMaximalCliques(); 
    void enumerateBestCliques(); 

    void printGraph(); 
    void printAllMC(); 
    void printCliques(); 
    void convertListToVec(const std::list< std::list<int> >& cliqueList, std::vector< std::vector<int> > &cliqueVec); 

private: 
    void eliminateClique(const std::vector<int> &clique); 
    std::vector<int> findBestClique();      // find clique with minimum weights 
    double getCliqueEnergyAvg(const std::vector<int> &clique); 
    double getCliqueEnergyTotal(const std::vector<int> &clique); 
    std::vector<bool> used;                 // record vertex in current cliques 
    int edgeNum; 
    Algorithm* pAlgorithm; 
    std::vector< std::vector<int> > adjacencyArray; 
    std::vector< std::vector<int> > adjacencyArrayCopy; 
    std::vector< std::vector<int> > allMC; 

}; 
