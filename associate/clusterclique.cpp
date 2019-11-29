#include "clusterclique.h" 
#include <algorithm> 

static std::list< std::list<int> > globalCliques;

ClusterClique::ClusterClique()
{
    vertexNum = 0; 
    edgeNum = 0; 
    threshold = 0; 
}

void ClusterClique::constructAdjacentGraph()
{
    adjacencyArray.resize(vertexNum); 
    used.resize(vertexNum, false); 

    for(int i = 0; i < vertexNum; i++)
    {
        for(int j=i+1; j < vertexNum; j++)
        {
            if(G(i,j) >= 0 && G(i,j)< threshold)
            {
                adjacencyArray[i].push_back(j); 
                adjacencyArray[j].push_back(i); 
            }
        }
    }
    adjacencyArrayCopy = adjacencyArray; 
}

void ClusterClique::enumerateMaximalCliques()
{
    Algorithm *pAlgorithm = new AdjacencyListAlgorithm(adjacencyArray);
    pAlgorithm->SetQuiet(true);
    globalCliques.clear(); 
    
    #ifdef RETURN_CLIQUES_ONE_BY_ONE
    auto storeCliqueInList = [&](std::list<int> const &clique) {
        ::globalCliques.push_back(clique);
    };
    pAlgorithm->AddCallBack(storeCliqueInList);
    #endif //RETURN_CLIQUES_ONE_BY_ONE
    
    pAlgorithm->Run(globalCliques); 
    convertListToVec(globalCliques, allMC); 

    delete pAlgorithm; 
}

double ClusterClique::getCliqueEnergyAvg(const std::vector<int> &clique)
{
    double energy = 0; 
    int edges = 0; 
    for(int i = 0; i < clique.size(); i++)
    {
        for(int j = i+1; j < clique.size(); j++)
        {
            int vid1 = clique[i]; 
            int vid2 = clique[j]; 
            double sim1 = G(vid1, vid2); 
            energy += sim1; 
            edges ++; 
        }
    }
    return energy / edges; 
}

double ClusterClique::getCliqueEnergyTotal(const std::vector<int> &clique)
{
    double energy = 0; 
    for(int i = 0; i < clique.size(); i++)
    {
        for(int j = i+1; j < clique.size(); j++)
        {
            int vid1 = clique[i]; 
            int vid2 = clique[j]; 
            double sim1 = G(vid1, vid2); 
            energy += sim1; 
        }
    }
    return energy;
}

void ClusterClique::eliminateClique(const std::vector<int> &clique)
{
    for(int i = 0; i < clique.size(); i++)
    {
        int vid = clique[i]; // for each vertex in clique 
        used[vid] = true; 
        for(int j = 0; j < adjacencyArrayCopy[vid].size(); j++) // for each neighbour j of vid
        {
            int nid = adjacencyArrayCopy[vid][j]; // neighbour id; 
            for(auto iter = adjacencyArray[nid].begin(); iter!=adjacencyArray[nid].end(); ) // clear vertex vid from j's neighbours 
            {
                // if(*iter == clique)
                if(*iter == vid) iter = adjacencyArray[nid].erase(iter); 
                else iter++; 
            }
        }
        adjacencyArray[vid].clear(); // clear all vid's neighbours
    }
}

std::vector<int> ClusterClique::findBestClique()
{
    std::vector<int> emptyClique; 
    int minEnergyId = -1; 
    int maxSize = 0; 
    float minEnergy = 10000; 
    for(int i = 0; i < allMC.size(); i++)
    {
        if(maxSize < allMC[i].size()) 
        {
            maxSize = allMC[i].size(); 
        }
    }
    if(maxSize == 1) return emptyClique; 
    for(int i = 0; i < allMC.size(); i++)
    {
        if(allMC[i].size() < maxSize) continue;
        float energy = getCliqueEnergyAvg(allMC[i]);
        if(energy < minEnergy) 
        {
            minEnergy = energy; 
            minEnergyId = i; 
        }
    }
    if(minEnergyId > -1)
    {
        return allMC[minEnergyId]; 
    }
    else return emptyClique; 
}

void ClusterClique::enumerateBestCliques()
{
    while(true)
    {
        enumerateMaximalCliques();  // compute allMC

        std::vector<int> bestClique = findBestClique(); 

        if(bestClique.size() == 0)
        {
            for(int vid = 0; vid < vertexNum; vid++)
            {
                if(!used[vid]) 
                {
                    std::vector<int> singleNode;
                    singleNode.push_back(vid); 
                    cliques.push_back(singleNode); 
                }
            }
            
            break;  
        }
        else 
        {
            cliques.push_back(bestClique);
            eliminateClique(bestClique);  

            #ifdef DEBUG_CLIQUE
            std::cout << "after eliminate " << std::endl;
            printGraph(); 
            #endif 
        }
    }
}

void ClusterClique::printGraph()
{
    std::cout << std::endl << "Graph: " << vertexNum << " vertex, " << edgeNum << " edges" << std::endl; 
    std::cout << "Adjacency: " << std::endl; 
    for(int i = 0; i < adjacencyArray.size(); i++)
    {
        std::cout << std::setw(2) << i << "(" << table[i].first << "," << table[i].second <<  "): "; 
        for(int j = 0; j < adjacencyArray[i].size(); j++)
            std::cout << std::setw(2) << adjacencyArray[i][j] << "(" << table[adjacencyArray[i][j]].first << "," << table[adjacencyArray[i][j]].second << ") ";
        std::cout << std::endl; 
    }
    std::cout << std::endl; 
}

void ClusterClique::printAllMC()
{
    std::cout << std::endl << "Maximal Cliques: " << allMC.size() << std::endl; 

    for(auto iteri = allMC.begin(); iteri != allMC.end(); iteri++)
    {
        std::cout << "clique : "; 
        for(auto iterj = iteri->begin(); iterj != iteri->end(); iterj++)
        {
            std::cout << std::setw(2) << *iterj << "(" << table[*iterj].first<< "," <<  table[*iterj].second << ") "; 
        }
        std::cout << std::endl; 
    }
}

void ClusterClique::printCliques()
{
    std::cout << std::endl << "Final Cliques: " << cliques.size() << std::endl; 

    for(auto iteri = cliques.begin(); iteri != cliques.end(); iteri++)
    {
        std::cout << "clique : "; 
        for(auto iterj = iteri->begin(); iterj != iteri->end(); iterj++)
        {
            std::cout << std::setw(2) << *iterj << "(" << table[*iterj].first<< "," <<  table[*iterj].second << ") "; 
        }
        std::cout << std::endl; 
    }
}

void ClusterClique::convertListToVec(const std::list< std::list<int> > &cliqueList, std::vector< std::vector<int> > &cliqueVec)
{
    cliqueVec.clear(); 
    cliqueVec.resize(cliqueList.size()); 
    int i = 0; 
    for(auto iter1 = cliqueList.begin(); iter1 != cliqueList.end(); iter1++)
    {
        cliqueVec[i].resize(iter1->size()); 
        int j = 0; 
        for(auto iter2 = iter1->begin(); iter2 != iter1->end(); iter2++)
        {
            cliqueVec[i][j] = *iter2; 
            j++; 
        }
        i++; 
    }
}
