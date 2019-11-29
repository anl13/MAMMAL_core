#include "parsing.h" 

// #define DEBUG_PARSING

int checkBothInBox(Eigen::Vector3d X, Eigen::Vector3d Y, Eigen::Vector4d box, Camera cam)
{
    Eigen::Vector3d x_l = cam.R * X + cam.T; 
    Eigen::Vector3d x_l_p = cam.K * x_l; 
    Eigen::Vector2d x_2d = x_l_p.segment<2>(0) / x_l_p(2); 
    Eigen::Vector3d y_l = cam.R * Y + cam.T; 
    Eigen::Vector3d y_l_p = cam.K * y_l; 
    Eigen::Vector2d y_2d = y_l_p.segment<2>(0) / y_l_p(2);
    bool x_in = in_box_test(x_2d, box); 
    bool y_in = in_box_test(y_2d, box); 
    if(x_in && y_in) return 1; 
    else return 0; 
}

void ParsingPool::constructGraph()
{
    assert(pruneThreshold > 0); 
    if(rawData.size() == 0) 
    {
        std::cout << "Empty concensusData ! " << std::endl; 
        return; 
    }

#ifdef DEBUG_PARSING
    std::cout << rawData.size() << std::endl; 
#endif 

    // count vertex number and build query table 
    m_vertexNum = 0; 
    m_invTable.resize(rawData.size()); 
    for(int i = 0; i < rawData.size(); i++)
    {
        m_invTable[i].resize(rawData[i].size()); 
        for(int j = 0; j < rawData[i].size(); j++)
        {
            m_invTable[i][j] = m_vertexNum; 
            m_vertexNum ++; 
            std::pair<int,int> query_record;
            query_record.first = i; 
            query_record.second = j; 
            m_table.push_back(query_record); 
        }
    }
#ifdef DEBUG_PARSING
    std::cout << "m_vertexNum: " << m_vertexNum << std::endl; 
#endif 
    // build graph 
    double maxConnectivity = 0; 
    m_G.resize(m_vertexNum, m_vertexNum); 

    for(int r = 0; r < m_vertexNum; r++)
    {
        for(int c = 0; c < m_vertexNum; c++)
        {
            int kpt_r = m_table[r].first; 
            int kpt_c = m_table[c].first; 
            if(kpt_r == kpt_c) 
            {
                m_G(r,c) = -1; 
                continue; 
            }
            int cand_r = m_table[r].second; 
            int cand_c = m_table[c].second; 
            Eigen::Vector3d X = rawData[kpt_r][cand_r].X;
            Eigen::Vector3d Y = rawData[kpt_c][cand_c].X; 
            int connectivity = 0; 
            for(int camid = 0; camid < camsUndist.size(); camid++)
            {
                Camera cam = camsUndist[camid]; 
                std::vector<Eigen::Vector4d> box_view = boxes[camid];
                for(int bid=0; bid < box_view.size(); bid++)
                {
                    Eigen::Vector4d a_box = box_view[bid]; 
                    connectivity += checkBothInBox(X,Y, a_box, cam); 
                }
            }
            m_G(r,c) = connectivity; 
            if(connectivity > maxConnectivity) maxConnectivity = connectivity; 
        }
    }

    // edge pruning 
    double R = maxConnectivity - pruneThreshold; 
#ifdef DEBUG_PARSING
    std::cout << "R: " << R << std::endl; 
    std::cout << "maxConenctivity: " << maxConnectivity << std::endl; 
#endif 
    assert(R>0);
    for(int i = 0; i < m_vertexNum; i++)
    {
        for(int j = 0; j < m_vertexNum; j++)
        {
            if(m_G(i,j) <= pruneThreshold) m_G(i,j) = -1; 
            else 
            {
                double delta = m_G(i,j) - pruneThreshold; 
                m_G(i,j) = 1 - delta/R; // more connnectivity, lower loss 
            }
        }
    }
#ifdef DEBUG_PARSING
    std::cout << "m_G: " << std::endl; 
    // std::cout << m_G << std::endl; 
#endif 
}

void ParsingPool::solveGraph()
{
    ClusterClique CC; 
    CC.G = m_G; 
    CC.table = m_table; 
    CC.invTable = m_invTable; 
    CC.vertexNum = m_vertexNum; 
    CC.threshold = 1; 
    std::cout << m_G.cols() << std::endl; 
    CC.constructAdjacentGraph(); 
    CC.enumerateBestCliques(); 
    m_cliques = CC.cliques; 
}

void ParsingPool::test()
{
    std::cout << "clique num; " << m_cliques.size() << std::endl; 
}

void ParsingPool::mappingToSkels()
{
    skels.resize(m_cliques.size(), Eigen::MatrixXd::Zero(4, 20)); 
    for(int i = 0; i < m_cliques.size(); i++)
    {
        if(m_cliques[i].size() < cliqueSizeThreshold) continue; 
        for(int k = 0; k < m_cliques[i].size(); k++)
        {
            int vid = m_cliques[i][k]; 
            int kid = m_table[vid].first;
            int cand = m_table[vid].second; 
            Eigen::Vector3d X = rawData[kid][cand].X;
            skels[i].col(kid) = ToHomogeneous(X); 
        }
    }
}