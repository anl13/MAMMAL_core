#include <json/json.h> 
#include <sstream> 
#include <vector>
#include <iostream> 
#include <fstream> 
#include <Eigen/Eigen> 

using std::vector; 

int main()
{   
    std::string jsonfile = "/home/al17/animal/pig-data/sequences/20190704_morning/boxes/boxes_000000.json";

    // parse
    Json::Value root; 
    Json::CharReaderBuilder rbuilder; 
    std::string errs; 
    std::ifstream is(jsonfile); 
    if (!is.is_open())
    {
        std::cout << "can not open " << jsonfile << std::endl; 
        exit(-1); 
    }
    bool parsingSuccessful = Json::parseFromStream(rbuilder, is, &root, &errs);
    if(!parsingSuccessful)
    {
        std::cout << "Fail to parse doc \n" << errs << std::endl;
        exit(-1); 
    } 

    vector<int> m_camids; 
    m_camids = {0, 1, 2, 5, 6, 7, 8, 9, 10, 11}; 
    vector<vector<Eigen::Vector4d > > boxes; 
    boxes.resize(m_camids.size()); 
    for(int i = 0; i < m_camids.size(); i++)
    {
        int camid = m_camids[i]; 
        Json::Value c = root[std::to_string(camid)]; 
        int boxnum = c.size(); 
        std::vector<Eigen::Vector4d> bb; 
        for(int bid = 0; bid < boxnum; bid++)
        {
            Json::Value box_jv = c[bid]; 
            Eigen::Vector4d B; 
            for(int k = 0; k < 4; k++)
            {
                double x = box_jv[k].asDouble(); 
                B(k) = x; 
            }
            bb.push_back(B); 
        }
        boxes[i] = bb; 
    }

    return 0; 
}