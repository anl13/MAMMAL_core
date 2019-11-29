#include "matching.h" 

bool equal_concensus(const ConcensusData& data1, const ConcensusData& data2)
{
    if(data1.num != data2.num) return false;
    return my_equal(data1.ids, data2.ids); 
}

bool compare_concensus(ConcensusData data1, ConcensusData data2)
{
    if(data1.num < data2.num) return true; 
    if(data1.num > data2.num) return false; 
    // if(data1.metric < data2.metric) return true; 
    // if(data1.metric > data2.metric) return false; 
    for(int i = 0; i < data2.ids.size(); i++)
    {
        if(data1.ids[i] < data2.ids[i]) return true; 
        if(data1.ids[i] > data2.ids[i]) return false; 
    }
    return false; 
}

bool equal_concensus_list( std::vector<ConcensusData> data1,  std::vector<ConcensusData> data2)
{
    if(data1.size() != data2.size()) return false; 
    std::sort(data1.begin(), data1.end(), compare_concensus); 
    std::sort(data2.begin(), data2.end(), compare_concensus); 
    for(int i = 0; i < data1.size(); i++)
    {
        if(!equal_concensus(data1[i], data2[i])) return false; 
    }
    return true; 
}
