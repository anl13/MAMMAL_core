#include "framedata.h"

#include "parsing.h"

void FrameData::parsing()
{
    ParsingPool PP; 
    PP.rawData = m_concensus;
    PP.boxes   = m_boxes_processed; 
    PP.camsUndist = m_camsUndist; 
    PP.pruneThreshold = m_pruneThreshold;
    PP.cliqueSizeThreshold = m_cliqueSizeThreshold;  
    PP.constructGraph(); 
    PP.solveGraph(); 
    PP.test(); 

    PP.mappingToSkels();

    m_skels = PP.skels; 
}