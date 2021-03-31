
#include "global_config.h"

#include <json/json.h>
#include <boost/filesystem.hpp> 

GlobalConfig::GlobalConfig()
{
	projectFolder = ""; 
	skelTopoUsed = "UNIV";
	solverConfigFile = ""; 
	isResume = false; 
	isDebug = false;
	isShowWindow = false; 
}
