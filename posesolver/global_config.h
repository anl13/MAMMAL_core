#pragma once

#include <string> 

class GlobalConfig
{
public: 
	GlobalConfig(); 
	
	std::string projectFolder; 
	std::string skelTopoUsed; 
	std::string solverConfigFile; 
	bool isResume;
	bool isDebug; 
	bool isShowWindow; 
	
};