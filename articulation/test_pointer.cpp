#include "test_main.h"

#include "pigsolverdevice.h"

void test_pointer()
{

	std::cout << " in test pointer. " << std::endl; 
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	std::shared_ptr<PigSolverDevice> p_smal = std::make_shared<PigSolverDevice>(smal_config); 

	p_smal->debug(); 

	
}