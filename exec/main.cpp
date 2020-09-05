#include "main.h"	
#include <string>

#include "../articulation/pigsolverdevice.h" 

int main(int argc, char** argv)
{
	std::string smal_config = "D:/Projects/animal_calib/articulation/artist_config.json";
	//std::shared_ptr<PigSolverDevice> p_smal = std::make_shared<PigSolverDevice>(smal_config);
	//p_smal->debug();
	PigSolverDevice model(smal_config); 
	model.debug(); 
	system("pause");
	exit(-1);
	//write_video();
	//test_nanogui();

	//render_smal_test(); 
	//run_shape();
	run_pose();
	//run_on_sequence();

	//removeSmalTail();

	//ComputeSymmetry();


	/// calibration and annotation
	//test_calib("D:/Projects/animal_calib/");

	//annotate(); 


	//testfunc(); 
	
	//modify_model();
}