#include "main.h"

#include "../bundle/annotator.h"
#include "../associate/framedata.h"

void annotate()
{
	std::string folder = "D:/Projects/animal_calib/data/pig_model_noeye/";
	std::string pig_config = "D:/Projects/animal_calib/smal/pigmodel_config.json";
	std::string conf_projectFolder = "D:/Projects/animal_calib/";
	
	SkelTopology topo = getSkelTopoByType("UNIV");
	FrameData frame;
	frame.configByJson(conf_projectFolder + "/associate/config.json");
	int startid = frame.get_start_id();
	int framenum = frame.get_frame_num();
	int frameid = 0;

	
	frame.set_frame_id(frameid);
	frame.fetchData();
	frame.matching_by_tracking();

	std::string result_folder = "E:/my_labels/";
	Annotator A;
	A.result_folder = result_folder;
	A.frameid = frameid;
	A.m_cams = frame.get_cameras();
	A.m_camNum = A.m_cams.size();
	A.setInitData(frame.get_matched());
	A.m_imgs = frame.get_imgs_undist();
	A.m_unmatched = frame.get_unmatched();
	
	A.show_panel();
}

