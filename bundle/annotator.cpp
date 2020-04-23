#include "annotator.h"
#include <Eigen/Eigen>

std::vector<std::pair<std::string, Eigen::Vector2i> > labels = {
	{ "pig 0",{ 0,0 } },
{ "pig 1",{ 0,1 } },
{ "pig 2",{ 0,2 } },
{ "pig 3",{ 0,3 } },

{ "0 nose",{ 2,0 } },
{ "1 leye",{ 2,1 } },
{ "2 reye",{ 2,2 } },
{ "3 lear",{ 2,3 } },
{ "4 rear",{ 2,4 } },
{ "5 lshou",{ 2,5 } },
{ "6 rshou",{ 2,6 } },
{ "7 lelb",{ 2,7 } },
{ "8 relb",{ 2,8 } },
{ "9 lpaw",{ 2,9 } },
{ "10 rpaw",{ 3,0 } },
{ "11 lhip",{ 3,1 } },
{ "12 rhip",{ 3,2 } },
{ "13 lknee",{ 3,3 } },
{ "14 rknee",{ 3,4 } },
{ "15 lfoot",{ 3,5 } },
{ "16 rfoot",{ 3,6 } },
{ "18 tail",{ 3,7 } },
{ "20 center",{ 3,8 } },

{ "Is visible",{ 5,0 } },

{ "JointLabel",{ 7,0 } },
{ "ID Label",{ 7,1 } },
{ "Motion",{ 7,2 } }
};

// util function 
void drawLabel(cv::Mat& img, int x, int y, int w, int h, std::string name, bool clicked)
{
	cv::Scalar color(255, 255, 255);
	if (clicked)
	{
		color = cv::Scalar(0, 255, 0);
	}
	cv::rectangle(img, cv::Rect(x+10, y+10, w-20, h-20), color,-1);
	cv::putText(img, name, cv::Point(x+10, y+30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1.5);
}

void CallBackFuncPanel(int event, int x, int y, int flags, void* userdata)
{
	/*
	data[0]: pig id
	data[1]: joint id
	data[2]: is_visible
	data[3]: mode
	*/
	vector<int>* data = (vector<int>*) userdata;
	int length = data->size();
	int cols = 10;
	int rows = 10;
	int cellw = 150;
	int cellh = 50;
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		int c = x / cellw;
		int r = y / cellh;
		for (int i = 0; i < labels.size(); i++)
		{
			if (labels[i].second(0) == r && labels[i].second(1) == c)
			{
				switch (i) {
				case 0:case 1: case 2:case 3: 
					(*data)[0] = i; 
					break;
				case 23:(*data)[2] = 1 - (*data)[2]; break;
				case 24:case 25: case 26:
					(*data)[3] = i - 24; break;

				case 22:(*data)[1] = 20; break;
				case 21:(*data)[1] = 18; break; 
				default: 
					(*data)[1] = i - 4; break;
				}
				break; 
			}
		}
	}
}


void CallBackImage(int event, int x, int y, int flags, void* userdata)
{
	SingleClickLabeledData* data = (SingleClickLabeledData*)userdata;
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		if (data->single_image)
		{
			data->cancel = false;
			data->x = x;
			data->y = y;
			data->ready = true;
		}
		else
		{
			int c = x / 1920;
			int r = y / 1080;
			int dx = x - c * 1920;
			int dy = y - r * 1080;
			int viewid = r * 4 + c;
			if (viewid < 10)
			{
				data->camid = viewid;
				data->cancel = false;
				data->x = dx;
				data->y = dy;
				data->ready = true;
			}
		}
	}
	else if (event == cv::EVENT_RBUTTONDOWN)
	{
		if (data->single_image)
		{
			data->cancel = true;
			data->ready = true;
		}
		else
		{
			int c = x / 1920;
			int r = y / 1080;
			int viewid = r * 4 + c;
			if (viewid < 10)
			{
				data->camid = viewid;
				data->cancel = true;
				data->ready = true;
			}
		}
	}
	else if (event == cv::EVENT_MBUTTONDOWN)
	{
		if (data->single_image)
		{
			data->single_image = false;
			data->ready = false;
		}
		else {
			int c = x / 1920;
			int r = y / 1080;
			int viewid = r * 4 + c;
			if (viewid < 10)
			{
				data->camid = viewid;
				data->cancel = false;
				data->ready = false;
				data->single_image = true;
			}
		}
	}
}

void Annotator::construct_panel_attr()
{
	m_camNum = -1;
	getColorMap("anliang_rgb", m_CM);
	m_topo = getSkelTopoByType("UNIV");
	int cols = 10;
	int rows = 10;
	int cellw = 150;
	int cellh = 50;
	m_panel_attr.create(cv::Size(cellw*cols, cellh*rows), CV_8UC3);

	for (int i = 0; i < labels.size(); i++)
	{
		int row = labels[i].second(0);
		int col = labels[i].second(1);
		int x = col * cellw;
		int y = row * cellh;

		drawLabel(m_panel_attr, x, y, cellw, cellh, labels[i].first, false);
	}
}

void Annotator::update_panel(const std::vector<int>& status)
{
	int cellw = 150;
	int cellh = 50;
	// draw pig id
	for(int i = 0; i < 4;i++)
	{
		int x = i * cellw;
		int y = 0 * cellh;
		drawLabel(m_panel_attr, x,y, cellw, cellh, labels[i].first, 
			i==status[0]);
	}
	// draw joint id
	int jointid = status[1];
	int labelid = 0; 
	if (jointid < 18) labelid = jointid + 4;
	else if (jointid == 18) labelid = 21;
	else if (jointid == 20) labelid = 22;
	else {}

	for (int i = 4; i < 23; i++)
	{
		int r = labels[i].second(0);
		int c = labels[i].second(1);
		int x = c * cellw;
		int y = r * cellh;
		drawLabel(m_panel_attr, x,y, cellw, cellh, labels[i].first,
			i == labelid);
	}
	// draw visibility
	drawLabel(m_panel_attr, 0, 5*cellh, cellw, cellh, labels[23].first, status[2]);
	// draw mode
	for (int i = 24; i < 27; i++)
	{
		int r = labels[i].second(0);
		int c = labels[i].second(1);
		int x = c * cellw;
		int y = r * cellh;
		drawLabel(m_panel_attr, x, y, cellw, cellh, labels[i].first, status[3] == i - 24);
	}
}


void Annotator::drawSkel(cv::Mat& img, const vector<Eigen::Vector3d>& _skel2d)
{
	for (int i = 0; i < _skel2d.size(); i++)
	{
		int colorid = m_topo.kpt_color_ids[i];
		Eigen::Vector3i color = m_CM[colorid];
		cv::Scalar cv_color(color(0), color(1), color(2));

		cv::Point2d p(_skel2d[i](0), _skel2d[i](1));
		double conf = _skel2d[i](2);
		if (conf < m_topo.kpt_conf_thresh[i]) continue;
		cv::circle(img, p, 9, cv_color, -1);
	}
	for (int k = 0; k < m_topo.bone_num; k++)
	{
		int jid = m_topo.bones[k](0);
		int colorid = m_topo.kpt_color_ids[jid];
		Eigen::Vector3i color = m_CM[colorid];
		cv::Scalar cv_color(color(0), color(1), color(2));

		Eigen::Vector2i b = m_topo.bones[k];
		Eigen::Vector3d p1 = _skel2d[b(0)];
		Eigen::Vector3d p2 = _skel2d[b(1)];
		if (p1(2) < m_topo.kpt_conf_thresh[b(0)] || p2(2) < m_topo.kpt_conf_thresh[b(1)]) continue;
		cv::Point2d p1_cv(p1(0), p1(1));
		cv::Point2d p2_cv(p2(0), p2(1));
		cv::line(img, p1_cv, p2_cv, cv_color, 4);
	}
}


void Annotator::update_image_labeled(
	const SingleClickLabeledData& input,
	const std::vector<int>& status)
{
	if (input.single_image)
	{
		int camid = input.camid;
		int pid = status[0];
		m_image_labeled = m_imgs[camid].clone();
		if (m_data[pid][camid].valid) {
			drawSkel(m_image_labeled, m_data[pid][camid].keypoints);
			my_draw_box(m_image_labeled, m_data[pid][camid].box, m_CM[pid]);
			//my_draw_mask(m_image_labeled, m_data[pid][camid].mask, m_CM[pid], 0.5);
		}
	}
	else {
		std::vector<cv::Mat> m_imgsDetect;
		cloneImgs(m_imgs, m_imgsDetect);

		int pid = status[0];

		for (int id = 0; id < m_data.size(); id++)
		{
			if (pid >= 0 && id != pid)continue;
			for (int camid = 0; camid < m_data[id].size(); camid++)
			{
				if (!m_data[id][camid].valid)continue;
				drawSkel(m_imgsDetect[camid], m_data[id][camid].keypoints);
				my_draw_box(m_imgsDetect[camid], m_data[id][camid].box, m_CM[id]);
				//my_draw_mask(m_imgsDetect[camid], m_data[id][camid].mask, m_CM[id], 0.5);
			}
		}
		packImgBlock(m_imgsDetect, m_image_labeled);
	}
}

void Annotator::setInitData(
	const vector<MatchedInstance>& matched
)
{
	if (m_camNum < 0)
	{
		std::cout << "you should init camera first" << std::endl;
		system("pause");
		exit(-1); 
	}
	m_data.resize(4);
	for (int pid = 0; pid < 4; pid++)m_data[pid].resize(m_camNum);
	for (int pid = 0; pid < 4; pid++)
	{
		for (int i = 0; i < matched[pid].view_ids.size(); i++)
		{
			int camid = matched[pid].view_ids[i];
			//int candid = matched[pid].cand_ids[i];
			m_data[pid][camid] = matched[pid].dets[i];
		}
	}
}

void Annotator::getMatchedData(vector<MatchedInstance>& matched)
{
	matched.clear();
	matched.resize(4);
	for (int pid = 0; pid < 4; pid++)
	{
		for (int camid = 0; camid < m_data[pid].size(); camid++)
		{
			if (m_data[pid][camid].valid)
			{
				matched[pid].view_ids.push_back(camid);
				matched[pid].dets.push_back(m_data[pid][camid]);
			}
		}
	}
}

void Annotator::update_data(const SingleClickLabeledData& input, 
	const std::vector<int>& status
)
{
	if (!input.ready)return;
	int pid = status[0];
	bool visible = (status[2] == 1);
	int jid = status[1];
	int camid = input.camid;
	if (m_data[pid][camid].valid)
	{
		Eigen::Vector3d joint;
		if (!input.cancel)
		{
			joint(0) = input.x;
			joint(1) = input.y;
			if (visible) joint(2) = 1;
			else joint(2) = 2; // invisible but labeled
		}
		else {
			joint = Eigen::Vector3d::Zero();
		}
		m_data[pid][camid].keypoints[jid] = joint;
	}
}

void Annotator::save_label_result(std::string filename)
{
	std::ofstream os;
	os.open(filename);
	if (!os.is_open())
	{
		std::cout << "file " << filename << " cannot open" << std::endl;
		return;
	}

	Json::Value root;
	Json::Value pigs(Json::arrayValue);
	for (int index = 0; index < m_data.size(); index++)
	{
		Json::Value singlepig;
		for (int camid = 0; camid < m_data[index].size(); camid++)
		{
			if (!m_data[index][camid].valid)continue;
			Json::Value dets;
			Json::Value pose(Json::arrayValue);
			for (int i = 0; i < m_topo.joint_num; i++)
			{
				// if a joint is empty, it is (0,0,0)^T
				pose.append(Json::Value(m_data[index][camid].keypoints[i](0)));
				pose.append(Json::Value(m_data[index][camid].keypoints[i](1)));
				pose.append(Json::Value(m_data[index][camid].keypoints[i](2)));
			}
			dets["keypoints"] = pose;
			Json::Value box(Json::arrayValue);
			for (int i = 0; i < 4; i++)
				box.append(Json::Value(m_data[index][camid].box(i)));
			dets["box"] = box;
			Json::Value mask(Json::arrayValue);
			for (int maskid = 0; maskid < m_data[index][camid].mask.size(); maskid++)
			{
				Json::Value apart(Json::arrayValue);
				for (int k = 0; k < m_data[index][camid].mask[maskid].size(); k++)
				{
					apart.append(Json::Value(m_data[index][camid].mask[maskid][k](0)));
					apart.append(Json::Value(m_data[index][camid].mask[maskid][k](1)));
				}
				mask.append(apart);
			}
			dets["mask"] = mask;

			singlepig[std::to_string(camid)] = dets;
		}
		pigs.append(singlepig);
	}
	root["pigs"] = pigs;

	// Json::StyledWriter stylewriter; 
	// os << stylewriter.write(root); 
	Json::StreamWriterBuilder builder;
	builder["commentStyle"] = "None";
	builder["indentation"] = "    ";
	std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
	writer->write(root, &os);
	os.close();
}

void Annotator::read_label_result(std::string filename)
{
	Json::Value root;
	Json::CharReaderBuilder rbuilder;
	std::string errs;
	std::ifstream instream(filename);
	if (!instream.is_open())
	{
		std::cout << "can not open " << filename << std::endl;
		exit(-1);
	}
	bool parsingSuccessful = Json::parseFromStream(rbuilder, instream, &root, &errs);
	if (!parsingSuccessful)
	{
		std::cout << "Fail to parse \n" << errs << std::endl;
		exit(-1);
	}

	m_data.clear();
	m_data.resize(4);
	int pid = 0; 
	for (auto const &pig : root["pigs"])
	{
		m_data[pid].resize(m_camNum);
		for (int camid = 0; camid < m_camNum; camid++)
		{
			m_data[pid][camid].keypoints.resize(m_topo.joint_num);
			std::string camname = std::to_string(camid);
			if (!pig[camname].empty())
			{
				auto const& det = pig[camname];
				// keypoints
				for (int i = 0; i < m_topo.joint_num; i++)
				{
					for (int k = 0; k < 3; k++)
						m_data[pid][camid].keypoints[i](k) = det["keypoints"][i * 3 + k].asDouble();
				}
				// box 
				for (int i = 0; i < 4; i++)
				{
					m_data[pid][camid].box(i) = det["box"][i].asDouble();
				}
				// mask 
				m_data[pid][camid].mask.clear();
				for (auto const& apart : det["mask"])
				{
					std::vector<Eigen::Vector2d> amask;
					int k = 0;
					for (auto const& num : apart)
					{
						if (k % 2 == 0)
						{
							Eigen::Vector2d x; 
							x(0) = num.asDouble();
							amask.push_back(x);
							k++;
						}
						else
						{
							amask[int(k / 2)](1) = num.asDouble();
							k++;
						}
					}
					m_data[pid][camid].mask.push_back(amask);
				}
				m_data[pid][camid].valid = true;
			}
		}
		pid++;
	}
	instream.close();
}

void Annotator::read_label_result()
{
	std::stringstream ss;
	ss << result_folder << "label_frame_" << std::setw(6) << std::setfill('0')
		<< frameid << ".json";
	read_label_result(ss.str());
}

void Annotator::save_label_result()
{
	std::stringstream ss;
	ss << result_folder << "label_frame_" << std::setw(6) << std::setfill('0')
		<< frameid << ".json";
	save_label_result(ss.str());
}

void Annotator::show_panel()
{
	std::vector<int> status = { 0,0,0,0 };
	SingleClickLabeledData labeldata;

	cv::namedWindow("panel", cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback("panel", CallBackFuncPanel, &status);

	cv::namedWindow("image", cv::WINDOW_KEEPRATIO);
	cv::setMouseCallback("image", CallBackImage, &labeldata);
	while (true)
	{
		if (status[3] == 0)
		{
			update_data(labeldata, status);
			labeldata.ready = false;
		}
		update_panel(status);
		update_image_labeled(labeldata, status);
		cv::imshow("panel", m_panel_attr);
		cv::imshow("image", m_image_labeled);
		char c = cv::waitKey(50);
		if (c == 27)
		{
			break;
		}
		else if (c == 's' || c == 'S')
		{
			save_label_result();
		}
		else if (c == 'r' || c == 'R')
		{
			read_label_result();
		}
	}
}