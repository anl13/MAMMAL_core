#include "../utils/definitions.h"
#include "framesolver.h" 


// 2021.10.1
void FrameSolver::fetchGtData()
{
	// keypoint: [camid, pigid, jointid]
	m_gt_keypoints_undist.clear();
	m_gt_keypoints_undist.resize(m_camNum);
	for (int i = 0; i < m_camNum; i++)
	{
		m_gt_keypoints_undist[i].resize(4);
		for (int j = 0; j < 4; j++) m_gt_keypoints_undist[i][j].resize(m_topo.joint_num, Eigen::Vector3f::Zero());
	}
	// mask: [camid, pigid, partid, pointid] 
	m_gt_masksUndist.clear();
	m_gt_masksUndist.resize(m_camNum);
	for (int i = 0; i < m_camNum; i++)
	{
		m_gt_masksUndist[i].resize(4);
	}

	for (int i = 0; i < m_camNum; i++)
	{
		int camid = m_camids[i];
		std::stringstream ss;
		ss << BamaPig3D_PATH << "/label_images/cam" << camid << "/" << std::setw(6) << std::setfill('0') << m_frameid << ".json";
		std::string labelpath = ss.str();
		Json::Value root;
		Json::CharReaderBuilder rbuilder;
		std::string errs;
		std::ifstream is(labelpath);
		if (!is.is_open())
		{
			std::cout << "can not open " << labelpath << std::endl;
			continue;
		}
		bool parsingSuccessful = Json::parseFromStream(rbuilder, is, &root, &errs);
		if (!parsingSuccessful)
		{
			std::cout << "parsing " << labelpath << " error!" << std::endl;
			exit(-1);
		}
		Json::Value shapes = root["shapes"];
		for (int k = 0; k < shapes.size(); k++)
		{
			Json::Value dict = shapes[k];
			if (dict["shape_type"] == "point")
			{
				Eigen::Vector3f p;
				p[0] = dict["points"][0][0].asFloat();
				p[1] = dict["points"][0][1].asFloat();
				p[2] = 1;
				int label = std::stoi(dict["label"].asString());
				int group = dict["group_id"].asInt();
				m_gt_keypoints_undist[i][group][label] = p;
			}
			else if (dict["shape_type"] == "polygon")
			{
				int group = dict["group_id"].asInt();
				std::vector<Eigen::Vector2f> points;
				int N = dict["points"].size();
				points.resize(N);
				for (int m = 0; m < N; m++)
				{
					points[m][0] = dict["points"][m][0].asFloat();
					points[m][1] = dict["points"][m][1].asFloat();
				}
				m_gt_masksUndist[i][group].push_back(points);
			}
		}
		is.close();
	}
	// assemble keypoints 
	m_gt_matched.clear();
	m_gt_matched.resize(4);
	for (int camid = 0; camid < m_camNum; camid++)
	{
		for (int group = 0; group < 4; group++)
		{
			m_gt_matched[group].view_ids.push_back(camid);
			DetInstance det;
			det.valid = true;
			det.keypoints = m_gt_keypoints_undist[camid][group];
			det.mask = m_gt_masksUndist[camid][group];
			m_gt_matched[group].dets.push_back(det);
			m_gt_matched[group].candids.push_back(-1);
		}
	}
}

// 2021.10.23 
void FrameSolver::compute_silhouette_loss()
{
	// step1: render depth with scene 
	std::vector<cv::Mat> masksWithScene; 
	masksWithScene.clear(); 
	masksWithScene.resize(m_camNum); 

	std::vector<Eigen::Vector3f> id_colors = {
		{1.0f, 0.0f,0.0f},
		{0.0f, 1.0f, 0.0f},
		{0.0f, 0.0f, 1.0f},
		{1.0f, 1.0f, 0.0f}
	};
	std::vector<Eigen::Vector3i> id_colors_cv = {
		{0,0,255},
		{0,255,0},
		{255,0,0},
		{0,255,255}
	};
	mp_renderEngine->clearAllObjs();
	mp_renderEngine->SetBackgroundColor(Eigen::Vector4f(0, 0, 0, 0)); 
	for (int i = 0; i < m_pignum; i++)
	{
		mp_bodysolverdevice[i]->UpdateNormalFinal();
		mp_bodysolverdevice[i]->map_reduced_vertices(); 
		RenderObjectColor* p_model = new RenderObjectColor();
		//p_model->SetVertices(mp_bodysolverdevice[i]->GetVertices());
		//p_model->SetFaces(mp_bodysolverdevice[i]->GetFacesVert());
		//p_model->SetNormal(mp_bodysolverdevice[i]->GetNormals());
		p_model->SetVertices(mp_bodysolverdevice[i]->m_reduced_vertices);
		p_model->SetFaces(mp_bodysolverdevice[i]->m_reduced_faces); 
		p_model->SetNormal(mp_bodysolverdevice[i]->m_reduced_normals); 
		p_model->SetColor(id_colors[i]);
		mp_renderEngine->colorObjs.push_back(p_model);
	}
	mp_renderEngine->createSceneDetailed(m_project_folder,1); 

	for (int view = 0; view < m_camNum; view++)
	{
		int camid = view;
		Camera cam = m_camsUndist[camid];
		mp_renderEngine->s_camViewer.SetExtrinsic(cam.R, cam.T);
		glDisable(GL_CULL_FACE); 
		mp_renderEngine->Draw("mask");
		masksWithScene[view] = mp_renderEngine->GetImage();
	}

	// step2: compute iou loss for a given est-gt pair 
	std::vector<cv::Mat> gtmasks; 
	gtmasks.resize(m_camNum); 
	for (int view = 0; view < m_camNum; view++)
	{
		gtmasks[view].create(cv::Size(m_imw, m_imh), CV_8UC1);
		for (int id = 0; id < 4; id++)
		{
			my_draw_mask_gray(gtmasks[view],
				m_gt_matched[id].dets[view].mask, id + 1);
		}
	}

	std::vector<cv::Mat> det_masks; 
	det_masks.resize(m_camNum); 
	for (int view = 0; view < m_camNum; view++)
	{
		det_masks[view].create(cv::Size(m_imw, m_imh), CV_8UC1);
	}
	for (int id = 0; id < 4; id++)
	{
		for (int k = 0; k < m_matched[id].view_ids.size(); k++)
		{
			int viewid = m_matched[id].view_ids[k];
			if (m_matched[id].dets[k].mask.size() > 0)
			{
				my_draw_mask_gray(det_masks[viewid], m_matched[id].dets[k].mask, id + 1); 
			}
		}
	}

	//for (int view = 0; view < m_camNum; view++)
	//{
	//	cv::namedWindow("det" + std::to_string(m_camids[view]), cv::WINDOW_NORMAL);
	//	cv::namedWindow("gt" + std::to_string(m_camids[view]), cv::WINDOW_NORMAL);
	//	cv::imshow("det" + std::to_string(m_camids[view]), det_masks[view]*63);
	//	cv::imshow("gt" + std::to_string(m_camids[view]), gtmasks[view]*63);
	//	int key = cv::waitKey();
	//	cv::destroyAllWindows();
	//	if (key == 27) break;
	//}
	

	Eigen::MatrixXf ious(4, m_camNum); 
	ious.setZero(); 
	Eigen::MatrixXf Is(4, m_camNum); 
	Is.setZero(); 
	Eigen::MatrixXf Us(4, m_camNum);
	Us.setZero(); 

	Eigen::MatrixXf Is_det(4, m_camNum); 
	Is_det.setZero(); 
	Eigen::MatrixXf Us_det(4, m_camNum);
	Us_det.setZero(); 

	Eigen::MatrixXf Gt_area(4, m_camNum); 
	Gt_area.setZero(); 
	for (int pid = 0; pid < m_pignum; pid++)
	{
		int gtid = m_pig_names[pid]; 
		for (int camid = 0; camid < m_camNum; camid++)
		{
			float I = 0;
			float U = 0;
			float I_det = 0; 
			float U_det = 0; 
			float gt_area = 0; 
			for (int x = 0; x < 1920; x++)
			{
				for (int y = 0; y < 1080; y++)
				{
					if (mp_sceneData->m_undist_mask.at<uchar>(y, x) == 0) continue;
					if ((masksWithScene[camid].at<cv::Vec3b>(y, x)[0] == id_colors_cv[pid](0)
						&& masksWithScene[camid].at<cv::Vec3b>(y, x)[1] == id_colors_cv[pid](1)
						&& masksWithScene[camid].at<cv::Vec3b>(y, x)[2] == id_colors_cv[pid](2)
						) && gtmasks[camid].at<uchar>(y, x) == (gtid + 1)
						)
					{
						I += 1; 
					}
					if ((masksWithScene[camid].at<cv::Vec3b>(y, x)[0] == id_colors_cv[pid](0)
						&& masksWithScene[camid].at<cv::Vec3b>(y, x)[1] == id_colors_cv[pid](1)
						&& masksWithScene[camid].at<cv::Vec3b>(y, x)[2] == id_colors_cv[pid](2)
						) || gtmasks[camid].at<uchar>(y, x) == (gtid + 1)
						)
					{
						U += 1;
					}

					if ((det_masks[camid].at<uchar>(y, x) == (pid+1)
						) && gtmasks[camid].at<uchar>(y, x) == (gtid + 1)
						)
					{
						I_det += 1;
					}
					if ((det_masks[camid].at<uchar>(y, x) == (pid + 1)
						) || gtmasks[camid].at<uchar>(y, x) == (gtid + 1)
						)
					{
						U_det += 1;
					}
					if (gtmasks[camid].at<uchar>(y, x) == (gtid + 1))
						gt_area += 1; 
				}
			}
			Is(gtid, camid) = float(I); 
			Us(gtid, camid) = float(U); 
			Is_det(gtid, camid) = I_det; 
			Us_det(gtid, camid) = U_det;
			Gt_area(gtid, camid) = gt_area; 
			if (I >= 1)
			{
				ious(gtid, camid) = float(I) / float(U);
			}
			if (U == 0)
				ious(gtid, camid) = -1; 
		}
	}

	std::cout << ious << std::endl;
#if 1  // set 1 to write results. 
	 
	std::stringstream ss; 
	ss << m_result_folder << "/eval/iou_" << m_frameid << ".txt"; 
	std::ofstream os(ss.str()); 
	os << ious;
	os.close(); 

	std::stringstream ss1;
	ss1 << m_result_folder << "/eval/I_" << m_frameid << ".txt";
	std::ofstream os1(ss1.str());
	os1 << Is;
	os1.close();

	std::stringstream ss2;
	ss2 << m_result_folder << "/eval/U_" << m_frameid << ".txt";
	std::ofstream os2(ss2.str());
	os2 << Us;
	os2.close();
#endif 

#if 0 // This part shows the IoU loss of masks detected by PointRend. It achieves the best performance compared with reconstructed ones. 
	std::stringstream ss3;
	ss3 << m_result_folder << "/eval_det/I_" << m_frameid << ".txt";
	std::ofstream os3(ss3.str());
	os3 << Is_det;
	os3.close();

	std::stringstream ss4;
	ss4 << m_result_folder << "/eval_det/U_" << m_frameid << ".txt";
	std::ofstream os4(ss4.str());
	os4 << Us_det;
	os4.close();

	std::stringstream ss5; 
	ss5 << m_result_folder << "/eval_det/Area_" << m_frameid << ".txt";
	std::ofstream os5(ss5.str()); 
	os5 << Gt_area;
	os5.close(); 
#endif 
	// step3: detection mask loss 

	// step4: detection keypoint loss 


}

vector<vector<Eigen::Vector3f>> FrameSolver::load_gt_joint23(std::string folder, int frameid)
{
	vector<vector<Eigen::Vector3f> > skels;
	skels.resize(4);

	for (int pid = 0; pid < 4; pid++)
	{
		skels[pid].resize(23);
		std::stringstream ss;
		ss << folder << "/pig_" << pid << "_frame_" << std::setw(6) << std::setfill('0') << frameid << ".txt";
		std::ifstream is(ss.str());
		if (!is.is_open())
		{
			std::cout << "in load_skel, " << folder << ", " << frameid << ", can not open" << std::endl;
			exit(-1);
		}
		for (int i = 0; i < 23; i++)
		{
			for (int k = 0; k < 3; k++)
			{
				is >> skels[pid][i](k);
			}
		}
		is.close();
	}
	return skels;
}
