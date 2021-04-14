
#include "image_utils.h"
#include "geometry.h"
#include "math_utils.h"
#include <opencv2/video/background_segm.hpp>

void my_undistort(const cv::Mat &input, cv::Mat &output, const Camera &camera, const Camera &newcam)
{
    cv::Vec<float, 5> distCoef; 
    distCoef[0] = camera.k(0); 
    distCoef[1] = camera.k(1); 
    distCoef[2] = camera.p(0); 
    distCoef[3] = camera.p(1); 
    distCoef[4] = camera.k(2); 

    cv::Mat K(3,3,CV_32FC1, 0.0f);
    cv::eigen2cv(camera.K, K); 
    cv::Mat K2(3,3, CV_32FC1, 0.0f); 
    cv::eigen2cv(newcam.K, K2); 

    cv::Mat R; 
    cv::Mat map1;
    cv::Mat map2; 

    cv::initUndistortRectifyMap(
        K, distCoef, R, K2, input.size(), CV_32FC1, map1, map2
    ); 
    cv::remap(input, output, map1, map2, cv::INTER_LINEAR); 
}

void my_undistort_points(const std::vector<Eigen::Vector3f>& points, 
    std::vector<Eigen::Vector3f>& out, const Camera &camera, const Camera &newcam)
{
    cv::Vec<float, 5> distCoef; 
    distCoef[0] = camera.k(0); 
    distCoef[1] = camera.k(1); 
    distCoef[2] = camera.p(0); 
    distCoef[3] = camera.p(1); 
    distCoef[4] = camera.k(2); 
    cv::Mat K(3,3,CV_32FC1, 0.0f);
    cv::eigen2cv(camera.K, K); 
    cv::Mat K2(3,3, CV_32FC1, 0.0f); 
    cv::eigen2cv(newcam.K, K2); 

    out = points; 

    size_t point_num = points.size(); 
    cv::Mat points_cv(1, point_num, CV_32FC2, 0.0f); 
    for(int i = 0; i < point_num; i++)
    {
        points_cv.ptr<float>(0)[2*i] = float(points[i](0)); 
        points_cv.ptr<float>(0)[2*i+1] = float(points[i](1)); 
    }

    cv::Mat points_cv_out; 
    cv::Mat R; // empty R
    cv::undistortPoints(points_cv, points_cv_out, K, distCoef, R, K2); 

    for(int i = 0; i < point_num; i++)
    {
        out[i](0) = points_cv_out.at<cv::Vec2f>(0,i)[0]; 
        out[i](1) = points_cv_out.at<cv::Vec2f>(0,i)[1];
    }
}

void my_undistort_points(const std::vector<Eigen::Vector2f>& points,
	std::vector<Eigen::Vector2f>& out, const Camera &cam, const Camera &newcam)
{
	std::vector<Eigen::Vector3f> points_in;
	std::vector<Eigen::Vector3f> points_out; 
	points_in.resize(points.size());
#pragma omp parallel for
	for (int i = 0; i < points.size(); i++)
	{
		points_in[i].segment<2>(0) = points[i];
		points_in[i](2);
	}
	my_undistort_points(points_in, points_out, cam, newcam);
	out.resize(points_out.size());
#pragma omp parallel for 
	for (int i = 0; i < points_out.size(); i++)
	{
		out[i] = points_out[i].segment<2>(0);
	}
}

void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3f> &points)
{
    size_t pnum = points.size(); 
    for(int i = 0; i < pnum; i++)
    {
        int x = int(points[i](0)); 
        int y = int(points[i](1)); 
        float conf = float(points[i](2));
        int radius = 10 * conf + 4; 
        cv::circle(img, cv::Point(x,y), 3, cv::Scalar(255,255,255), -1); 
    }
}

void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3f> &points, const Eigen::Vector3i &c)
{
    size_t pnum = points.size(); 
    for(int i = 0; i < pnum; i++)
    {
        int x = int(points[i](0)); 
        int y = int(points[i](1)); 
        float conf = float(points[i](2));
        int radius = 14; 
        cv::circle(img, cv::Point(x,y), radius, cv::Scalar(c(0),c(1),c(2)), -1); 
    }
}

void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3f> &points, const Eigen::Vector3i &c, int radius)
{
    size_t pnum = points.size(); 
    for(int i = 0; i < pnum; i++)
    {
        int x = int(points[i](0)); 
        int y = int(points[i](1)); 
        float conf = float(points[i](2));
        cv::circle(img, cv::Point(x,y), radius, cv::Scalar(c(0),c(1),c(2)), -1); 
    }
}

void my_draw_point(cv::Mat& img, const Eigen::Vector3f &point, const Eigen::Vector3i& c, int radius)
{
    int x = int(point(0)); 
    int y = int(point(1)); 
    float conf = float(point(2));
    cv::circle(img, cv::Point(x,y), radius, cv::Scalar(c(0),c(1),c(2)), -1); 
}

void draw_line(cv::Mat &img, Eigen::Vector3f ep, Eigen::Vector3i c)
{
    int w = img.cols;
    int h = img.rows; 
    cv::Point2f A; 
    cv::Point2f B; 
    cv::Point2f C; 
    cv::Point2f D; 
    A.x = 0; 
    A.y = (-ep(2) - ep(0)*A.x)/ep(1); 
    B.x = w-1;
    B.y = (-ep(2) - ep(0)*B.x)/ep(1); 
    C.y = 0; 
    C.x = (-ep(2) - ep(1)*C.y)/ep(0);
    D.y = h-1; 
    D.x = (-ep(2) - ep(1)*D.y)/ep(0);
    std::vector<cv::Point2f> list; 
    if(in_image(w,h,A.x, A.y)) list.push_back(A);
    if(in_image(w,h,B.x, B.y)) list.push_back(B);
    if(in_image(w,h,C.x, C.y)) list.push_back(C);
    if(in_image(w,h,D.x, D.y)) list.push_back(D);
    if(list.size() == 2)
    {
        cv::line(img, list[0], list[1], cv::Scalar(c(2),c(1),c(0)), 2); 
    }
    else 
    {
        std::cout << "intersection points: " << list.size() << std::endl; 
    }
}

void packImgBlock(const std::vector<cv::Mat> &imgs, cv::Mat &output)
{
    int H = imgs[0].rows; 
    int W = imgs[0].cols; 
    int m_camNum = imgs.size(); 

    int a = sqrt((double)m_camNum);
    if(a * a < m_camNum) a = a+1; 
    int cols = a; 
    int rows = m_camNum / cols;
    if(rows*cols < m_camNum) rows = rows+1; 

    output.create(cv::Size(cols*W, rows*H), imgs[0].type()); 
    try{
        for(int i = 0; i < m_camNum; i++)
        {
            int hid = i / cols;
            int wid = i % cols; 
            imgs[i].copyTo( output(cv::Rect(wid*W, hid*H, W, H)) );
        }
    }
    catch(std::exception e)
    {
        std::cout << "exception in packImgBlock()" << std::endl; 
    }
}

void getColorMap(std::string cm_type, std::vector<Eigen::Vector3i> &colormap)
{
#ifdef _WIN32
	std::string cm_folder = "D:/Projects/animal_calib/data/colormaps/"; 
#else 
	std::string cm_folder = "/home/al17/animal/animal_calib/data/colormaps/";
#endif 
    std::stringstream ss; 
    ss << cm_folder << cm_type << ".txt";

    colormap.clear(); 
    std::ifstream colorstream; 
    colorstream.open(ss.str());
    if(!colorstream.is_open())
    {
        std::cout << "can not open " << ss.str() << std::endl; 
        exit(-1);
    }
    while(!colorstream.eof() )
    {
        int a,b,c;
        colorstream >> a >> b; 
        if(colorstream.eof()) break; 
        colorstream >> c;
        Eigen::Vector3i color; 
        color << a ,b, c;  
        colormap.push_back(color);   
    }
}

//std::vector<float4> getColorMapFloat4(std::string cm_type)
//{
//	std::vector<Eigen::Vector3i> CM; 
//	getColorMap(cm_type, CM);
//	std::vector<float4> CM4; 
//	if (CM.size() > 0)
//	{
//		CM4.resize(CM.size());
//		for (int i = 0; i < CM.size(); i++)
//		{
//			CM4[i] = make_float4(
//				CM[i](0)/255.f, CM[i](1)/255.f, CM[i](2)/255.f, 1.0f);
//		}
//	}
//	return CM4; 
//}

std::vector<Eigen::Vector3i> getColorMapEigen(std::string cm_type)
{
	std::vector<Eigen::Vector3i> CM;
	getColorMap(cm_type, CM); 
	return CM;
}

std::vector<Eigen::Vector3f> getColorMapEigenF(std::string cm_type)
{
	std::vector<Eigen::Vector3f> CM_out; 
	std::vector<Eigen::Vector3i> CM_in;
	getColorMap(cm_type, CM_in); 
	CM_out.resize(CM_in.size());
	for (int i = 0; i < CM_in.size(); i++)
	{
		CM_out[i](0) = CM_in[i](0) / 255.f;
		CM_out[i](1) = CM_in[i](1) / 255.f;
		CM_out[i](2) = CM_in[i](2) / 255.f;
	}
	return CM_out; 
}

//std::vector<nanogui::Vector4f> getColorMapNano(std::string cm_type)
//{
//	std::vector<Eigen::Vector3i> CM;
//	getColorMap(cm_type, CM); 
//	std::vector<nanogui::Vector4f> nano; 
//	nano.resize(CM.size());
//	for (int i = 0; i < CM.size(); i++)
//	{
//		nano[i][0] = CM[i](0) / 255.f; 
//		nano[i][1] = CM[i](1) / 255.f;
//		nano[i][2] = CM[i](2) / 255.f;
//		nano[i][3] = 1.f; 
//	}
//	return nano;
//}

void my_draw_segment(cv::Mat &img, const Eigen::Vector3f& s, const Eigen::Vector3f& e, const Eigen::Vector3i c)
{
    cv::Point2f s_cv; 
    cv::Point2f e_cv; 
    s_cv.x = s(0); 
    s_cv.y = s(1); 
    e_cv.x = e(0); 
    e_cv.y = e(1); 

    cv::line(img, s_cv, e_cv, cv::Scalar(c(0),c(1),c(2)), 10); 
}

void my_draw_segment(cv::Mat &img, const Eigen::Vector3f& s, const Eigen::Vector3f& e, const Eigen::Vector3i c, int linewidth, int pointRadius)
{
    cv::Point2f s_cv; 
    cv::Point2f e_cv; 
    s_cv.x = s(0); 
    s_cv.y = s(1); 
    e_cv.x = e(0); 
    e_cv.y = e(1); 
 
    // cv::line(img, s_cv, e_cv, cv::Scalar(255-c(0), 255-c(1), 255-c(2)), linewidth); 
    if(linewidth > 0) 
    cv::line(img, s_cv, e_cv, cv::Scalar(c(0), c(1), c(2)), linewidth); 

    if(pointRadius > 0)
    {
    cv::circle(img, s_cv, pointRadius, cv::Scalar(c(0),c(1),c(2)), -1); 
    cv::circle(img, e_cv, pointRadius, cv::Scalar(c(0),c(1),c(2)), -1);
    }
}


void cloneImgs(const std::vector<cv::Mat> &input, std::vector<cv::Mat>&output)
{
    int im_num = input.size(); 
    output.resize(im_num); 
    for(int i = 0; i < im_num; i++)
    {
        output[i] = input[i].clone(); 
    }
}

//void getLegend(cv::Mat & out) 
//{
//    std::vector<std::string> legend_names = {
//        "face", "left front leg", "left back leg", 
//        "torso", "right front leg", "right back leg"
//    };
//    std::vector<Eigen::Vector3i> m_CM;
//    getColorMap("anliang_rgb", m_CM); 
//    // legend size: 50 * 1024
//    cv::Mat img;
//    img.create(cv::Size(1024,50), CV_8UC3);
//    for(int i = 0; i < 6; i++)
//    {
//        auto c = m_CM[i]; 
//        cv::rectangle(img, cv::Point2f(170*i+15, 15), cv::Point2f(170*i+45, 35), cv::Scalar(c(0), c(1), c(2)),-1);
//        cv::putText(img, legend_names[i], cv::Point2f(170*i+60, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255));
//    }
//    out = img;  
//}


std::vector<Eigen::Vector3f> read_points(std::string filename)
{
    std::ifstream infile(filename); 
    if(!infile.is_open())
    {
        std::cout << ("can not open file ") << filename << std::endl; 
        exit(-1); 
    }
    std::vector<Eigen::Vector3f> points; 
    while(!infile.eof())
    {
        double x, y, z; 
        infile >> x >> y; 
        if(infile.eof()) break;
        infile >> z; 
        Eigen::Vector3f p;
        p << (float)x, (float)y, (float)z;
        points.push_back(p); 
    };
    infile.close(); 
    return points;
}

Eigen::Vector4f my_undistort_box(Eigen::Vector4f x, const Camera &cam, const Camera &newcam)
{
/*
  p1-------p4
  |  a box  |
  p3-------p2
*/
    std::vector<Eigen::Vector3f> points; 
    Eigen::Vector3f p1; p1(0) = x(0); p1(1) = x(1); p1(2) = 1; 
    Eigen::Vector3f p2; p2(0) = x(2); p2(1) = x(3); p2(2) = 1; 
    Eigen::Vector3f p3; p3(0) = x(0); p3(1) = x(3); p3(2) = 1; 
    Eigen::Vector3f p4; p4(0) = x(2); p4(1) = x(1); p4(2) = 1; 
    points.push_back(p1); 
    points.push_back(p2); 
    points.push_back(p3); 
    points.push_back(p4); 
    std::vector<Eigen::Vector3f> points_out; 
    my_undistort_points(points, points_out, cam, newcam); 
    Eigen::Vector4f new_box; 
    float x_min = my_max(my_min(points_out[0](0), points_out[2](0)), 0);
    float y_min = my_max(my_min(points_out[0](1), points_out[3](1)), 0);
    float x_max = my_min(my_max(points_out[3](0), points_out[1](0)), 1919); 
    float y_max = my_min(my_max(points_out[2](1), points_out[1](1)), 1079); 
    new_box(0) = x_min;
    new_box(1) = y_min; 
    new_box(2) = x_max; 
    new_box(3) = y_max; 
    return new_box; 
}

Eigen::Vector4f expand_box(Eigen::Vector4f x, float ratio)
{
    Eigen::Vector4f out; 
    float w = x(2) - x(0); 
    float h = x(3) - x(1); 
    float dx = w * ratio;
    float dy = h * ratio; 
    out(0) = my_max(x(0) - dx, 0); 
    out(1) = my_max(x(1) - dy, 0); 
    out(2) = my_min(x(2) + dx, 1919); 
    out(3) = my_min(x(3) + dy, 1079); 
    return out; 
}

/*
  p1-------p4
  |  a box  |
  p3-------p2
*/
void my_draw_boxes(cv::Mat& img, const std::vector<Eigen::Vector4f>& boxes)
{
    for(int bid = 0; bid < boxes.size(); bid++)
    {
        Eigen::Vector4f x = boxes[bid]; 
        Eigen::Vector3f p1; p1(0) = x(0); p1(1) = x(1); p1(2) = 1; 
        Eigen::Vector3f p2; p2(0) = x(2); p2(1) = x(3); p2(2) = 1; 
        Eigen::Vector3f p3; p3(0) = x(0); p3(1) = x(3); p3(2) = 1; 
        Eigen::Vector3f p4; p4(0) = x(2); p4(1) = x(1); p4(2) = 1; 
        Eigen::Vector3i color = Eigen::Vector3i::Zero(); 
        color(1) = 255; 
        my_draw_segment(img, p1, p3, color, 3, 6); 
        my_draw_segment(img, p1, p4, color, 3, 6); 
        my_draw_segment(img, p3, p2, color, 3, 6); 
        my_draw_segment(img, p4, p2, color, 3, 6); 
    }
}

void my_draw_box(cv::Mat& img, const Eigen::Vector4f& box, Eigen::Vector3i c)
{
    Eigen::Vector4f x = box; 
    Eigen::Vector3f p1; p1(0) = x(0); p1(1) = x(1); p1(2) = 1; 
    Eigen::Vector3f p2; p2(0) = x(2); p2(1) = x(3); p2(2) = 1; 
    Eigen::Vector3f p3; p3(0) = x(0); p3(1) = x(3); p3(2) = 1; 
    Eigen::Vector3f p4; p4(0) = x(2); p4(1) = x(1); p4(2) = 1; 
	Eigen::Vector3i color = c;
    my_draw_segment(img, p1, p3, color, 3, 6); 
    my_draw_segment(img, p1, p4, color, 3, 6); 
    my_draw_segment(img, p3, p2, color, 3, 6); 
    my_draw_segment(img, p4, p2, color, 3, 6); 
}

void my_draw_mask(cv::Mat& img, 
    vector<vector<Eigen::Vector2f> > contour_list, 
    Eigen::Vector3i c, 
    float alpha)
{
    cv::Mat raw = img.clone(); 
    vector<vector<cv::Point2i> > contours; 
    for(int i = 0; i < contour_list.size(); i++)
    {
        vector<cv::Point2i> contour_part; 
        for(int j = 0; j < contour_list[i].size(); j++)
        {
            cv::Point2i p;
            p.x = round(contour_list[i][j](0)); 
            p.y = round(contour_list[i][j](1)); 
            contour_part.push_back(p); 
        }
        contours.push_back(contour_part); 
    }
    cv::Scalar color(c(0), c(1), c(2)); 
    cv::fillPoly(img, contours, color, 1, 0); 

    img = blend_images(raw, img, alpha); 
}

void my_draw_mask_gray(cv::Mat& grey,
	vector<vector<Eigen::Vector2f> > contour_list,
	int c)
{
	//cv::Mat grey(cv::Size(1920, 1080), CV_8UC1); 
	vector<vector<cv::Point2i> > contours;
	for (int i = 0; i < contour_list.size(); i++)
	{
		vector<cv::Point2i> contour_part;
		for (int j = 0; j < contour_list[i].size(); j++)
		{
			cv::Point2i p;
			p.x = int(contour_list[i][j](0));
			p.y = int(contour_list[i][j](1));
			contour_part.push_back(p);
		}
		contours.push_back(contour_part);
	}
	if(contours.size() > 0 && contours[0].size() > 0)
		cv::fillPoly(grey, contours, c);
	//img = grey;
}

void my_draw_box_fill_gray(cv::Mat& img, const Eigen::Vector4f& box, unsigned int c)
{
	vector < vector < cv::Point2i > > contour_list;
	cv::Point2i P1;
	P1.x = int(box(0)); 
	P1.y = int(box(1)); 
	cv::Point2i P2(int(box(2)), int(box(1)));
	cv::Point2i P3(int(box(2)), int(box(3))); 
	cv::Point2i P4(int(box(0)), int(box(3))); 
	contour_list.resize(1); 
	contour_list[0].push_back(P1); 
	contour_list[0].push_back(P2); 
	contour_list[0].push_back(P3); 
	contour_list[0].push_back(P4);
	cv::fillPoly(img, contour_list, c); 
}

cv::Mat blend_images(cv::Mat img1, cv::Mat img2, float alpha)
{
    cv::Mat out = img1 * alpha + img2 * (1-alpha); 
    return out; 
}

// assume render is black background with non-black foreground. 
// Comment: This function is not true overlay, It is still blend over channels. 
// Comment date: 2020 09 04
cv::Mat overlay_renders(cv::Mat rawimg, cv::Mat render, float a)
{
	cv::Mat mask_fore;
	cv::Mat mask_back;
	cv::Mat gray;
	cv::cvtColor(render, gray, cv::COLOR_BGR2GRAY);
	cv::threshold(gray, mask_fore, 1, 1, cv::THRESH_BINARY);
	cv::Mat mask_fore_3;
	cv::cvtColor(mask_fore, mask_fore_3, cv::COLOR_GRAY2BGR); 
	cv::Mat mask_back_3;
	cv::cvtColor(1 - mask_fore, mask_back_3, cv::COLOR_GRAY2BGR); 
	cv::Mat alpha, beta;
	cv::multiply(mask_fore_3, render, alpha);
	cv::multiply(mask_back_3, rawimg, beta);
	cv::Mat gamma = alpha + beta;
	cv::Mat overlay = a * rawimg + (1 - a) * gamma;
	return overlay;
}

Eigen::Vector3f rgb2bgr(const Eigen::Vector3f& color)
{
	Eigen::Vector3f color2; 
	color2(0) = color(2); 
	color2(1) = color(1); 
	color2(2) = color(0); 
	return color2; 
}

cv::Mat resizeAndPadding(cv::Mat img, const int width, const int height)
{
	cv::Mat final_image(cv::Size(width, height), CV_8UC3, cv::Scalar(228, 228, 240));

	int in_width = img.cols; 
	int in_height = img.rows; 
	int out_width, out_height; 
	int start_x, start_y;
	float r_final = (float)width / height; 
	float r_in = (float)in_width / in_height;
	if (r_in > r_final)
	{
		out_width = width; 
		out_height = (int)(width / r_in); 
		start_x = 0; 
		start_y = (height - out_height) / 2; 
	}
	else
	{
		out_height = height; 
		out_width = (int)(height * r_in); 
		start_y = 0;
		start_x = (width - out_width) / 2; 
	}
	cv::Mat resized; 
	cv::resize(img, resized, cv::Size(out_width, out_height)); 
	cv::Rect2i roi(start_x, start_y, out_width, out_height); 
	resized.copyTo(final_image(roi));
	return final_image; 
}

/*
ATTENTION: 
in mask, non-zero value is foreground. 
*/
cv::Mat get_dist_trans(cv::Mat input)
{
	cv::Mat gray; 
	if (input.channels() == 3)
	{
		cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	}
	else gray = input; 
	//cv::threshold(gray, gray, 40, 255, cv::/*THRESH_BINARY*/ | cv::THRESH_OTSU);
	cv::Mat chamfer; 
	cv::distanceTransform(gray, chamfer, cv::DIST_L2, 5);
	return chamfer; 
}



/// ROIDescripter 
float ROIdescripter::queryChamfer(const Eigen::Vector3f& point)
{
	Eigen::Vector3f proj = project(cam, point);
	Eigen::Vector2i p_int;
	p_int(0) = int(round(proj(0)));
	p_int(1) = int(round(proj(1)));
	if (p_int(0) < 0 || p_int(0) >= 1920 || p_int(1) < 0 || p_int(1) >= 1080)
		return -10000; // out of image
	if (undist_mask.at<uchar>(p_int(1), p_int(0)) == 0) return -10000; 
	return chamfer.at<float>(p_int(1), p_int(0));
}

int ROIdescripter::queryMask(const Eigen::Vector3f& point)
{
	vector<int> list = { 1,2,4,8 };
	Eigen::Vector3f proj = project(cam, point);
	Eigen::Vector2i p_int;
	p_int(0) = int(round(proj(0)));
	p_int(1) = int(round(proj(1)));
	if (p_int(0) < 0 || p_int(0) >= 1920 || p_int(1) < 0 || p_int(1) >= 1080)
		return -1; 
	if (undist_mask.at<uchar>(p_int(1), p_int(0)) == 0)return -1; // outof image
	int code = mask.at<uchar>(p_int(1), p_int(0));
	if (code == idcode) return 1; // true mask 
	
	if (code > 0 && in_list(code, list)) return 2; // occluded by others
	if (code == 0) {
		if (scene_mask.at<uchar>(p_int(1), p_int(0)) > 0) return 3; 
		else 
			return 0; // background  
	}
	return 4; // unknown error  
}	

float ROIdescripter::keypointsMaskOverlay()
{
	int total = 0;
	int valid = 0; 
	for (int i = 0; i < keypoints.size(); i++)
	{
		if (keypoints[i](2) < 0.5) continue;
		total += 1; 
		int y = keypoints[i](1); 
		int x = keypoints[i](0); 
		if (mask.at<uchar>(y, x) == idcode) valid += 1;
	}
	if (total == 0) return 0; 
	return (float)valid / (float)total;
}

float checkKeypointsMaskOverlay(const cv::Mat& mask, const std::vector<Eigen::Vector3f>& keypoints,
	const int& idcode)
{
	int total = 0;
	int valid = 0;
	for (int i = 0; i < keypoints.size(); i++)
	{
		if (keypoints[i](2) < 0.5) continue;
		total += 1;
		int y = keypoints[i](1) ;
		int x = keypoints[i](0) ;
		if (y < 0 || x < 0 || y >= 1080 || x >= 1920)continue; 
		int value = mask.at<uchar>(y, x); 
		if (value == idcode) valid += 1;
	}
	if (total == 0) return 0;
	return (float)valid / (float)total;
}

float queryPixel(const cv::Mat& img, const Eigen::Vector3f& point, const Camera& cam)
{
	Eigen::Vector3f proj = project(cam, point);
	Eigen::Vector2i p_int;
	p_int(0) = int(round(proj(0)));
	p_int(1) = int(round(proj(1)));
	Eigen::Vector4i box(0, 0, img.cols, img.rows);
	bool is_in_box = in_box_test(p_int, box);
	if (!is_in_box)
	{
		return -1;
	}
	else
	{
		return img.at<float>(p_int(1), p_int(0));
	}
}


float queryDepth(const cv::Mat& img, float x, float y)
{
	Eigen::Vector2i p_int; 
	p_int(0) = int(round(x));
	p_int(1) = int(round(y));
	Eigen::Vector4i box(0, 0, img.cols, img.rows);
	bool is_in_box = in_box_test(p_int, box);
	if (!is_in_box)
	{
		return -1;
	}
	return img.at<float>(p_int(1), p_int(0));
	//int x1 = int(x); int x2 = x1 + 1;
	//int y1 = int(y); int y2 = y1 + 1; 
	//float dx = x - x1;
	//float dy = y - y1; 
	//float d_tl = img.at<float>(y1, x1);
	//float d_tr = img.at<float>(y1, x2);
	//float d_bl = img.at<float>(y2, x1);
	//float d_br = img.at<float>(y2, x2);
	//if (d_tl == 0 || d_tr == 0 || d_bl == 0 || d_br == 0)
	//{
	//	float max_d;
	//	max_d = 0 > d_tl ? 0 : d_tl;
	//	max_d = max_d > d_tr ? max_d : d_tr;
	//	max_d = max_d > d_bl ? max_d : d_bl;
	//	max_d = max_d > d_br ? max_d : d_br; 
	//	if (max_d == 0) return -1; 
	//	else return max_d; 
	//}
	//else
	//{
	//	float d;
	//	d = d_tl * (1 - dx) * (1 - dy) + d_tr * (dx) * (1 - dy)
	//		+ d_bl * (1 - dx) *dy + d_br * dx * dy;
	//	return d; 
	//}
}


cv::Mat reverseChamfer(const cv::Mat& chamfer)
{
	cv::Mat out = -chamfer;
	return out; 
}

std::vector<Eigen::Vector2f> computeContourNormal(const std::vector<Eigen::Vector2f>& points)
{
	std::vector<Eigen::Vector2f> normals; 
	normals.resize(points.size());
	int N = points.size();
	for(int i = 0; i < N; i++)
	{
		const Eigen::Vector2f& a = points[(i - 1) % N];
		const Eigen::Vector2f& b = points[i];
		const Eigen::Vector2f& c = points[(i + 1) % N];
		Eigen::Vector2f ac = (c - a).normalized();
		Eigen::Vector2f n;
		n(0) = -ac(1);
		n(1) = ac(0); 
		normals[i] = n;
	}
	return normals;
}

std::vector<std::vector<Eigen::Vector2f>  > 
computeContourNormalsAll(
	const std::vector<std::vector<Eigen::Vector2f> >&points)
{
	std::vector<std::vector<Eigen::Vector2f> > normals;
	normals.resize(points.size());
	for (int i = 0; i < points.size(); i++)
	{
		normals[i] = computeContourNormal(points[i]);
	}
	return normals;
}

cv::Mat my_background_substraction(cv::Mat raw, cv::Mat bg)
{
	cv::Ptr<cv::BackgroundSubtractor> pBackSub;
	pBackSub = cv::createBackgroundSubtractorMOG2();
	cv::Mat fg; 
	pBackSub->apply(raw, fg);
	return fg; 
}

// assume: mask is CV_8UC1 image with non-zero foreground
cv::Mat computeSDF2d(const cv::Mat& render, int thresh)
{
	cv::Mat gray; 
	if (render.channels() > 1)
	{
		cv::cvtColor(render, gray, cv::COLOR_BGR2GRAY);
	}
	else
	{
		gray = render; 
	}

	cv::Mat gray_half; 
	cv::resize(gray, gray_half, cv::Size(960, 540)); 
	cv::Mat inner, outer;
	cv::Mat mask_half, mask_inv_half; 
	cv::threshold(gray_half, mask_half, 1, 255, cv::THRESH_BINARY);
	mask_inv_half = 255 - mask_half; 
	
	//cv::imshow("mask", mask);
	//cv::imshow("mask_inv", mask_inv); 
	//cv::waitKey();
	// innner
	cv::distanceTransform(mask_half, inner, cv::DIST_L2, 5); 
	// outer 
	cv::distanceTransform(mask_inv_half, outer, cv::DIST_L2, 5);

	// final chamfer as sdf
	cv::Mat sdf = inner - outer;
	cv::resize(sdf, sdf, cv::Size(1920, 1080)); 
	sdf = sdf * 2; 
	return sdf;
}

cv::Mat computeSDF2dFromDepthf(const cv::Mat& depth, int thresh)
{
	cv::Mat mask(depth.size(), CV_8UC1);
#pragma omp parallel for
	for (int x = 0; x < mask.cols; x++)
	{
		for (int y = 0; y < mask.rows; y++)
		{
			if (depth.at<float>(y, x) > 0) mask.at<uchar>(y, x) = 255; 
			else mask.at<uchar>(y, x) = 0;
		}
	}
	return computeSDF2d(mask, thresh); 
}

cv::Mat fromDepthToColorMask(cv::Mat depth)
{
	cv::Mat img(depth.size(), CV_8UC3);
#pragma omp parallel for 
	for (int x = 0; x < img.cols; x++)
	{
		for (int y = 0; y < img.rows; y++)
		{
			if (depth.at<float>(y, x) > 0) img.at<cv::Vec3b>(y, x) = cv::Vec3b(0,0,255);
			else img.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
		}
	}
	return img; 
}

cv::Mat visualizeSDF2d(cv::Mat tsdf, int thresh)
{
	std::vector<Eigen::Vector3i> CM;
	getColorMap("bwr", CM);
	int w = tsdf.cols;
	int h = tsdf.rows;
	cv::Mat img(cv::Size(w, h), CV_8UC3);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			float v = tsdf.at<float>(i, j);
			if (thresh < 0)
			{
				int index = 0; 
				if (v <= -128) index = 0;
				else if (v >= 127)index = 255; 
				else { index = (int)(v + 128); }
				img.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(CM[index](2)), uchar(CM[index](1)), uchar(CM[index](0)) );
			}
			else
			{
				int index = (v + thresh) / thresh * 127;
				if (index > 255) index = 255;
				if (index < 0) index = 0;
				img.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(CM[index](2)), uchar(CM[index](1)), uchar(CM[index](0)));
			}
		}
	}
	return img;
}

cv::Mat pseudoColor(cv::Mat depth)
{
	std::vector<Eigen::Vector3i> CM;
	getColorMap("jet", CM);
	int w = depth.cols;
	int h = depth.rows;
	cv::Mat img(cv::Size(w, h), CV_8UC3);
	img.setTo(0);
	double minv, maxv;
	cv::minMaxLoc(depth, &minv, &maxv);
	std::cout << "min: " << minv << "  max: " << maxv << std::endl; 
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			int index; 
			float v = depth.at<float>(i, j);
			if (v < 0 || v > 1) index = 0; 
			index = int((v - minv) / (maxv-minv+0.0001) * 255);
			Eigen::Vector3i c = CM[index];
			img.at<cv::Vec3b>(i, j) = cv::Vec3b(
				uchar(c(2)), uchar(c(1)), uchar(c(0))
			);
		}
	}
	return img; 
}

void computeGradient(cv::Mat input, cv::Mat& outx, cv::Mat& outy)
{
	int scale = 1;
	int delta = 0;
	cv::Sobel(input, outx, CV_32F, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
	cv::Sobel(input, outy, CV_32F, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
}

cv::Mat drawCVDepth(Eigen::MatrixXf vertices, Eigen::MatrixXu faces, Camera cam)
{
	cv::Mat image(cv::Size(1920, 1080), CV_8UC3);
	for (int i = 0; i < faces.cols(); i++)
	{
		int index1 = faces(0, i);
		int index2 = faces(1, i);
		int index3 = faces(2, i);
		Eigen::Vector3f v1d = vertices.col(index1);
		Eigen::Vector3f v2d = vertices.col(index2);
		Eigen::Vector3f v3d = vertices.col(index3);
		Eigen::Vector3f v1 = project(cam, v1d);
		Eigen::Vector3f v2 = project(cam, v2d);
		Eigen::Vector3f v3 = project(cam, v3d);
		std::vector<std::vector<cv::Point2i> > points; 
		points.resize(1);
		points[0].resize(3); 
		int deviation = -0; 
		points[0][0] = cv::Point2i(int(v1(0)+0.5) + deviation, int(v1(1)+0.5));
		points[0][1] = cv::Point2i(int(v2(0)+0.5) + deviation, int(v2(1)+0.5));
		points[0][2] = cv::Point2i(int(v3(0)+0.5) + deviation, int(v3(1)+0.5));
		cv::fillPoly(image, points, cv::Scalar(255,255,255));
	}
	return image; 
}

float silhouette_iou(const cv::Mat& mask1, const cv::Mat& mask2)
{
	return 0; 
}
