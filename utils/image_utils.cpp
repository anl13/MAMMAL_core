
#include "image_utils.h"
#include "geometry.h"
#include "math_utils.h"

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

void my_undistort_points(const std::vector<Eigen::Vector3d>& points, 
    std::vector<Eigen::Vector3d>& out, const Camera &camera, const Camera &newcam)
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

void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3d> &points)
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

void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3i &c)
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

void my_draw_points(cv::Mat &img, const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3i &c, int radius)
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

void my_draw_point(cv::Mat& img, const Eigen::Vector3d &point, const Eigen::Vector3i& c, int radius)
{
    int x = int(point(0)); 
    int y = int(point(1)); 
    float conf = float(point(2));
    cv::circle(img, cv::Point(x,y), radius, cv::Scalar(c(0),c(1),c(2)), -1); 
}

void draw_line(cv::Mat &img, Eigen::Vector3d ep, Eigen::Vector3i c)
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

void my_draw_segment(cv::Mat &img, const Vec3& s, const Vec3& e, const Eigen::Vector3i c)
{
    cv::Point2f s_cv; 
    cv::Point2f e_cv; 
    s_cv.x = s(0); 
    s_cv.y = s(1); 
    e_cv.x = e(0); 
    e_cv.y = e(1); 

    cv::line(img, s_cv, e_cv, cv::Scalar(c(0),c(1),c(2)), 10); 
}

void my_draw_segment(cv::Mat &img, const Vec3& s, const Vec3& e, const Eigen::Vector3i c, int linewidth, int pointRadius)
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

void getLegend(cv::Mat & out) 
{
    std::vector<std::string> legend_names = {
        "face", "left front leg", "left back leg", 
        "torso", "right front leg", "right back leg"
    };
    std::vector<Eigen::Vector3i> m_CM;
    getColorMap("anliang_rgb", m_CM); 
    // legend size: 50 * 1024
    cv::Mat img;
    img.create(cv::Size(1024,50), CV_8UC3);
    for(int i = 0; i < 6; i++)
    {
        auto c = m_CM[i]; 
        cv::rectangle(img, cv::Point2f(170*i+15, 15), cv::Point2f(170*i+45, 35), cv::Scalar(c(0), c(1), c(2)),-1);
        cv::putText(img, legend_names[i], cv::Point2f(170*i+60, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255));
    }
    out = img;  
}


std::vector<Eigen::Vector3d> read_points(std::string filename)
{
    std::ifstream infile(filename); 
    if(!infile.is_open())
    {
        std::cout << RED_TEXT("can not open file ") << filename << std::endl; 
        exit(-1); 
    }
    std::vector<Eigen::Vector3d> points; 
    while(!infile.eof())
    {
        double x, y, z; 
        infile >> x >> y; 
        if(infile.eof()) break;
        infile >> z; 
        Eigen::Vector3d p;
        p << x, y, z;
        points.push_back(p); 
    };
    infile.close(); 
    return points;
}

Eigen::Vector4d my_undistort_box(Eigen::Vector4d x, const Camera &cam, const Camera &newcam)
{
/*
  p1-------p4
  |  a box  |
  p3-------p2
*/
    std::vector<Eigen::Vector3d> points; 
    Eigen::Vector3d p1; p1(0) = x(0); p1(1) = x(1); p1(2) = 1; 
    Eigen::Vector3d p2; p2(0) = x(2); p2(1) = x(3); p2(2) = 1; 
    Eigen::Vector3d p3; p3(0) = x(0); p3(1) = x(3); p3(2) = 1; 
    Eigen::Vector3d p4; p4(0) = x(2); p4(1) = x(1); p4(2) = 1; 
    points.push_back(p1); 
    points.push_back(p2); 
    points.push_back(p3); 
    points.push_back(p4); 
    std::vector<Eigen::Vector3d> points_out; 
    my_undistort_points(points, points_out, cam, newcam); 
    Eigen::Vector4d new_box; 
    double x_min = my_max(my_min(points_out[0](0), points_out[2](0)), 0);
    double y_min = my_max(my_min(points_out[0](1), points_out[3](1)), 0);
    double x_max = my_min(my_max(points_out[3](0), points_out[1](0)), 1919); 
    double y_max = my_min(my_max(points_out[2](1), points_out[1](1)), 1079); 
    new_box(0) = x_min;
    new_box(1) = y_min; 
    new_box(2) = x_max; 
    new_box(3) = y_max; 
    return new_box; 
}

Eigen::Vector4d expand_box(Eigen::Vector4d x, double ratio)
{
    Eigen::Vector4d out; 
    double w = x(2) - x(0); 
    double h = x(3) - x(1); 
    double dx = w * ratio;
    double dy = h * ratio; 
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
void my_draw_boxes(cv::Mat& img, const std::vector<Eigen::Vector4d>& boxes)
{
    for(int bid = 0; bid < boxes.size(); bid++)
    {
        Eigen::Vector4d x = boxes[bid]; 
        Eigen::Vector3d p1; p1(0) = x(0); p1(1) = x(1); p1(2) = 1; 
        Eigen::Vector3d p2; p2(0) = x(2); p2(1) = x(3); p2(2) = 1; 
        Eigen::Vector3d p3; p3(0) = x(0); p3(1) = x(3); p3(2) = 1; 
        Eigen::Vector3d p4; p4(0) = x(2); p4(1) = x(1); p4(2) = 1; 
        Eigen::Vector3i color = Eigen::Vector3i::Zero(); 
        color(1) = 255; 
        my_draw_segment(img, p1, p3, color, 3, 6); 
        my_draw_segment(img, p1, p4, color, 3, 6); 
        my_draw_segment(img, p3, p2, color, 3, 6); 
        my_draw_segment(img, p4, p2, color, 3, 6); 
    }
}

void my_draw_box(cv::Mat& img, const Eigen::Vector4d& box, Eigen::Vector3i c)
{
    Eigen::Vector4d x = box; 
    Eigen::Vector3d p1; p1(0) = x(0); p1(1) = x(1); p1(2) = 1; 
    Eigen::Vector3d p2; p2(0) = x(2); p2(1) = x(3); p2(2) = 1; 
    Eigen::Vector3d p3; p3(0) = x(0); p3(1) = x(3); p3(2) = 1; 
    Eigen::Vector3d p4; p4(0) = x(2); p4(1) = x(1); p4(2) = 1; 
    Eigen::Vector3i color = c;
    my_draw_segment(img, p1, p3, color, 3, 6); 
    my_draw_segment(img, p1, p4, color, 3, 6); 
    my_draw_segment(img, p3, p2, color, 3, 6); 
    my_draw_segment(img, p4, p2, color, 3, 6); 
}

void my_draw_mask(cv::Mat& img, 
    vector<vector<Eigen::Vector2d> > contour_list, 
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
            p.x = int(contour_list[i][j](0)); 
            p.y = int(contour_list[i][j](1)); 
            contour_part.push_back(p); 
        }
        contours.push_back(contour_part); 
    }
    cv::Scalar color(c(0), c(1), c(2)); 
    cv::fillPoly(img, contours, color, 1, 0); 

    img = blend_images(raw, img, alpha); 
}

void my_draw_mask_gray(cv::Mat& img,
	vector<vector<Eigen::Vector2d> > contour_list,
	int c)
{
	cv::Mat color_img(cv::Size(1920, 1080), CV_8UC3); 
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
	cv::fillPoly(color_img, contours, cv::Scalar(c, c, c));
	cv::cvtColor(color_img, img, cv::COLOR_BGR2GRAY);

}

cv::Mat blend_images(cv::Mat img1, cv::Mat img2, float alpha)
{
    cv::Mat out = img1 * alpha + img2 * (1-alpha); 
    return out; 
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
	cv::Mat final_image(cv::Size(width, height), CV_8UC3, cv::Scalar(255, 255, 255));

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
	cv::distanceTransform(gray, chamfer, cv::DIST_L2, 3);
	//std::cout << chamfer.type() << std::endl;
	return chamfer; 
}

cv::Mat vis_float_image(cv::Mat chamfer)
{
	int w = chamfer.cols; 
	int h = chamfer.rows; 
	cv::Mat img(cv::Size(w, h), CV_8UC1);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if(chamfer.at<float>(i,j)<256)
				img.at<uchar>(i, j) = uchar(chamfer.at<float>(i, j));
			else img.at<uchar>(i, j) = 255; 
		}
	}
	return img; 
}

/// ROIDescripter 
float ROIdescripter::queryChamfer(const Eigen::Vector3d& point)
{
	Eigen::Vector3d proj = project(cam, point);
	Eigen::Vector2i p_int;
	p_int(0) = int(round(proj(0)));
	p_int(1) = int(round(proj(1)));
	if (p_int(0) < 0 || p_int(0) >= 1920 || p_int(1) < 0 || p_int(1) >= 1080)
		return -1;
	return chamfer.at<float>(p_int(1), p_int(0));
}

int ROIdescripter::queryMask(const Eigen::Vector3d& point)
{
	Eigen::Vector3d proj = project(cam, point);
	Eigen::Vector2i p_int;
	p_int(0) = int(round(proj(0)));
	p_int(1) = int(round(proj(1)));
	if (p_int(0) < 0 || p_int(0) >= 1920 || p_int(1) < 0 || p_int(1) >= 1080)
		return -1; 
	return mask.at<uchar>(p_int(1), p_int(0));
}	

float queryPixel(const cv::Mat& img, const Eigen::Vector3d& point, const Camera& cam)
{
	Eigen::Vector3d proj = project(cam, point);
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


cv::Mat reverseChamfer(const cv::Mat& chamfer)
{
	cv::Mat out;
	out.create(cv::Size(chamfer.cols, chamfer.rows), CV_8UC1);
	for (int j = 0; j < out.rows; j++)
	{
		for (int i = 0; i < out.cols; i++)
		{
			if (chamfer.at<float>(j, i) == 0)out.at<uchar>(j, i) = 1;
			else out.at<uchar>(j, i) = 0;
		}
	}
	out = get_dist_trans(out); 
	return out; 
}