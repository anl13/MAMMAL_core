#include "image_utils.h"

#include <opencv2/core/eigen.hpp> 

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
    cv::remap(input, output, map1, map2, CV_INTER_LINEAR); 
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


bool in_image(float w, float h, float x, float y)
{
    return (x>=0 && x<w && y>=0 && y<h); 
}

void draw_line(cv::Mat &img, Eigen::Vector3d ep)
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
        cv::line(img, list[0], list[1], cv::Scalar(0,255,255), 10); 
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

    output.create(cv::Size(cols*W, rows*H), CV_8UC3); 
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
    std::stringstream ss; 
    ss << "/home/al17/animal/animal_calib/data/colormaps/" << cm_type << ".txt";

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
