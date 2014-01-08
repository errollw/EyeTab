#include "stdafx.h"

#include <iostream>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Math.h>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "utils.h"

using namespace Eigen;
using namespace std;
using namespace cv;

const float LIMBUS_R_MM = 6;

// ThinkpadHelix Params
const double FOCAL_LEN_X_PX = 957.648052597;
const double FOCAL_LEN_Y_PX = 960.154605354;
const double FOCAL_LEN_Z_PX = (FOCAL_LEN_X_PX + FOCAL_LEN_Y_PX) / 2;
const cv::Point2d PRIN_POINT(634.799023712, 367.91715841);

vector<Vector3d> ellipse_to_limbus(cv::RotatedRect ellipse){

	vector<Vector3d> limbus_to_return;

	double maj_axis_px = ellipse.size.width, min_axis_px = ellipse.size.height;

	// Using iris_r_px / focal_len_px = iris_r_mm / distance_to_iris_mm
	double iris_z_mm = (LIMBUS_R_MM * 2 * FOCAL_LEN_Z_PX) / maj_axis_px;
    
    // Using (x_screen_px - prin_point) / focal_len_px = x_world / z_world
	double iris_x_mm = -iris_z_mm * (ellipse.center.x - PRIN_POINT.x) / FOCAL_LEN_X_PX;
    double iris_y_mm = iris_z_mm * (ellipse.center.y - PRIN_POINT.y) / FOCAL_LEN_Y_PX;

	Vector3d limbus_center(iris_x_mm, iris_y_mm, iris_z_mm);

	double psi = CV_PI / 180.0 * (ellipse.angle+90);    // z-axis rotation (radians)
    double tht_1 = acos(min_axis_px / maj_axis_px);     // y-axis rotation (radians)
	double tht_2 = -tht_1;                              // as acos has 2 ambiguous solutions
    
	cout << "ANGLE " << ellipse.angle << endl;

	//Vector3d limb_normal = Vector3d(limbus_center) * -1;
	//limb_normal.normalize();

	//AngleAxisd rot1(psi, Vector3d(0,0,1));
	//AngleAxisd rot2(-tht_1, Vector3d(0,1,0));
	//limb_normal = rot1 * limb_normal;
	//limb_normal = rot2 * limb_normal;

    // Ignore other possible normal - found in practise to be wrong in general
    Vector3d limb_normal(sin(tht_1) * cos(psi), -sin(tht_1) * sin(psi), -cos(tht_1));

	// Now correct for weak perspective by modifying angle by offset between camera axis and limbus
    double x_correction = -atan2(iris_y_mm, iris_z_mm);
    double y_correction = -atan2(iris_x_mm, iris_z_mm);
	AngleAxisd rot1(y_correction, Vector3d(0,-1,0));
	AngleAxisd rot2(x_correction, Vector3d(1,0,0));
	limb_normal = rot1 * limb_normal;
	limb_normal = rot2 * limb_normal;

	cout << "LIMBUS CENTER " << limbus_center.x() << " " << limbus_center.y() << " " << limbus_center.z() << endl;
	cout << "LIMBUS NORMAL " << limb_normal.x() << " " << limb_normal.y() << " " << limb_normal.z() << endl;

	limbus_to_return.push_back(limbus_center);
	limbus_to_return.push_back(limb_normal);
	return limbus_to_return;
}

// Returns intersection with z-plane of optical axis vector (mm)
Point2d get_gaze_point_mm(Vector3d limb_center, Vector3d limb_normal){
    
    // ray/plane intersection
    double t = -limb_center.z() / limb_normal.z();
    return Point2d(limb_center.x() + limb_normal.x() * t, limb_center.y() + limb_normal.y() * t);
}

Point2d get_gaze_pt_mm(RotatedRect& ellipse){

	vector<Vector3d> limbus = ellipse_to_limbus(ellipse);
	Point2d p = get_gaze_point_mm(limbus[0], limbus[1]);

	return p;
}

const Size SCREEN_SIZE_MM(256,144);
const Size SCREEN_SIZE_PX(1920, 1080);	// Screen size in pixels
const Point2i CAMERA_OFFSET_MM(128, 155);	// Vector from camera to top left of screen

Point2i convert_gaze_pt_mm_to_px(Point2d gaze_pt_mm){

	int gp_px_x = (gaze_pt_mm.x + CAMERA_OFFSET_MM.x) / SCREEN_SIZE_MM.width * SCREEN_SIZE_PX.width;
    int gp_px_y = (gaze_pt_mm.y + CAMERA_OFFSET_MM.y) / SCREEN_SIZE_MM.height * SCREEN_SIZE_PX.height;
    
    return Point2i(gp_px_x, gp_px_y);
}

// Draws the gaze-point on-screen
float scale = 720 / float(SCREEN_SIZE_PX.height);

void show_gaze(Mat& img, vector<Point2i> gaze_pt_px_s, vector<Scalar> colors){
	//resizeWindow("Gaze Output", SCREEN_SIZE_PX.width, SCREEN_SIZE_PX.height);

	Mat screen(SCREEN_SIZE_PX.height, SCREEN_SIZE_PX.width, CV_8UC3);
	screen.setTo(YELLOW);

	for (int i=0; i<gaze_pt_px_s.size(); i++)
		circle(img, gaze_pt_px_s[i] * scale, 20, colors[i], -1);
}