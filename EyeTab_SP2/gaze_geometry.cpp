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

vector<Vector3d> ellipse_to_limbus(cv::RotatedRect ellipse, bool limbus_switch=true){

	vector<Vector3d> limbus_to_return;

	double maj_axis_px = ellipse.size.width, min_axis_px = ellipse.size.height;

	// Using iris_r_px / focal_len_px = iris_r_mm / distance_to_iris_mm
	double iris_z_mm = (LIMBUS_R_MM * 2 * FOCAL_LEN_Z_PX) / maj_axis_px;
    
    // Using (x_screen_px - prin_point) / focal_len_px = x_world / z_world
	double iris_x_mm = -iris_z_mm * (ellipse.center.x - PRIN_POINT.x) / FOCAL_LEN_X_PX;
    double iris_y_mm = iris_z_mm * (ellipse.center.y - PRIN_POINT.y) / FOCAL_LEN_Y_PX;

	Vector3d limbus_center(iris_x_mm, iris_y_mm, iris_z_mm);

	double psi = CV_PI / 180.0 * (ellipse.angle+90);    // z-axis rotation (radians)
    double tht = acos(min_axis_px / maj_axis_px);       // y-axis rotation (radians)

	if (limbus_switch) tht = -tht;                      // ambiguous acos, so sometimes switch limbus

    // Get limbus normal for chosen theta
    Vector3d limb_normal(sin(tht) * cos(psi), -sin(tht) * sin(psi), -cos(tht));

	// Now correct for weak perspective by modifying angle by offset between camera axis and limbus
    double x_correction = -atan2(iris_y_mm, iris_z_mm);
    double y_correction = -atan2(iris_x_mm, iris_z_mm);
	AngleAxisd rot1(y_correction, Vector3d(0,-1,0));
	AngleAxisd rot2(x_correction, Vector3d(1,0,0));
	limb_normal = rot1 * limb_normal;
	limb_normal = rot2 * limb_normal;

	limbus_to_return.push_back(limbus_center);
	limbus_to_return.push_back(limb_normal);
	return limbus_to_return;
}

// returns intersection with z-plane of optical axis vector (mm)
Point2d get_gaze_point_mm(Vector3d limb_center, Vector3d limb_normal){
    
    // ray/plane intersection
    double t = -limb_center.z() / limb_normal.z();
    return Point2d(limb_center.x() + limb_normal.x() * t, limb_center.y() + limb_normal.y() * t);
}


Point2d get_gaze_pt_mm(RotatedRect& ellipse){

	// get two possible limbus centres and normals because of ambiguous trig
	vector<Vector3d> limbus_a = ellipse_to_limbus(ellipse, true);
	vector<Vector3d> limbus_b = ellipse_to_limbus(ellipse, false);

	// calculate gaze points for each possible limbus
	Point2d gp_mm_a = get_gaze_point_mm(limbus_a[0], limbus_a[1]);
	Point2d gp_mm_b = get_gaze_point_mm(limbus_b[0], limbus_b[1]);

	// calculate distance from centre of screen for each possible gaze point
	int dist_a = std::abs(gp_mm_a.x) + std::abs(gp_mm_a.y);
	int dist_b = std::abs(gp_mm_b.x) + std::abs(gp_mm_b.y);

	// return gaze point closest to screen centre
	return (dist_a < dist_b) ? gp_mm_a : gp_mm_b;
}


const Size SCREEN_SIZE_MM(236, 134);
const Size SCREEN_SIZE_PX(1920, 1080);		// screen size in pixels
const Point2i CAMERA_OFFSET_MM(120, 140);	// vector from top left of screen to camera


Point2i convert_gaze_pt_mm_to_px(Point2d gaze_pt_mm){

	int gp_px_x = (gaze_pt_mm.x + CAMERA_OFFSET_MM.x) / SCREEN_SIZE_MM.width * SCREEN_SIZE_PX.width;
    int gp_px_y = (gaze_pt_mm.y + CAMERA_OFFSET_MM.y) / SCREEN_SIZE_MM.height * SCREEN_SIZE_PX.height;
    
    return Point2i(gp_px_x, gp_px_y);
}


float scale = 720 / float(SCREEN_SIZE_PX.height);

// draws the gaze-points on-screen as circles and crosses
void show_gaze(Mat& img, vector<Point2i> gaze_pt_raw_s, vector<Scalar> colors_raw, Point2i gaze_pt_smoothed, Scalar color_smoothed){

	Mat screen(SCREEN_SIZE_PX.height, SCREEN_SIZE_PX.width, CV_8UC3);
	screen.setTo(YELLOW);

	// draw ra
	for (int i=0; i<gaze_pt_raw_s.size(); i++)
		circle(img, gaze_pt_raw_s[i] * scale, 10, colors_raw[i], -1);

	circle(img, gaze_pt_smoothed * scale, 20, color_smoothed, -1);
}