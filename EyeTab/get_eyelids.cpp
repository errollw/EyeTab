#include "stdafx.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

#include "utils.h"

using namespace std;
using namespace cv;

// Distance from eye-centers to guessed canthi positions
const double CANTH_HORIZ_RATIO = 0.2;
const double CANTH_VERT_RATIO = 0.08;

// Returns two eye-corners (canthi) for a given eye, and between-eye vector
vector<Point> guess_eye_corners(Point2i& eye_center, Point2i& eye_cs_vec){

	float eye_dist = norm(eye_cs_vec);
	Point2f v_norm = normVec(eye_cs_vec);			// normalized vec along eye-pair line
	Point2f v_perp = Point2f(v_norm.y, v_norm.x);	// perp. vec to eye-pair line

	// Guess positions of two canthi offset from eye-center
	vector<Point> canthi;
	Point2i v1 = v_norm * eye_dist * CANTH_HORIZ_RATIO, v2 = v_perp * eye_dist * CANTH_VERT_RATIO;	// Use copy-initialize to avoid overload ambiguity
	canthi.push_back(eye_center - v1 - v2);
	canthi.push_back(eye_center + v1 - v2);

	return canthi;
}

// Offset gradient maxima to push eyelid below incorrect limbus pts
const double EYELID_OFFSET_RATIO = 0.1;

// Takes the point where dark eye-shadow meets lighter skin in the pixel column of the eye-center
Point find_upper_eyelid_pt(Point2i& eye_center, const Mat& eye_c_col){

	Mat filter_col;
	Sobel(eye_c_col, filter_col, -1, 0, 1, 3);

	// Find filter maximum in the top half of the ROI
	int maxIdx[2];
	minMaxIdx(filter_col.rowRange(0, filter_col.rows/3), NULL, NULL, NULL, maxIdx);

	return Point(eye_center.x, eye_center.y - eye_c_col.rows/2 + maxIdx[0] + eye_c_col.rows*EYELID_OFFSET_RATIO);
}

Mat get_upper_eyelid(Point2i& eye_center, Point2i& eye_cs_vec, Mat& eye_bgr){

	Mat eye_grey;
	cvtColor(eye_bgr, eye_grey, CV_BGR2GRAY);
	GaussianBlur(eye_grey,eye_grey,Size(5,5),0);

	// Invert color to find correct dark > light transition
	eye_grey = 255-eye_grey;

	// pts is 3-long vector for internal and external canthi, and upper-eyelid point
	vector<Point> pts = guess_eye_corners(eye_center, eye_cs_vec);
	pts.push_back(find_upper_eyelid_pt(eye_center, eye_grey.col(eye_bgr.size().width/2)));

	// Make matrices and solve for parabola intersecting the 3 points
	float mat_a[3][3] = {
		{pts[0].x * pts[0].x, pts[0].x, 1},
		{pts[1].x * pts[1].x, pts[1].x, 1},
		{pts[2].x * pts[2].x, pts[2].x, 1}};

	float mat_b[3][1] = {
		{pts[0].y},
		{pts[1].y},
		{pts[2].y}};

	Mat A(3, 3, CV_32F, &mat_a), B(3, 1, CV_32F, &mat_b);

	// Return parabola in the form of a*x^2 + b*x + c as ( Mat of [a, b, c] )
	return A.inv() * B;
}

vector<Point2f> filter_poss_limb_pts(vector<Point2f>& poss_limb_pts, Mat& parabola, Point2i& roi_tl){
	vector<Point2f> pts_to_return;

	float a = parabola.at<float>(0), b = parabola.at<float>(1), c = parabola.at<float>(2); // Get parabola params

	float x, y;
	for (Point2f& p : poss_limb_pts){
		y = p.y + roi_tl.y, x = p.x + roi_tl.x;					// Shift limbus points from refined eye-roi coords
		if (y > a*x*x + b*x + c) pts_to_return.push_back(p);	// Only return a point if below upper eyelid parabola
	}
	
	return pts_to_return;
}

const int EYELID_PARABOLA_GAP = 5;
void draw_eyelid(Mat& img, Mat& parabola, Rect& roi, Scalar color = RED){

	float a = parabola.at<float>(0), b = parabola.at<float>(1), c = parabola.at<float>(2); // Get parabola params

	// Approximate eyelid with a poly-line, at x-intervals of EYELID_PARABOLA_GAP
	for (int x=roi.x; x<roi.x+roi.width; x += EYELID_PARABOLA_GAP){
		int x_next = min(x+EYELID_PARABOLA_GAP,(roi.x+roi.width-1));
		Point p1(x,a*x*x + b*x + c), p2(x_next,a*x_next*x_next + b*x_next + c);
		line(img, p1, p2, color);
	}
}

