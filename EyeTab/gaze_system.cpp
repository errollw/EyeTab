#include "stdafx.h"

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
#include <iostream>
#include <stdio.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <math.h>       /* floor */
 
#include "eye_center.h"
#include "erase_specular.h"
#include "get_poss_limb_pts.h"
#include "fit_ellipse.h"
#include "utils.h"
#include "get_eyelids.h"
#include "gaze_geometry.h"
#include "gaze_smoothing.h"

using namespace std;
using namespace cv;

/** Global variables */
const String CASCADE_FILENAME = "haarcascade_mcs_eyepair_big.xml";
const int SMALL_FRAME_WIDTH = 256;

//                              eye_1     eye_2
//                        skin  |    nose |    skin
//                        |     |    |    |    |
float EYE_PART_RATIOS [] = {0.05, 0.3, 0.3, 0.3, 0.05};

// Ratio of eye-pair distance to refined eye-ROI size
float EYE_PAIR_DIST_ROI_RATIO = 0.15;

const Size MIN_EYE_PAIR_SIZE = Size(67, 17);

CascadeClassifier eyepair_cascade;

void gaze_system_init() {
	eyepair_cascade.load(CASCADE_FILENAME);
}

// Chooses biggest eye-pair rectangle
Rect choose_best_eye_pair(vector<Rect> eye_pairs){
	Rect best_rect = eye_pairs[0];
	for (Rect& r : eye_pairs){
		if (r.area() > best_rect.area()) best_rect = r;
	}
	return best_rect;
}

void track_gaze(Mat& frame_bgr, Mat& frame_gray) {

	// Calculate smallsize and its scale based on aspect ratio
	Mat smallFrame;
	float scale = SMALL_FRAME_WIDTH / (float)frame_gray.size().width;
	resize(frame_gray, smallFrame, Size(0,0), scale, scale);
 
    // Find eye_pairs and store them in the vector
    vector<Rect> eye_pairs;
	eyepair_cascade.detectMultiScale(smallFrame, eye_pairs, 1.2, 3, 0, MIN_EYE_PAIR_SIZE);

	if (eye_pairs.size() == 0) return;

	Rect eye_pair = eye_pairs[0];

    // Start and end pts for eye-pair ROI in full-frame
    Point roi_start(eye_pair.x / scale, eye_pair.y / scale);
	Point roi_end((eye_pair.x + eye_pair.width)/scale, (eye_pair.y + eye_pair.height)/scale);
	int width = roi_end.x - roi_start.x;

	// Extract two coarse eye-rois from the eye-pair roi
	Rect eye0_roi(roi_start.x + width*EYE_PART_RATIOS[0], roi_start.y,
		width*(EYE_PART_RATIOS[1]-EYE_PART_RATIOS[0]), roi_end.y - roi_start.y);
	Rect eye1_roi(roi_start.x + width*(EYE_PART_RATIOS[0]+EYE_PART_RATIOS[1]+EYE_PART_RATIOS[2]), roi_start.y,
		width*(EYE_PART_RATIOS[3]-EYE_PART_RATIOS[4]), roi_end.y - roi_start.y);
	
	// Inpaint small specular reflections
	erase_specular(frame_bgr(eye0_roi));
	erase_specular(frame_bgr(eye1_roi));

	// Find relative eye-centers for each eye-roi
	Point c0 = find_eye_center(frame_bgr(eye0_roi));
	Point c1 = find_eye_center(frame_bgr(eye1_roi));

	// Get vector and inter-eye distance between eye-centers
	Point eye_cs_vec = (eye0_roi.tl() + c0)-(eye1_roi.tl() + c1);
	double eye_centre_dist = norm(eye_cs_vec);

	// Refine eye-rois about detected eye-center
	Rect eye0_roi_ref = roiAround(eye0_roi.tl() + c0, (int) (eye_centre_dist * EYE_PAIR_DIST_ROI_RATIO));
	Rect eye1_roi_ref = roiAround(eye1_roi.tl() + c1, (int) (eye_centre_dist * EYE_PAIR_DIST_ROI_RATIO));

	// If a ROI lies outside capture frame bounds, terminate early
	Rect capFrameRect = Rect(Point(0,0), frame_bgr.size());
	if (eye0_roi_ref != (capFrameRect & eye0_roi_ref) || eye1_roi_ref != (capFrameRect & eye1_roi_ref))
		return;

	// Get inital possible limbus points (including erroneous upper-eyelid points)
	vector<Point2f> poss_limb_pts_0 = get_poss_limb_pts(frame_bgr(eye0_roi_ref));
	vector<Point2f> poss_limb_pts_1 = get_poss_limb_pts(frame_bgr(eye1_roi_ref));

	// Get upper-eyelid parabolae for each eye, and filter poss-limbus points
	Mat eye0_eyelid = get_upper_eyelid(eye0_roi.tl() + c0, eye_cs_vec, frame_bgr(eye0_roi_ref));
	Mat eye1_eyelid = get_upper_eyelid(eye1_roi.tl() + c1, eye_cs_vec, frame_bgr(eye1_roi_ref));
	poss_limb_pts_0 = filter_poss_limb_pts(poss_limb_pts_0, eye0_eyelid, eye0_roi_ref.tl());
	poss_limb_pts_1 = filter_poss_limb_pts(poss_limb_pts_1, eye1_eyelid, eye1_roi_ref.tl());

	// Calculate image gradients, and fit ellipse for each set of filtered poss-limbus points
	Mat grad_x, grad_y;
	Sobel(frame_gray(eye0_roi_ref), grad_x, CV_32F, 1, 0, 5);
	Sobel(frame_gray(eye0_roi_ref), grad_y, CV_32F, 0, 1, 5);
	RotatedRect ellipse0 = fit_ellipse(poss_limb_pts_0, grad_x, grad_y);
	Sobel(frame_gray(eye1_roi_ref), grad_x, CV_32F, 1, 0, 5);
	Sobel(frame_gray(eye1_roi_ref), grad_y, CV_32F, 0, 1, 5);
	RotatedRect ellipse1 = fit_ellipse(poss_limb_pts_1, grad_x, grad_y);

	// Shift ellipses fit by their ROI offset
	ellipse0 = RotatedRect(Point2i(ellipse0.center) + eye0_roi_ref.tl(), ellipse0.size, ellipse0.angle);
	ellipse1 = RotatedRect(Point2i(ellipse1.center) + eye1_roi_ref.tl(), ellipse1.size, ellipse1.angle);

	/*Point2d gaze_pt_mm = (get_gaze_pt_mm(ellipse0) + get_gaze_pt_mm(ellipse1)) * 0.5;
	gaze_pt_mm = smooth_gaze(gaze_pt_mm);
	Point gaze_pt_px = convert_gaze_pt_mm_to_px(gaze_pt_mm);
	cout << "GAZE PT MM " << gaze_pt_mm.x << " " << gaze_pt_mm.y << endl;
	cout << "GAZE PT PX " << gaze_pt_px.x << " " << gaze_pt_px.y << endl;*/

	Point2d gaze_pt_mm_0 = get_gaze_pt_mm(ellipse0);
	Point2d gaze_pt_mm_1 = get_gaze_pt_mm(ellipse1);
	Point2i gaze_pt_px_0 = convert_gaze_pt_mm_to_px(gaze_pt_mm_0);
	Point2i gaze_pt_px_1 = convert_gaze_pt_mm_to_px(gaze_pt_mm_1);

	cout << "GAZE PT 1 " << gaze_pt_mm_0 << " " << gaze_pt_px_0 << endl;
	cout << "GAZE PT 2 " << gaze_pt_mm_1 << " " << gaze_pt_px_1 << endl;

	vector<Point2i> gp_px_s;
	gp_px_s.push_back(gaze_pt_px_0);
	gp_px_s.push_back(gaze_pt_px_1);
	vector<Scalar> colors;
	colors.push_back(RED);
	colors.push_back(BLUE);

	show_gaze(frame_bgr, gp_px_s, colors);

	// *** DEBUG DRAWING ***

	// DEBUG DRAWING - coarse ROIs
	rectangle(frame_bgr, eye0_roi, GREEN);
	rectangle(frame_bgr, eye1_roi, GREEN);

	// DEBUG DRAWING - eye-centres
	cross(frame_bgr, c0 + eye0_roi.tl(), 3, RED);
	cross(frame_bgr, c1 + eye1_roi.tl(), 3, RED);

	// DEBUG DRAWING - refined ROIs
    rectangle(frame_bgr, eye0_roi_ref, RED);
	rectangle(frame_bgr, eye1_roi_ref, RED);

	// DEBUG DRAWING - eyelids
	draw_eyelid(frame_bgr, eye0_eyelid, eye0_roi_ref);
	draw_eyelid(frame_bgr, eye1_eyelid, eye1_roi_ref);

	// DEBUG DRAWING - limbus ellipses
	ellipse(frame_bgr, ellipse0, YELLOW, 1);
	ellipse(frame_bgr, ellipse1, YELLOW, 1);

	// DEBUG DRAWING - draw possible limb pts
	for (Point2f& p : poss_limb_pts_0)
		circle(frame_bgr, Point2i(p) + eye0_roi_ref.tl(), 1, YELLOW, -1);
	for (Point2f& p : poss_limb_pts_1)
		circle(frame_bgr, Point2i(p) + eye1_roi_ref.tl(), 1, YELLOW, -1);

}