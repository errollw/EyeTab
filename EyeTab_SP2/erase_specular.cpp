#include "stdafx.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <iostream>     // cout
#include <stdio.h>
#include <time.h>       // clock_t, clock, CLOCKS_PER_SEC

using namespace std;
using namespace cv;

const Mat ERASE_SPEC_KERNEL = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

void erase_specular(Mat& eye_bgr) {

	// Rather arbitrary decision on how large a specularity may be
	int max_spec_contour_area = (eye_bgr.size().width + eye_bgr.size().height)/2;

	// Convert BGR coarse ROI to gray, blur it slightly to reduce noise
	Mat eye_grey;
	cvtColor(eye_bgr, eye_grey, CV_BGR2GRAY);
	GaussianBlur(eye_grey, eye_grey, Size(5,5), 0);

	// Close to suppress eyelashes
	morphologyEx(eye_grey, eye_grey, MORPH_CLOSE, ERASE_SPEC_KERNEL);

	// Compute thresh value (using of highest and lowest pixel values)
	double m, M; // m(in) and (M)ax values in image
	minMaxLoc(eye_grey, &m, &M, NULL, NULL);
	double thresh = (m + M) * 3/4;

	// Threshold the image
	Mat eye_thresh;
	threshold(eye_grey, eye_thresh, thresh, 255, THRESH_BINARY);

	// Find all contours in threshed image (possible specularities)
	vector< vector<Point> > all_contours, contours;
	findContours(eye_thresh, all_contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	// Only save small ones (assumed to be spec.s)
	for (int i=0; i<all_contours.size(); i++){
		if( contourArea(all_contours[i]) < max_spec_contour_area )
			contours.push_back(all_contours[i]);
	}

	// Draw the contours into an inpaint mask
	Mat small_contours_mask = Mat::zeros(eye_grey.size(), eye_grey.type());
	drawContours(small_contours_mask, contours, -1, 255, -1);
	dilate(small_contours_mask, small_contours_mask, ERASE_SPEC_KERNEL);

	// Inpaint within contour bounds
	inpaint(eye_bgr, small_contours_mask, eye_bgr, 2, INPAINT_TELEA);
}

void test_erase_specular() {

	// Load test image and convert to gray
	Mat eye_bgr, eye_grey, eye_grey_small;
	eye_bgr = imread("images\\erroll1_l.png", CV_LOAD_IMAGE_COLOR);

	erase_specular(eye_bgr);
}