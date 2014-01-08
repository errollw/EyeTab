#include "stdafx.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <time.h>       // clock_t, clock, CLOCKS_PER_SEC

#include <math.h>

using namespace std;
using namespace cv;

Mat map_x, map_y;

int fixed_size = 150;
int min_limb_r = int(fixed_size * 0.2);
int max_limb_r = int(fixed_size * 0.5);
int angle_ignore = 30;

Size polar_size(360,100);

// Gabor kernel approximating a blurred y-derivative
Mat gabor_kern = getGaborKernel(Size(7,7), 2.0, CV_PI/2.0, CV_PI * 2, 2, CV_PI/2.0, CV_32F);

void lin_polar_init() {

	// Create re-mapping matricies
	map_x.create( polar_size, CV_32FC1 );
	map_y.create( polar_size, CV_32FC1 );

	// Fill re-mapping matricies with reverse transform to cart. coords
	for(int y = 0; y < map_x.rows; y++) {
        for(int x = 0; x < map_x.cols; x++) {

			float t_rad = x * CV_PI / 180;

			int i = (int) fixed_size/2 + y * sin(t_rad);
            int j = (int) fixed_size/2 + y * cos(t_rad);

            map_x.at<float>(y,x) = j;
			map_y.at<float>(y,x) = i;
        }
    }
}

vector<Point2f> get_poss_limb_pts(Mat eye_bgr) {

	Mat eye_grey;
	cvtColor(eye_bgr, eye_grey, CV_BGR2GRAY);
	medianBlur(eye_grey, eye_grey, 5);

	// Scale to fixed size image for re-using transform matrix
	float scale = eye_grey.size().width / (float) fixed_size;
	resize(eye_grey, eye_grey, Size(fixed_size,fixed_size));

	// Transform image into polar coords and blur
	Mat eye_polar = Mat( polar_size, eye_grey.type() );
	remap( eye_grey, eye_polar, map_x, map_y, CV_INTER_LINEAR );
	GaussianBlur(eye_polar, eye_polar, Size(5,5), 0);

	// Filter polar image
	Mat eye_filter;
	filter2D(eye_polar, eye_filter, -1, gabor_kern);
	
	// Normalize filtered image
	normalize(eye_filter, eye_filter, 0, 255, NORM_MINMAX);
	eye_filter.convertTo(eye_filter, CV_8U);

	// Blank-out ignored radii (rows in radial image)
	eye_filter.rowRange(0, min_limb_r).setTo(0);
	eye_filter.rowRange(max_limb_r, eye_filter.rows).setTo(0);

	int maxIdx[2];
	vector<float> mags, thts, xs, ys;
	for (int i = 0; i < eye_filter.cols; i++){

		// Skip ignore angle region
		if (i == 90-angle_ignore) i = 90+angle_ignore;
		if (i == 270-angle_ignore) i = 270+angle_ignore;

		minMaxIdx(eye_filter.col(i), NULL, NULL, NULL, maxIdx);

		thts.push_back(i);
		mags.push_back(maxIdx[0]);
	}
	
	polarToCart(mags, thts, xs, ys, true);

	vector<Point2f> poss_limb_pts;
	for (int i=0; i < thts.size(); i++){
		poss_limb_pts.push_back((Point2f(xs[i], ys[i]) + Point2f(fixed_size/2,fixed_size/2)) * scale);
	}

	return poss_limb_pts;
}

void test_lin_polar(){

	// Load test image
	Mat eye_bgr = imread("images\\poppy.jpg", CV_LOAD_IMAGE_COLOR);

	lin_polar_init();

	vector<Point2f> poss_limb_pts = get_poss_limb_pts(eye_bgr);

	waitKey();
}