#include "stdafx.h"

#include <iostream>
#include <stdio.h>

#include <Math.h>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

const int SMOOTHING_WINDOW_SIZE = 10;

vector<Point2d> point_history;

vector<float> weights;
float weight_sum = 0;

// create and initialize weights and history vectors
void gaze_smoothing_init(){
	for (float i = 1; i < SMOOTHING_WINDOW_SIZE; i++){
		weights.push_back(i);
		weight_sum += i;
		point_history.push_back(Point2d(0,0));
	}
}

// very simple smoothing low-pass filter
Point2d smooth_gaze(Point2d gaze_point){
	
	Point2d point_to_return(0,0);

	rotate(point_history.begin(), point_history.begin() + 1, point_history.end());
	point_history[point_history.size()-1] = gaze_point;

	for (int i = 0; i < weights.size(); i++)
		point_to_return += point_history[i] * (weights[i]/weight_sum);

	return point_to_return;
}