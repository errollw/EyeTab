#include "stdafx.h"

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
 
#include "eye_center.h"
#include "erase_specular.h"
#include "get_poss_limb_pts.h"
#include "fit_ellipse.h"
#include "utils.h"
#include "get_eyelids.h"
#include "gaze_system.h"
#include "gaze_smoothing.h"

using namespace std;
using namespace cv;

int num_screenshots = 0;
String screenshot_filename;

int main(int argc, const char** argv)
{
	// initialize modules
	lin_polar_init();
	gaze_system_init();
	gaze_smoothing_init();

    // start reading in camera frames (720p video)
    VideoCapture cap("videos\\sample_video.avi");
 
    // setup image files used in the capture process
    Mat captureFrame, grayscaleFrame, smallFrame;
 
    // create a window to present the results
    namedWindow("output", 1);
 
    // main loop, terminates when out of frames
    while(cap.isOpened()) {
		clock_t start = clock();

        // read in a new image frame
        cap >> captureFrame;
 
        // convert captured image to equalized gray scale
        cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
        equalizeHist(grayscaleFrame, grayscaleFrame);

		track_gaze(captureFrame, grayscaleFrame);
 
		// show calculated FPS (two draw functions to simulate text shadow)
		String fps_string = to_string(int(1 / ( ((float)clock()-start) / CLOCKS_PER_SEC ))) + " FPS";
		putText(captureFrame, fps_string, Point2i(11, 21), FONT_HERSHEY_SIMPLEX, 0.5, BLACK);
		putText(captureFrame, fps_string, Point2i(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, WHITE);

		// Show the output
        imshow("output", captureFrame);

		switch (waitKey(1)){
			case 's':
				screenshot_filename = "SS_" + to_string(num_screenshots++);
				imwrite(screenshot_filename + ".jpg", captureFrame);
				break;
			case 'q': return 0;
		}
    }
 
    return 0;
}