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
#include "gaze_system.h"
#include "gaze_smoothing.h"

using namespace std;
using namespace cv;

int num_screenshots = 0;
String screenshot_filename;

int main(int argc, const char** argv)
{
	// init modules
	lin_polar_init();
	gaze_system_init();
	//init_gaze_smoothing();

    //setup video capture device
    // VideoCapture cap("C:\\Users\\Erroll\\Documents\\Part 3 Project (local)\\Gaze Data\\P01\\P01_L2_D1_20130517_133459.mp4");
	// VideoCapture cap("C:\\Users\\Erroll\\Documents\\Part 3 Project (local)\\Gaze Data\\P03\\P03_L1_D1_20130517_163141.mp4");
	// VideoCapture cap("C:\\Users\\Erroll\\Documents\\Part 3 Project (local)\\Gaze Data\\P06\\P06_L1_D1_20130520_154535.mp4");

	int cam_idx = argc > 1 ? atoi(argv[1]) : 0;
	VideoCapture cap(cam_idx);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	bool rot_180 = argc > 2 ? true : false;
 
    //setup image files used in the capture process
    Mat captureFrame, grayscaleFrame, smallFrame;
 
    //create a window to present the results
    namedWindow("outputCapture", 1);
	//namedWindow("Gaze Output", 1);
	//moveWindow("Gaze Output", 0, 0);
 
    //create a loop to capture and find eye-pairs
    while(true)
    {
		clock_t start = clock();

        //capture a new image frame
        cap >> captureFrame;
		// captureFrame = imread("images\\SS_0.jpg", CV_LOAD_IMAGE_COLOR);
		// flip(captureFrame,captureFrame,1);

		if (rot_180)
			flip(captureFrame, captureFrame, -1);
 
        //convert captured image to gray scale and equalize
        cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
        equalizeHist(grayscaleFrame, grayscaleFrame);

		track_gaze(captureFrame, grayscaleFrame);
 
		// Show calculated FPS
		String fps_string = to_string(int(1 / ( ((float)clock()-start) / CLOCKS_PER_SEC ))) + " FPS";
		putText(captureFrame, fps_string, Point2i(11, 21), FONT_HERSHEY_SIMPLEX, 0.5, BLACK);
		putText(captureFrame, fps_string, Point2i(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, WHITE);

		// Show the output
        imshow("outputCapture", captureFrame);

		switch (waitKey(5)){
		case 's':
			screenshot_filename = "SS_" + to_string(num_screenshots++);
			imwrite(screenshot_filename + ".jpg", captureFrame);
			break;
		case 'q': return 0;
		}
    }
 
    return 0;
}