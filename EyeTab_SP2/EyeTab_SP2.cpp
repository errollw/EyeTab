#include "stdafx.h"
#include "videoInput.h"

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


void StopEvent(int deviceID, void *userData) {
    videoInput vidInput = videoInput::getInstance();
    vidInput.closeDevice(deviceID);
}
 

int main(int argc, const char** argv) {

	// initialize modules
	lin_polar_init();
	gaze_system_init();
	gaze_smoothing_init();

	// intialize camera API 
    videoInput vidInput = videoInput::getInstance();
    int num_devices = vidInput.listDevices();
	int cam_id = 0;

	// check we actually have attached devices, if not exit
	if (num_devices == 0){
		return -1;		
	}

	// if we fail to setup a camera, exit
	if(!vidInput.setupDevice(cam_id, 1280, 720, 60)) {
		return -1;	
	}

	// extract camera parameters, and set a default brightness
	CamParametrs CP = vidInput.getParametrs(cam_id);                        
    CP.Brightness.CurrentValue = 180; 
    CP.Brightness.Flag = 0; 
    vidInput.setParametrs(cam_id, CP);

	// sets callback function for emergency stop, e.g. pulling out camera
    vidInput.setEmergencyStopEvent(cam_id, NULL, StopEvent);
 
    // create a window to present the results
    namedWindow("output", 1);
    CvSize size = cvSize(vidInput.getWidth(cam_id), vidInput.getHeight(cam_id));

	// VideoInput library copies data into IplImage
    IplImage* frame;
    frame = cvCreateImage(size, 8, 3);

	// setup image files used in the capture process
    Mat captureFrame, grayscaleFrame, smallFrame;
 
	// ##################
	// MAIN LOOP
    // ##################
	while(true) {

        if(vidInput.isFrameNew(cam_id)) {
			clock_t start = clock();

			// copy data into the IplImage frame
            vidInput.getPixels(cam_id, (unsigned char *)frame->imageData);      

			// make a Mat of the IplImage
			Mat captureFrame(frame);

			// flip image for reverse-portrait orientation
			flip(captureFrame, captureFrame, -1);

			// convert captured image to equalized gray scale
			cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
			equalizeHist(grayscaleFrame, grayscaleFrame);

			track_gaze(captureFrame, grayscaleFrame);
 
			// show calculated FPS (two draw functions to simulate text shadow)
			String fps_string = to_string(int(1 / ( ((float)clock()-start) / CLOCKS_PER_SEC ))) + " FPS";
			putText(captureFrame, fps_string, Point2i(11, 21), FONT_HERSHEY_SIMPLEX, 0.5, BLACK);
			putText(captureFrame, fps_string, Point2i(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, WHITE);

			// resize and show the output
			resize(captureFrame, captureFrame, Size(1920,1080));
			imshow("output", captureFrame);
        }
 
        char c = cv::waitKey(5);
 
        if(c == 27) {
            break;
		} else if(c == 49) {                     
            CP.Brightness.CurrentValue = 50; 
            CP.Brightness.Flag = 1; 
            vidInput.setParametrs(cam_id, CP);
        }else if(c == 50) {                      
            CP.Brightness.CurrentValue = 0; 
            CP.Brightness.Flag = 2; 
            vidInput.setParametrs(cam_id, CP);
        }
 
        if(!vidInput.isDeviceSetup(cam_id)) {
            break;
        }
    }
 
	// tidy up on exit
    vidInput.closeDevice(cam_id);
    cv::destroyAllWindows();

    
    return 0;
}