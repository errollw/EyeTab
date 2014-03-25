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
 
	// setup image files used in the capture process
    Mat captureFrame, grayscaleFrame, smallFrame;
 
    // create a window to present the results
    namedWindow("output", 1);
    CvSize size = cvSize(vidInput.getWidth(cam_id), vidInput.getHeight(cam_id));

	// VideoInput library uses IplImage
    IplImage* frame;
    frame = cvCreateImage(size, 8, 3);
 
	// main loop
    while(true) {

        if(vidInput.isFrameNew(cam_id)) {
            vidInput.getPixels(cam_id, (unsigned char *)frame->imageData);      

			cv::Mat image(frame);
			cv::imshow("output", image);
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