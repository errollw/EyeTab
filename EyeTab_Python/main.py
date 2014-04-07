import cv2
import numpy as np
import os
import socket
import gaze_system as gaze_system
import device_constants

from datetime import datetime

host = '192.168.137.195'
camera_stream_port = 8080
device_control_port = 9999

# Dummy parameter '?dummy=video.mjpg' is used to trick OpenCV into opening stream
# see: http://stackoverflow.com/q/14204185/2023516
mjpg_stream_url = 'http://%s:%d/?dummy=param.mjpg' % (host, camera_stream_port)

vid_path = 'C:\\Users\\Erroll\\Documents\\Part 3 Project (local)\\Gaze Data\\P03\\P03_L2_D2_20130517_161358.mp4'

use_network_stream = False
use_webcam = False
use_local_video = not (use_network_stream or use_webcam)

debug = False
recording = False

marker_flags_ms = [3500] + [x for x in range(7500, 50000, 4000)]

if __name__ == '__main__':
    
    paused = False;
    
    while True:
        
        device_control_socket = None
        stream_open = False
        
        while stream_open is False:
            
            if use_webcam:
                vc = cv2.VideoCapture(0)
                vc.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280) # Try to force HD resolution
                vc.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720) # No success with Microsoft Lifecam on Win8
            elif use_network_stream:
                vc = cv2.VideoCapture(mjpg_stream_url)
            elif use_local_video:
                vc = cv2.VideoCapture(vid_path)
                
                # If file is variable frame-rate, warn
                frames, fps =  vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT), vc.get(cv2.cv.CV_CAP_PROP_FPS)
                if abs(48-frames / fps) > 3:
                    print 'WARNING - VARIABLE FRAME RATE'
                
            stream_open, frame = vc.read()
            
            if not stream_open:
                print 'Failed to open stream @ %s' % datetime.now().strftime('%X')
                cv2.waitKey(5000)   # Wait 5s before trying to open VideoCapture again

        # VideoCapture is open from now onwards
        print 'Successfully opened stream @ %s' % datetime.now().strftime('%X')
        
        # For managing marker movement
        if use_local_video:
            active_marker_ind = 0
        
        device = device_constants.Device(device_constants.WEBCAM if use_webcam else device_constants.NEXUS_7_INV)
        
        if use_network_stream:
            device_control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            device_control_socket.connect((host, device_control_port))
            
        g_sys = gaze_system.GazeSystem(device, debug, recording)
        
        # Activate 1st marker
        g_sys.activate_marker(active_marker_ind)                                    
        
        while stream_open:
            
            frame = np.rot90(frame, device.rot90s)
            
            # Increment activated marker
            ms_passed = vc.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
            if(ms_passed >= marker_flags_ms[active_marker_ind]):
                active_marker_ind += 1
                g_sys.activate_marker(active_marker_ind)
            
            gaze_pt = g_sys.get_gaze_from_frame(frame)
            
            if use_network_stream and gaze_pt is not None:
                (x, y) = gaze_pt
                device_control_socket.sendall('%d %d \n' % (int(x), int(y)))
            
            stream_open, frame = vc.read()
            
            # Pause on pressing P, p or [space]
            key = cv2.waitKey(5)
            if key in [112, 80, 32]: cv2.waitKey(0)
        
        # Stream is now closed, clean up
        print 'Stream interrupted @ %s' % datetime.now().strftime('%X')
        
        if device_control_socket is not None: device_control_socket.close()
