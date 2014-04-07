import numpy as np

NEXUS_7 = 0x1
NEXUS_7_INV = 0x2
WEBCAM = 0x3

# Camera coefficients
dist_coefs_n7 = np.array([[-0.04784793, -0.21276658, -0.00432402, 0.00086078, 0.69334691]])

cam_mat_n7 = np.array([[1062.348, 0.0     , 344.629],
                       [0.0     , 1065.308, 626.738],
                       [0.0     , 0.0     , 1.0]])

# Screen size constants
screen_w_mm_n7, screen_h_mm_n7 = 94, 151
screen_w_px_n7, screen_h_px_n7 = 800, 1280
screen_y_offset_px_n7 = 32                     # For notification bar at top of screen

class Device:
    
    def __init__(self, device_type):
        
        self.device_type = device_type
        
        # General Nexus 7 settings
        if device_type == NEXUS_7 or device_type == NEXUS_7_INV:
            self.fx, self.fy = cam_mat_n7[0][0], cam_mat_n7[1][1]
            self.cx, self.cy = cam_mat_n7[0][2], cam_mat_n7[1][2]
            self.screen_size_mm = screen_w_mm_n7, screen_h_mm_n7
            self.screen_size_px = screen_w_px_n7, screen_h_px_n7
            self.screen_y_offset_px = screen_y_offset_px_n7
            self.offset_mm = 47, 16
            self.rot90s = 1
            self.mirror = False
        
        # Special settings for inverted Nexus 7    
        if device_type == NEXUS_7_INV:
            self.offset_mm = 47, -(self.screen_size_mm[1] + 16)
            self.rot90s = -1
            
        # Dummy values for Webcam (use N7 as default), these have not been calibrated
        if device_type == WEBCAM:
            self.fx, self.fy = cam_mat_n7[0][0], cam_mat_n7[1][1]
            self.cx, self.cy = cam_mat_n7[0][0], cam_mat_n7[1][1]
            self.screen_size_mm = screen_w_mm_n7, screen_h_mm_n7
            self.screen_size_px = screen_w_px_n7, screen_h_px_n7
            self.screen_y_offset_px = 0
            self.offset_mm = 0, 0
            self.rot90s = 0
            self.mirror = True
        
    def get_intrisic_cam_params(self):
        return self.fx, self.fy, self.cx, self.cy
    
    def get_dist_coeffs(self):
        return dist_coefs_n7
