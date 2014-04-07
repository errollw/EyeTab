import cv2
import numpy as np
import image_utils
import draw_utils

import eye_extractor
import pre_processing
import eye_center_locator_combined
import eyelid_locator 
import ransac_ellipse
import gaze_geometry
import gaze_smoothing

from eyelid_locator import find_eyelids
from find_limbus_points import get_limb_pts

import visualize_in_3d
from conic_section import Ellipse

import limbus_outlier_removal

winname = 'Gaze System'

#                eye1 - user's right (cyan)
#                |              eye2 - user's left (magenta)
#                |              |              smoothed gaze point (yellow)
#                |              |              |
debug_colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]

# ONLY FOR USE WITH 720p PORTRAIT VIDEOS
def make_vid_writer_gaze_sys():
    return cv2.VideoWriter(filename='gazeVisualization.avi',
                           fourcc=cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'),
                           fps=13,
                           frameSize=(1060, 640))  # (WIDTH, HEIGHT)

# Camera coeffs (distortion & intrinsic)
dist_coefs_n7 = np.array([[-0.04784793, -0.21276658, -0.00432402, 0.00086078, 0.69334691]])

cam_mat_n7 = np.array([[1062.348, 0.0     , 344.629],
                       [0.0     , 1065.308, 626.738],
                       [0.0     , 0.0     , 1.0]])

class GazeSystem:

    def __init__(self, device, debug=False, recording=False, init_vpython=True, filename=None):

        self.device = device
        self.cam_mat = device.get_intrisic_cam_params()
        self.dist_coeffs = device.get_dist_coeffs()
        
        self.debug = debug
        
        self.recording = False
        if self.recording: self.vid_writer = make_vid_writer_gaze_sys()
        
        self.visualizer3d = visualize_in_3d.Visualizer3d(win_size=(700, 640),
                                                         device=device,
                                                         filename=filename)
        
        self.pre_proc = pre_processing.PreProcessor()
        self.smoother = gaze_smoothing.GazeSmoother(8, gaze_smoothing.TRIANGLE_WEIGHTS)
        
    def activate_marker(self, marker_index):
        self.visualizer3d.activate_marker(marker_index)

    def get_gaze_from_frame(self, frame):
        
        frame = cv2.undistort(frame, cam_mat_n7, dist_coefs_n7)
        
        frame_pyr = image_utils.make_gauss_pyr(frame, 4)
        full_frame = frame_pyr[1].copy()
        half_frame = frame_pyr[2].copy()
        
        limbuses = [None, None]
        gaze_pts_mm = [None, None]
        gaze_pts_px = [None, None]
        
        try:
            sub_img_cx0, sub_img_cy0 = None, None
            eye_r_roi, eye_l_roi = eye_extractor.get_eye_rois(frame_pyr, 4, debug=self.debug, device=self.device)
            
            for i, eye_roi in enumerate([eye_r_roi, eye_l_roi]):
                
                try:
                    if eye_roi.img is None: break
                    
                    # Gives unique winnames for each ROI
                    debug_index = ((i + 1) if self.debug else False)  
            
                    eye_roi.img = self.pre_proc.erase_specular(eye_roi.img, debug=debug_index)
            
                    pupil_x0, pupil_y0 = eye_center_locator_combined.find_pupil(eye_roi.img,
                                                                                fast_width_grads=25.0,
                                                                                fast_width_iso=80.0,
                                                                                weight_grads=0.8,
                                                                                weight_iso=0.2,
                                                                                debug_index=debug_index)
                    eye_roi.refine_pupil((pupil_x0, pupil_y0), full_frame)
                    roi_x0, roi_y0, roi_w, roi_h = eye_roi.roi_x0, eye_roi.roi_y0, eye_roi.roi_w, eye_roi.roi_h
                    
                    u_eyelid, l_eyelid = find_eyelids(eye_roi.img, debug_index)
                    
                    pts_found = get_limb_pts(eye_img=eye_roi.img,
                                             phi=20,
                                             angle_step=1,
                                             debug_index=debug_index)
                    pts_found = eyelid_locator.filter_limbus_pts(u_eyelid, l_eyelid, pts_found)
                    
                    ellipse = ransac_ellipse.ransac_ellipse_fit(points=pts_found,
                                                                bgr_img=eye_roi.img,
                                                                roi_pos=(roi_x0, roi_y0),
                                                                ransac_iters_max=5,
                                                                refine_iters_max=3,
                                                                max_err=1,
                                                                debug=False)
                    
                    # Shift 2D limbus ellipse and points to account for eye ROI coords
                    (ell_x0, ell_y0), (ell_w, ell_h), angle = ellipse.rotated_rect               
                    new_rotated_rect = (roi_x0 + ell_x0, roi_y0 + ell_y0), (ell_w, ell_h), angle
                    ellipse = Ellipse(new_rotated_rect)                                                                                
                    pts_found_to_draw = [(px + roi_x0, py + roi_y0) for (px, py) in pts_found]
                    
                    # Correct coords when extracting eye for half-frame
                    (sub_img_cx0, sub_img_cy0) = (roi_x0 + ell_x0, roi_y0 + ell_y0)
                    
                    # Ignore incorrect limbus
                    limbus = gaze_geometry.ellipse_to_limbuses_persp_geom(ellipse, self.device)
                    limbuses[i] = limbus
                    
                    # Draw eye features onto debug image
                    draw_utils.draw_limbus(full_frame, limbus, color=debug_colors[i], scale=1)
                    draw_utils.draw_points(full_frame, pts_found_to_draw, color=debug_colors[i], width=1, thickness=2)
                    draw_utils.draw_normal(full_frame, limbus, self.device, color=debug_colors[i], scale=1)
                    draw_utils.draw_normal(half_frame, limbus, self.device, color=debug_colors[i], scale=0.5, arrow_len_mm=20)
                    eye_img = full_frame[eye_roi.roi_y0:(eye_roi.roi_y0 + eye_roi.roi_h),
                                         eye_roi.roi_x0:(eye_roi.roi_x0 + eye_roi.roi_w)]
                    draw_utils.draw_eyelids(u_eyelid, l_eyelid, eye_img)
                    
                except ransac_ellipse.NoEllipseFound:
                    if self.debug: print 'No Ellipse Found'
                    cv2.rectangle(full_frame, (roi_x0, roi_y0), (roi_x0 + roi_w, roi_y0 + roi_h), (0, 0, 255), thickness=4)
                    
                except ransac_ellipse.CoverageTooLow as e:
                    if self.debug: print 'Ellipse Coverage Too Low : %s' % e.msg
                    cv2.rectangle(full_frame, (roi_x0, roi_y0), (roi_x0 + roi_w, roi_y0 + roi_h), (0, 0, 255), thickness=4)
                    
                finally:
                    
                    # Extract only eye_roi block after other drawing methods
                    if sub_img_cx0 is not None: 
                        eye_img = full_frame[sub_img_cy0 - 60:sub_img_cy0 + 60,
                                             sub_img_cx0 - 60:sub_img_cx0 + 60]
                    else:
                        eye_img = full_frame[eye_roi.roi_y0:(eye_roi.roi_y0 + eye_roi.roi_h),
                                             eye_roi.roi_x0:(eye_roi.roi_x0 + eye_roi.roi_w)]
                    
                    # Transfer eye_img block to section of half_frame
                    half_frame[half_frame.shape[0] - eye_img.shape[0]:half_frame.shape[0],
                               (half_frame.shape[1] - eye_img.shape[1]) * i: half_frame.shape[1] if i else eye_img.shape[1]] = eye_img
                               
        except eye_extractor.NoEyesFound as e:
            if self.debug: print 'No Eyes Found: %s' % e.msg
            
        # Remove any extreme outliers
        limbuses = limbus_outlier_removal.remove_outliers(limbuses)
        
        # Get gaze points
        for i, limbus in enumerate(limbuses):
            if limbus is None: continue
            gaze_pts_mm[i] = gaze_geometry.get_gaze_point_mm(limbus)
            gaze_pts_px[i] = gaze_geometry.convert_gaze_pt_mm_to_px(gaze_pts_mm[i], self.device)
        
        smoothed_gaze_pt_mm = self.smoother.smooth_gaze(gaze_pts_mm)
        smoothed_gaze_pt_px = gaze_geometry.convert_gaze_pt_mm_to_px(smoothed_gaze_pt_mm, self.device)
        
        # Visualize in 2D and 3D
        cv2.imshow('gaze system', half_frame)
        self.visualizer3d.update_vis(limbuses, smoothed_gaze_pt_mm)
        
        # If recording, take a screenshot of vpython and add to vid. capture
        if self.recording:
            vis_screen = self.visualizer3d.take_screenshot()
            stacked_imgs = image_utils.stack_imgs_horizontal([vis_screen, half_frame])
            self.vid_writer.write(stacked_imgs)
            
        return smoothed_gaze_pt_px
    

#----------------------------------------
# EXAMPLE USAGE 
#----------------------------------------

