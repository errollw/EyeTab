from __future__ import  division

import numpy as np
import gaze_geometry
import image_utils
import anatomical_constants

from PIL import ImageGrab
from visual import *
import marker_manager

# VPYTHON : GAZETRACKER
# x, y, z : x, -y, z
cvt_pt = lambda x, y, z : (x, -y, z)
cvt_pts = lambda pts : [(x, -y, z) for (x, y, z) in pts]
cvt_col = lambda (b, g, r) : (r / 255.0, g / 255.0, b / 255.0)

screen_w_mm, screen_h_mm = 94, 151

gaze_pt_trail_len = 10
smoothed_pt_trail_len = 20
smoothed_normal_grad_len = 10

limbus_r_mm = anatomical_constants.limbus_r_mm
eye_r_mm = anatomical_constants.eye_r_mm

ambient_default = 1

#                range (distance from camera to center)
#                |     center (where the camera looks)
#                |     |                  forward (direction camera points)
#                |     |                  |
scene_params = [(70, (100 , 100 , 180), (-0.8, -0.2, -0.6)),
                (150, (0   , 075 , 100), (0.0 , -0.1, -0.6)),
                (100, (-100, 100 , 180), (0.8 , -0.2, -0.6)),
                (200, (0   , 0   , 000), (0.0 , 0.0 , 1.0))]

#                eye1 - user's right (cyan)
#                |              eye2 - user's left (magenta)
#                |              |              smoothed gaze point (yellow)
#                |              |              |              problem (red)
#                |              |              |              |
debug_colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 255)]

win_os_margin_small, win_os_margin_large = 8, 32  # Margins around window drawn by OS (Win8)

class Visualizer3d:
    
    def __init__(self, device, win_size=(800, 600), recording=False, filename=None):
        
        self.win_w = win_size[0] + win_os_margin_small * 2 
        self.win_h = win_size[1] + win_os_margin_large + win_os_margin_small
        self.active_marker_ind = 0
        self.active_marker_vpy_id = 0
        self.marker_switch_cooldown = 0
        self.vis_mode = 1
        
        if filename is not None:
            self.eval_file = open(filename + '.csv', 'a')
        else:
            self.eval_file = None
        
        self.initialize_scene()
        self.draw_axes()
        self.draw_device_screen(device)
        self.initialize_vis_objs()

    def initialize_scene(self):
        
        # Delete all objects in scene for re-initialization
        for obj in scene.objects:
            obj.visible = False
            del obj
        
        # Only init if it looks like the scene hasn't been init already
        if scene.width == self.win_w:
            return
        
        scene.width, scene.height = self.win_w, self.win_h
        scene.lights, scene.ambient = [], ambient_default
        scene.autoscale = False
        scene.range, scene.center, scene.forward = scene_params[0]
    
    
    def draw_axes(self):
        
        curve(pos=cvt_pts([(0, 0, 0), (10, 0, 0), (8, -1, 0), (10, 0, 0), (8, 1, 0)]))
        curve(pos=cvt_pts([(0, 0, 0), (0, 10, 0), (-1, 8, 0), (0, 10, 0), (1, 8, 0)]))
        curve(pos=cvt_pts([(0, 0, 0), (0, 0, 10), (0, -1, 8), (0, 0, 10), (0, 1, 8)]))
        label(pos=cvt_pt(14, 0, 0), text='x', opacity=0, box=False, line=False)
        label(pos=cvt_pt(0, 14, 0), text='y', opacity=0, box=False, line=False)
        label(pos=cvt_pt(0, 0, 14), text='z', opacity=0, box=False, line=False)
        
        
    def draw_device_screen(self, device):
        
        # Draw and label device screen
        screen = box(pos=(0, screen_h_mm / 2 + 16, 0),
                     size=(screen_w_mm, screen_h_mm, 0),
                     color=(0.1, 0.1, 0.1))
        label(pos=(screen_w_mm / 2, screen.pos.y + screen_h_mm / 4, screen.pos.z),
              text='screen', xoffset=32, yoffset=32, linecolor=(0.5, 0.5, 0.5))
        
        # Draw device body: hardcoded to Nexus 7
        nexus_rect = shapes.rectangle(width=110, height=200, roundness=0.1)
        extrusion(pos=[(0, 180 / 2, -2), (0, 180 / 2, -3)], shape=nexus_rect, color=(0.2, 0.2, 0.2))
        
        # Draw & label camera
        cylinder(pos=(0, 0, -1), axis=(0, 0, -1), color=(0.1, 0.1, 0.1), radius=4)
        label(pos=(0, 0, -1), text='camera', xoffset=32, yoffset= -64, linecolor=(0.5, 0.5, 0.5))
        
        # Draw 4x3 gaze markers
        self.markers = []
        for i in range(4):
            for j in range(3):
                self.markers.append(cylinder(pos=(j * 31 - 31, i * 35 + 40, 0), axis=(0, 0, 1), color=(1, 1, 1), radius=1))
        
    def activate_marker(self, marker_ind):
        
        self.active_marker_ind = marker_ind
        
        # Translates marker index (order looked at) into order created by vpython
        vis_marker_id = marker_manager.get_marker_id_for_vis(marker_ind)
        
        if self.active_marker_vpy_id == vis_marker_id:return
        self.active_marker_vpy_id = vis_marker_id;
        
        # reset all markers
        for marker in self.markers:
            marker.color = (1, 1, 1)
            marker.radius = 1
            
        # make active marker more visible
        self.markers[self.active_marker_vpy_id].color = (1, 0.3, 0.3)
        self.markers[self.active_marker_vpy_id].radius = 3
        
        # register for logging cool-down over a few frames
        self.marker_switch_cooldown = 4
    
    def initialize_vis_objs(self):

        self.limb_circles = []
        self.limb_normals = []
        self.eye_spheres = []
        self.raw_gaze_pts = []
        self.raw_gaze_pts_trail_colors = []
        # self.limb_labels = []
        
        for i in range(2):
            (r, g, b) = cvt_col(debug_colors[i])
            
            # Circle & normal for each eye
            self.limb_circles.append(cylinder(color=(r, g, b), radius=limbus_r_mm))
            self.limb_normals.append(curve(color=(r, g, b)))
            
            # Gaze point and trail for each eye
            self.raw_gaze_pts.append(box(make_trail=True, retain=gaze_pt_trail_len, color=(r, g, b)))
            trail_colors_weights = [x / gaze_pt_trail_len for x in range(1, gaze_pt_trail_len)]
            trail_colors = map(lambda w : (r * w, g * w, b * w), trail_colors_weights)
            self.raw_gaze_pts_trail_colors.append(trail_colors)
            
            # Label for each eye
            # self.limb_labels.append(label(yoffset= -64, linecolor=(0.5, 0.5, 0.5)))

            # Include eyes in vis
            self.eye_spheres.append(sphere(radius=eye_r_mm))
            self.eye_spheres[i].opacity = 0.3

        # Prepare smoothed gaze marker
        (r, g, b) = cvt_col(debug_colors[2])
        self.smoothed_gaze_pt = box(make_trail=True, retain=smoothed_pt_trail_len, color=(r, g, b))
        trail_colors_weights = [x / smoothed_pt_trail_len for x in range(1, smoothed_pt_trail_len)]
        trail_colors = map(lambda w : (r * w, g * w, b * w), trail_colors_weights)
        self.smoothed_gaze_pt_trail_colors = trail_colors
        self.smoothed_gaze_vector = curve(color=(r, g, b), radius=0.8)

    def handle_keyboard(self):
        
        if scene.kb.keys:
            key = scene.kb.getkey()
            
            # Cycle through viewing angles, hack for ignoring non-numerals to avoid ValueException
            if key in '1234567890' and (0 < int(key) <= len(scene_params)):
                scene.range, scene.center, scene.forward = scene_params[int(key) - 1]
                
            # Change visibility mode
            if key is 'q': self.vis_mode = 0
            if key is 'w': self.vis_mode = 1
            if key is 'e': self.vis_mode = 2


    def take_screenshot(self):
        pil_img = ImageGrab.grab((win_os_margin_small,
                                  win_os_margin_large,
                                  self.win_w - win_os_margin_small,
                                  self.win_h - win_os_margin_small))
        return image_utils.pil_to_cv2(pil_img)
        
          
    def update_vis(self, limbuses, smoothed_gaze_pt_mm=None):
        
        self.handle_keyboard()
        
        for i, limbus in enumerate(limbuses):
        
            # Change colors, continue to next eye if limbus is none (not found)
            limb_color = cvt_col(debug_colors[3 if limbus is None else i])
            self.limb_circles[i].color = limb_color
            self.limb_normals[i].color = limb_color
            if limbus is None: continue
        
            # Show 3D limbus circle
            self.limb_circles[i].pos = cvt_pt(*limbus.center_mm)
            self.limb_circles[i].axis = cvt_pt(*limbus.normal)
            self.limb_circles[i].length = 0.1
            
            # Move eye-sphere
            self.eye_spheres[i].pos = self.limb_circles[i].pos - self.limb_circles[i].axis * 100
            
            # Show intersection with normal on screen plane
            gaze_pt_x_mm, gaze_pt_y_mm = gaze_geometry.get_gaze_point_mm(limbus)
            
            # Either show long or short limb normal
            self.limb_normals[i].pos = [cvt_pt(*limbus.center_mm),
                                        cvt_pt(gaze_pt_x_mm, gaze_pt_y_mm, 0) if self.vis_mode == 0 else
                                        (self.limb_circles[i].pos + self.limb_circles[i].axis * 200)]
            
            # Show raw gaze points and trail
            self.raw_gaze_pts[i].pos = cvt_pt(gaze_pt_x_mm, gaze_pt_y_mm, 0.01)
            self.raw_gaze_pts[i].trail_object.color = self.raw_gaze_pts_trail_colors[i]
            
            # Label eye position
            # self.limb_labels[i].pos = self.limb_circles[i].pos
            # self.limb_labels[i].text = '%0.2d, %0.2d, %0.2d' % limbus.center_mm

        # Handle smoothed gaze pt and save evaluation data
        if smoothed_gaze_pt_mm is not None:
            sm_gaze_pt_x, sm_gaze_pt_y = smoothed_gaze_pt_mm
            self.smoothed_gaze_pt.pos = cvt_pt(sm_gaze_pt_x, sm_gaze_pt_y, 0)
            self.smoothed_gaze_pt.trail_object.color = self.smoothed_gaze_pt_trail_colors
            
            limb_mid_pt = (self.limb_circles[0].pos + self.limb_circles[1].pos) / 2
            end_pt = self.smoothed_gaze_pt.pos if self.vis_mode == 1 else limb_mid_pt
            smoothed_vector_pts = [(1 - a) * limb_mid_pt + a * end_pt for a in np.arange(0, 1.05, 0.05)]
            self.smoothed_gaze_vector.pos = smoothed_vector_pts
            self.smoothed_gaze_vector.color = self.smoothed_gaze_pt_trail_colors
            
            # ---------------------------------
            # SAVE EVALUATION DATA
            # ---------------------------------
            if self.eval_file is not None:
                
                # Only record gaze-data after short cooldown
                if self.marker_switch_cooldown <= 0:
                
                    correct_pos = self.markers[self.active_marker_vpy_id].pos
                    gaze_pos = self.smoothed_gaze_pt.pos
                    dist_offset = mag(correct_pos - gaze_pos)
                    x_offset = abs(correct_pos.x - gaze_pos.x)
                    correct_vec = correct_pos - limb_mid_pt
                    gaze_vec = gaze_pos - limb_mid_pt
                    angle_offset = math.degrees(correct_vec.diff_angle(gaze_vec))
                    
                    # marker x,y, gaze x,y, dist to eye-pair, dist_err, dist_err_x, angle_err
                    line_to_write = ','.join([str(x) for x in
                                              [self.active_marker_ind,
                                               ("%.4f" % correct_pos.x), ("%.4f" % correct_pos.y),
                                               ("%.4f" % gaze_pos.x), ("%.4f" % gaze_pos.y),
                                               ("%.4f" % mag(limb_mid_pt)),
                                               ("%.4f" % dist_offset), ("%.4f" % x_offset), ("%.4f" % angle_offset)]])
                    self.eval_file.write(line_to_write + '\n')
                    
                else:
                    self.marker_switch_cooldown -= 1
            
