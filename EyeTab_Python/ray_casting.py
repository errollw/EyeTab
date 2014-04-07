import cv2, numpy as np

from math import sin, cos, radians
from draw_utils import draw_cross, draw_points
from image_utils import stack_imgs_vertical

winname = 'Ray Casting'

class NoLimbusFound(Exception):
    pass


class RayCaster:

    def __init__(self, kernel=[1, 2, 0, -2, -1], __limb_r_ratios=(0.1, 0.4)):
        self.kernel = kernel
        self.limb_r_ratios = __limb_r_ratios
        self.full_debug_img = None

    def get_intensity(self, (b, g, r)):
        return int(b / 2 + g / 2)
    
    def cast_rays_spread(self, bgr_img, start_pos, angle_mean, spread, limb_r_range, step_angle=2):
        
        (x0, y0) = start_pos
        (min_limb_r, max_limb_r) = limb_r_range
        
        points_found_in_spread = set()
        for tht in range(angle_mean - spread / 2, angle_mean + spread / 2, step_angle):
            points_found_in_spread.add(self.ray_sample(bgr_img, radians(tht), (x0, y0), min_limb_r, max_limb_r))
        
        return points_found_in_spread
    
    
    def ray_sample(self, bgr_img, angle, (x0, y0), ray_start, ray_end, step_ray=1):
        
        x0_off, y0_off = x0 + ray_start * cos(angle), y0 + ray_start * sin(angle)
        dx, dy = step_ray * cos(angle), step_ray * sin(angle)
        len_to_travel = ray_end - ray_start
        
        if bgr_img is None: raise NoLimbusFound
        
        img_h, img_w = bgr_img.shape[:2]
        
        sample_list = []
        
        for i in range(len_to_travel / step_ray):
            x_sample, y_sample = x0_off + dx * i, y0_off + dy * i
            if not(0 < x_sample < img_w) or not(0 < y_sample < img_h): break
            sample_list.append(self.get_intensity(bgr_img[y_sample][x_sample]))
        
        conv = np.convolve(sample_list, self.kernel, 'valid')
        
        best_x = x0_off + dx * (conv.argmax() + len(self.kernel) / 2)
        best_y = y0_off + dy * (conv.argmax() + len(self.kernel) / 2)
        
        return int(best_x), int(best_y)
    
    
    def find_limbus_edge_pts(self, eye_roi, debug=False):
        
        blurred_eye_roi_img = cv2.GaussianBlur(eye_roi.img, (5, 5), 5)

        pupil_x0, pupil_y0 = eye_roi.img.shape[1] / 2, eye_roi.img.shape[0] / 2
    
        min_limb_r = int(eye_roi.img.shape[0] * self.limb_r_ratios[0])
        max_limb_r = int(eye_roi.img.shape[0] * self.limb_r_ratios[1])
        
        pts_found = set()
        pts_found = pts_found.union(self.cast_rays_spread(bgr_img=blurred_eye_roi_img,
                                                          start_pos=(pupil_x0, pupil_y0),
                                                          angle_mean=0, spread=120,
                                                          limb_r_range=(min_limb_r, max_limb_r)))
        pts_found = pts_found.union(self.cast_rays_spread(bgr_img=blurred_eye_roi_img,
                                                          start_pos=(pupil_x0, pupil_y0),
                                                          angle_mean=180, spread=120,
                                                          limb_r_range=(min_limb_r, max_limb_r)))
                    
        if debug:
            debug_img = blurred_eye_roi_img.copy()
            cv2.circle(blurred_eye_roi_img, (eye_roi.img.shape[1] / 2, eye_roi.img.shape[0] / 2), min_limb_r, (255, 255, 0))
            cv2.circle(blurred_eye_roi_img, (eye_roi.img.shape[1] / 2, eye_roi.img.shape[0] / 2), max_limb_r, (255, 255, 0))
            draw_cross(debug_img, (pupil_x0, pupil_y0), color=(255, 255, 0), width=6)
            draw_points(debug_img, pts_found, (0, 0, 255), width=1, thickness=2)
            stacked_imgs = np.concatenate([eye_roi.img, blurred_eye_roi_img, debug_img], axis=1)
            
            if debug == 1:
                self.full_debug_img = stacked_imgs
            elif debug == 2:
                self.full_debug_img = stack_imgs_vertical([self.full_debug_img, stacked_imgs])
                cv2.imshow(winname, self.full_debug_img)
            elif debug == 3:
                cv2.imshow(winname, stacked_imgs);
        
        return pts_found

