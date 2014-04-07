from time import time

import cv2
import numpy as np
import draw_utils
import image_utils

__fast_width = 80.0
__min_rad = __fast_width / 8
__max_rad = __fast_width / 2

__weight_ratio_edge = 1
__weight_ratio_darkness = 0.0

__winname = "Pupil (isophote)"
__debug_imgs = {}

def get_gradients(img):

    """Find image gradients using OpenCV built-ins
    """
    
    f_x = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=9)
    f_y = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=9)
    f_xy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=9)
    f_xx = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=2, dy=0, ksize=9)
    f_yy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=2, ksize=9)
    
    return f_x, f_y, f_xy, f_xx, f_yy


def get_center_map(eye_img):

    f_x, f_y, f_xy, f_xx, f_yy = get_gradients(eye_img)

    # Calculate the curved-ness and weighting function
    curvedness = np.sqrt(f_xx ** 2 + 2 * f_xy ** 2 + f_yy ** 2)
    curvedness_norm = cv2.normalize(curvedness, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("curvedness", curvedness_norm)
    weight_edge = cv2.normalize(curvedness, 0, 255 * __weight_ratio_edge, norm_type=cv2.NORM_MINMAX)               # higher weight to stronger edges
    weight_middle = cv2.normalize((255 - eye_img), 0, 255 * __weight_ratio_darkness, norm_type=cv2.NORM_MINMAX)       # higher center weight to darker areas
    
    # Calculate the displacement vectors
    temp_top = f_x ** 2 + f_y ** 2
    temp_bot = ((f_y ** 2) * f_xx) - (2 * f_x * f_xy * f_y) + ((f_x ** 2) * f_yy) + 0.0001      # hack to offset against division by 0
    d_vec_mul = -temp_top / temp_bot
    d_vec_x = d_vec_mul * f_x
    d_vec_y = d_vec_mul * f_y
    
    # Remove infinite displacements for straight lines
    d_vec_x = np.nan_to_num(d_vec_x)
    d_vec_y = np.nan_to_num(d_vec_y)
    mag_d_vec = cv2.magnitude(d_vec_x, d_vec_y)
    
    # Prevent using weights with bad radius sizes
    weight_edge[mag_d_vec < __min_rad] = 0
    weight_edge[mag_d_vec > __max_rad] = 0
    
    # Calculate curvature to ensure we use gradients which point in right direction
    curvature = temp_bot / (0.0001 + (temp_top ** 1.5))
    weight_edge[curvature < 0] = 0
    weight_edge[curvedness_norm < 20] = 0
    
    # Make indexes into accumulator using basic grid and vector offsets
    grid = np.indices(eye_img.shape[:2], np.uint8)
    acc_inds_x = grid[1] + d_vec_x.astype(int)
    acc_inds_y = grid[0] + d_vec_y.astype(int)
    
    # Prevent indexing outside of accumulator
    acc_inds_x[acc_inds_x < 0] = 0
    acc_inds_y[acc_inds_y < 0] = 0
    acc_inds_x[acc_inds_x >= eye_img.shape[1]] = 0
    acc_inds_y[acc_inds_y >= eye_img.shape[0]] = 0
    
    # Make center map accumulator
    accumulator = np.zeros(eye_img.shape[:2])
    
    # Use numpy fancy indexing to add weights in one go
    # (This is fast as it avoids python loops)
    accumulator[acc_inds_y, acc_inds_x] += weight_edge
    accumulator += weight_middle
    
    # Post-processing
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    accumulator = cv2.morphologyEx(accumulator, cv2.MORPH_DILATE, morph_kernel)
    # accumulator = cv2.blur(accumulator, (10, 10))
    accumulator = cv2.GaussianBlur(accumulator, (13, 13), 0)
    
    return accumulator


def find_pupil(eye_img_bgr, debug_index=False):
    
    eye_img_r = cv2.cvtColor(eye_img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Scale to small image for faster computation
    scale = __fast_width / eye_img_r.shape[1]
    small_size = (int(__fast_width), int((__fast_width / eye_img_r.shape[1]) * eye_img_r.shape[0]))
    eye_img_small = cv2.resize(eye_img_r, small_size)
    eye_img_small = cv2.GaussianBlur(eye_img_small, (3, 3), 0)
    
    center_map = get_center_map(eye_img_small)
    
    max_val_index = np.argmax(center_map)
    pupil_y0, pupil_x0 = max_val_index // center_map.shape[1], max_val_index % center_map.shape[1]
    
    # Scale back to original coordinates
    pupil_y0, pupil_x0 = int((pupil_y0 + 0.5) / scale), int((pupil_x0 + 0.5) / scale)
    
    if debug_index:
        
        eye_img_r_debug = cv2.cvtColor(eye_img_r, cv2.COLOR_GRAY2BGR)
        debug_img = eye_img_bgr.copy()
        
        cmap_norm = cv2.normalize(center_map, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        center_map_big = cv2.resize(cmap_norm, (eye_img_r.shape[1], eye_img_r.shape[0])).astype(np.uint8)
        center_map_big = cv2.cvtColor(center_map_big, cv2.COLOR_GRAY2BGR)
        
        overlay_img = cv2.addWeighted(center_map_big, 0.9, eye_img_r_debug, 0.1, 1)
                
        draw_utils.draw_cross(debug_img, (pupil_x0, pupil_y0), (0, 255, 255), 6)
        draw_utils.draw_cross(overlay_img, (pupil_x0, pupil_y0), (255, 0, 0), 6)
        
        # stacked_small_size = image_utils.stack_imgs_vertical([eye_img_small, cmap_norm])
        stacked_imgs = image_utils.stack_imgs_horizontal([debug_img, overlay_img])
        __debug_imgs[debug_index] = stacked_imgs
        
        if debug_index == 2:
            full_debug_img = image_utils.stack_imgs_vertical([__debug_imgs[1], __debug_imgs[2]]);
            cv2.imshow(__winname, full_debug_img)
        elif debug_index > 2:
            cv2.imshow(__winname, stacked_imgs);

    return pupil_x0, pupil_y0

#----------------------------------------
# EXAMPLE USAGE 
#----------------------------------------
if __name__ == '__main__':
    
    eye_img_bgr = cv2.imread('test_images/eyelid3.png', 3)
    # eye_img = cv2.imread('eye_images/erroll9_l.png', 3)

    tic = time()
    (pupil_x0, pupil_y0) = find_pupil(eye_img_bgr, debug_index=3)
    print 'Time taken to find pupil: %0.3f' % (time() - tic)

    cv2.waitKey()
    
