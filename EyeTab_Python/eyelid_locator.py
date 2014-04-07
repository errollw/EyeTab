import cv2
import numpy as np

from ransac_eyelids import ransac_line, ransac_parabola
from image_utils import stack_imgs_horizontal, stack_imgs_vertical
from draw_utils import draw_points
from time import time
from math import pi

__gabor_params = {'ksize':(5, 5),
                  'sigma':3,
                  'theta':pi / 4,
                  'lambd':pi * 2,
                  'gamma':2,
                  'psi':pi / 2,
                  'ktype':cv2.CV_32F}
__gabor_kern_diag = cv2.getGaborKernel(**__gabor_params)

__winname = "Eyelid Detection"
__debug_imgs_upper = {}
__debug_imgs_lower = {}

__min_thresh = 50


def filter_limbus_pts(u_eyelid, l_eyelid, pts):
    
    # --- Filters to return only pts between eyelids ---
    #    [[ x1   y1]
    #     [ x2   y2]
    #         ...
    #     [ xn   yn]]
    
    pts_filtered = pts                              # Default to returning all pts

    # Top eyelid is parabola   (a(x^2) + b(x) + c)
    if u_eyelid is not None:
        
        a, b, c = u_eyelid                          
        pts_x, pts_y = np.split(pts, 2, axis=1)
        y_lid_pts = (a * pts_x ** 2) + (b * pts_x) + c
        
        ok_pts_inds = np.squeeze([pts_y > y_lid_pts])
        if ok_pts_inds.size > 0:
            pts_filtered = pts_filtered[ok_pts_inds]
        
    # Bottom eyelid is line    (a(x) + b)
    if l_eyelid is not None:
        
        a, b = l_eyelid                             
        pts_x, pts_y = np.split(pts_filtered, 2, axis=1)
        y_lid_pts = (a * pts_x) + b
        
        ok_pts_inds = np.squeeze([pts_y < y_lid_pts])
        if ok_pts_inds.size > 0:
            pts_filtered = pts_filtered[ok_pts_inds]
    
    return pts_filtered


def find_eyelids(eye_img, debug_index):
    
    u_eyelid = find_upper_eyelid(eye_img, debug_index)
    l_eyelid = find_lower_eyelid(eye_img, debug_index)

    if debug_index == 2:
        debug_img_1 = stack_imgs_horizontal([__debug_imgs_upper[1], __debug_imgs_lower[1]])
        debug_img_2 = stack_imgs_horizontal([__debug_imgs_upper[2], __debug_imgs_lower[2]])
        full_debug_img = stack_imgs_vertical([debug_img_1, debug_img_2])
        cv2.imshow(__winname, full_debug_img)

    return u_eyelid, l_eyelid


# --------------------- Upper Eyelid Specific ---------------------

__u_win_rats_w_l = [0.0, 0.3, 0.7]              # Margins around ROI windows
__u_win_rats_w_r = [0.7, 0.3, 0.0]
__u_win_rats_h = [0.3, 0.6, 0.1]

__y_pos_weight = 1                              # Try and weight sclera-boundary higher than eyelid crease
__crease_offset = __gabor_kern_diag.shape[0]

__parabola_y_offset = -10                       # Amount to shift eye-lid by after detection

__min_num_pts_u = 10

__gabor_params = {'ksize':(7, 7),
                  'sigma':3,
                  'theta':-pi / 2,
                  'lambd':pi * 3,
                  'gamma':2,
                  'psi':pi / 2,
                  'ktype':cv2.CV_32F}
__gabor_kern_horiz = cv2.getGaborKernel(**__gabor_params)

def find_upper_eyelid(eye_img, debug_index):

    u_2_win_rats_w = [0.0, 1.0, 0.0]              # Margins around ROI windows
    u_2_win_rats_h = [0.0, 0.5, 0.5]

    # FIXME - using r channel?
    img_blue = cv2.split(eye_img)[2]
    img_w, img_h = eye_img.shape[:2]
    
    # Indexes to extract window sub-images
    w_y1, w_y2 = int(img_h * u_2_win_rats_h[0]), int(img_h * sum(u_2_win_rats_h[:2]))
    w_x1, w_x2 = int(img_w * u_2_win_rats_w[0]), int(img_w * sum(u_2_win_rats_w[:2]))

    # Split image into two halves
    window_img = img_blue[w_y1:w_y2, w_x1:w_x2]
    
    # Supress eyelashes
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    window_img = cv2.morphologyEx(window_img, cv2.MORPH_CLOSE, morph_kernel)
 
    # Filter right half with inverse kernel of left half to ignore iris/sclera boundary    
    filter_img_win = cv2.filter2D(window_img, -1, __gabor_kern_horiz)
    
    # Copy windows back into correct places in full filter image
    filter_img = np.zeros(eye_img.shape[:2], dtype=np.uint8)
    filter_img[w_y1:w_y2, w_x1:w_x2] = filter_img_win
    
    # Mask with circles
    cv2.circle(filter_img, (3*filter_img.shape[1]/7, filter_img.shape[0]/2), filter_img.shape[1]/4, 0,-1)
    cv2.circle(filter_img, (4*filter_img.shape[1]/7, filter_img.shape[0]/2), filter_img.shape[1]/4, 0,-1)
    
    ys = np.argmax(filter_img, axis=0)
    xs = np.arange(filter_img.shape[1])[ys > 0]
    ys = (ys)[ys > 0]

    u_lid_pts = []
    
    for i, x in enumerate(xs):
        col = filter_img.T[x]
        start_ind, end_ind = ys[i] + 5, min(ys[i] + 100, len(col) - 2)
        col_window = col[start_ind:end_ind]
        max_col = np.max(col)
        max_win = np.max(col_window)
        if max_col - max_win < 50 :
            new_y = np.argmax(col_window) + ys[i] + 5
            u_lid_pts.append((x, new_y))
        else:u_lid_pts.append((x, ys[i]))
    
    # Only RANSAC fit eyelid if there are enough points
    if len(u_lid_pts) < __min_num_pts_u * 2:
        eyelid_upper_parabola = None
        u_lid_pts = []
    else:
        u_lid_pts_l = [(x,y) for (x,y) in u_lid_pts if x < filter_img.shape[1]/2]
        u_lid_pts_r = [(x,y) for (x,y) in u_lid_pts if x > filter_img.shape[1]/2]
        
        # Fit eye_img coord points of sclera-segs to degree 2 polynomial
        # a(x^2) + b(x) + c
        eyelid_upper_parabola = ransac_parabola(u_lid_pts_l, u_lid_pts_r,
                                                ransac_iters_max=5,
                                                refine_iters_max=2,
                                                max_err=4)
    if eyelid_upper_parabola is not None:
        a, b, c = eyelid_upper_parabola
        c = c - __parabola_y_offset
        eyelid_upper_parabola = a, b, c

    # --------------------- Debug Drawing ---------------------
    if debug_index:
        debug_img = eye_img.copy()
        
        if eyelid_upper_parabola is not None:
            lid_xs = np.arange(21) * img_w / 20
            lid_ys = a * lid_xs ** 2 + b * lid_xs + c
            lid_pts = np.dstack([lid_xs, lid_ys]).astype(int)
            cv2.polylines(debug_img, lid_pts, False, (0, 255, 0), 1)
        
        draw_points(debug_img, u_lid_pts, (0, 0, 255), 1,2)
        filter_img = cv2.cvtColor(filter_img, cv2.COLOR_GRAY2BGR)
        draw_points(filter_img, u_lid_pts, (0, 0, 255), 1,2)
        
        stacked_windows = stack_imgs_vertical([window_img, filter_img])
        stacked_imgs = stack_imgs_horizontal([stacked_windows, debug_img])
        __debug_imgs_upper[debug_index] = stacked_imgs
    
        if debug_index > 2:
            cv2.imshow(__winname + repr(debug_index) + "u", stacked_imgs);
    # --------------------- Debug Drawing ---------------------

    return eyelid_upper_parabola


# --------------------- Lower Eyelid Specific ---------------------

__l_win_rats_w_l = [0.2, 0.3, 0.5]              # Margins around ROI windows
__l_win_rats_w_r = [0.5, 0.3, 0.2]
__l_win_rats_h = [0.6, 0.4, 0.0]

__min_num_pts_l = 30

__min_thresh = 80

def find_lower_eyelid(eye_img, debug_index):

    line_y_offset = 0                           # Amount to shift eye-lid by after detection

    img_blue = cv2.split(eye_img)[2]
    img_w, img_h = eye_img.shape[:2]

    # Indexes to extract window sub-images
    w_y1, w_y2 = int(img_h * __l_win_rats_h[0]), int(img_h * sum(__l_win_rats_h[:2]))
    wl_x1, wl_x2 = int(img_w * __l_win_rats_w_l[0]), int(img_w * sum(__l_win_rats_w_l[:2]))
    wr_x1, wr_x2 = int(img_w * __l_win_rats_w_r[0]), int(img_w * sum(__l_win_rats_w_r[:2]))

    # Split image into two halves
    window_img_l = img_blue[w_y1:w_y2, wl_x1:wl_x2]
    window_img_r = img_blue[w_y1:w_y2, wr_x1:wr_x2]
    window_img_l = cv2.GaussianBlur(window_img_l, (5, 5), 20)
    window_img_r = cv2.GaussianBlur(window_img_r, (5, 5), 20)

    filter_img_l = cv2.filter2D(window_img_l, -1, __gabor_kern_diag)
    filter_img_r = cv2.filter2D(window_img_r, -1, cv2.flip(__gabor_kern_diag, 1))
    filter_img = np.concatenate([filter_img_l, filter_img_r], axis=1)

    # In polar image, x <-> theta, y <-> magnitude         
    max_vals = np.max(filter_img, axis=0)
    ys = np.argmax(filter_img, axis=0)                      # Take highest filter response as limbus points
    xs = (np.arange(filter_img.shape[1]) + wl_x1)[max_vals > __min_thresh]
    ys = (ys + w_y1)[max_vals > __min_thresh]
    
    l_lid_pts = np.squeeze(np.dstack([xs, ys]), axis=0)
    
    # Only RANSAC fit eyelid if there are enough points
    if l_lid_pts.size < __min_num_pts_u * 2:
        eyelid_lower_line = None
    else:
        eyelid_lower_line = ransac_line(l_lid_pts)
    
    if eyelid_lower_line is not None:
        a, b = eyelid_lower_line
        b = b + line_y_offset
        eyelid_lower_line = a, b
    
    if debug_index:
        debug_img = eye_img.copy()
        
        filter_img = cv2.cvtColor(filter_img, cv2.COLOR_GRAY2BGR)
        
        if l_lid_pts.size > 2:
            draw_points(debug_img, l_lid_pts, (0, 0, 255), 1, 2)
        
        if eyelid_lower_line is not None:
            cv2.line(debug_img, (0, int(b)), (img_w, int(a * img_w + b)), (0, 255, 0))
        
        window_img = np.concatenate([window_img_l, window_img_r], axis=1)
        stacked_windows = stack_imgs_vertical([window_img, filter_img])
        stacked_imgs = stack_imgs_horizontal([stacked_windows, debug_img])
        __debug_imgs_lower[debug_index] = stacked_imgs
    
        if debug_index > 2:
            cv2.imshow(__winname + repr(debug_index) + "l", stacked_imgs);
     
    return eyelid_lower_line