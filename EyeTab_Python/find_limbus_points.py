import cv2, numpy as np

from math import pi 
from linpolar_transform import linpolar
from image_utils import stack_imgs_horizontal, stack_imgs_vertical
from draw_utils import draw_points

__fixed_width = 400
__limb_r_ratios = (0.15, 0.4)
__min_limb_r = int(__fixed_width * __limb_r_ratios[0])    
__max_limb_r = int(__fixed_width * __limb_r_ratios[1])

__gabor_params = {'ksize':(7, 7),
          'sigma':2, 'theta':pi / 2,
          'lambd':pi * 2,
          'gamma':2,
          'psi':pi / 2,
          'ktype':cv2.CV_32F}
__gabor_kern = cv2.getGaborKernel(**__gabor_params)

__winname = "Limbus Points (filtered polar img)"
__debug_imgs = {}


#                eye_img - bgr img of eye ROI
#                |        phi - angle to ignore at extreme ranges (close to 90 or 270)
#                |        |       angles considered = 360 / angle_step
#                |        |       |
def get_limb_pts(eye_img, phi=20, angle_step=1, debug_index=False):
    
    polar_img_w = 360 / angle_step                                      # Polar image has one column per angle of interest
    phi_range_1 = ((90 - phi) / angle_step, (90 + phi) / angle_step)    # Ranges of angles to be ignored (too close to lids)
    phi_range_2 = ((270 - phi) / angle_step, (270 + phi) / angle_step)
    
    eye_img_grey = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)      # Do BGR-grey
    eye_img_grey = cv2.medianBlur(eye_img_grey, 5)
    
    # Scale to fixed size image for re-using transform matrix
    scale = eye_img.shape[0] / float(__fixed_width)
    img_fixed_size = cv2.resize(eye_img_grey, (__fixed_width, __fixed_width))
    
    # Transform image into polar coords and blur
    img_polar = linpolar(img_fixed_size, trans_w=polar_img_w, trans_h=__fixed_width / 2)
    img_polar = cv2.GaussianBlur(img_polar, (5, 5), 0)
    
    # Take the segment between min & max radii and filter with Gabor kernel
    img_polar_seg = img_polar[__min_limb_r:__max_limb_r, :]
    filter_img = cv2.filter2D(img_polar_seg, -1, __gabor_kern)
    
    # Black out ignored angles
    filter_img.T[ phi_range_1[0] : phi_range_1[1] ] = 0
    filter_img.T[ phi_range_2[0] : phi_range_2[1] ] = 0

    # In polar image, x <-> theta, y <-> magnitude         
    pol_ys = np.argmax(filter_img, axis=0)                      # Take highest filter response as limbus points
    pol_xs = np.arange(filter_img.shape[1])[pol_ys > 0]
    mags = (pol_ys + __min_limb_r)[pol_ys > 0]
    thts = np.radians(pol_xs * angle_step)

    # Translate each point back into fixed img coords
    xs, ys = cv2.polarToCart(mags.astype(float), thts)
    xs = (xs + __fixed_width / 2) * scale                       # Shift and scale cart. coords back to original eye-ROI coords
    ys = (ys + __fixed_width / 2) * scale
    
    # Points returned in form
    #    [[ x1   y1]
    #     [ x2   y2]
    #         ...
    #     [ xn   yn]]
    pts_cart = np.concatenate([xs, ys], axis=1)
    
    # --------------------- Debug Drawing ---------------------
    if debug_index != False:
        debug_img = eye_img.copy()
        debug_polar = cv2.cvtColor(img_polar, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite("polar.jpg",debug_polar)
        
        cv2.line(debug_polar, (0, __min_limb_r), (img_polar.shape[1], __min_limb_r), (255, 255, 0))
        cv2.line(debug_polar, (0, __max_limb_r), (img_polar.shape[1], __max_limb_r), (255, 255, 0))
        cv2.circle(debug_img, (debug_img.shape[1] / 2, debug_img.shape[0] / 2), int(debug_img.shape[0] * __limb_r_ratios[0]), (255, 255, 0))
        cv2.circle(debug_img, (debug_img.shape[1] / 2, debug_img.shape[0] / 2), int(debug_img.shape[0] * __limb_r_ratios[1]), (255, 255, 0))
        
        pts_polar = np.squeeze(np.dstack([pol_xs, mags]))
        draw_points(debug_polar, pts_polar, (0, 0, 255), width=1)
        draw_points(debug_img, pts_cart, (0, 0, 255), width=1)
    
        stacked_imgs_polar = stack_imgs_vertical([debug_polar, filter_img])
        stacked_imgs = stack_imgs_horizontal([debug_img, eye_img_grey, stacked_imgs_polar])
        
        __debug_imgs[debug_index] = stacked_imgs
        
        if debug_index == 2:
            full_debug_img = stack_imgs_vertical([__debug_imgs[1], __debug_imgs[2]]);
            cv2.imshow(__winname, full_debug_img)
        elif debug_index > 2:
            cv2.imshow(__winname, stacked_imgs);
    # --------------------- Debug Drawing ---------------------

    return pts_cart

