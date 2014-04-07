import cv2, numpy as np

import eye_center_locator_gradients
import eye_center_locator_isophote
import image_utils
import draw_utils

__winname = "Eye Centre (combined)"
__debug_imgs = {}

w_grads, w_iso = 0.7, 0.3

def find_pupil(eye_img_bgr, fast_width_grads=25.5, fast_width_iso=80, weight_grads=0.9, weight_iso=0.1, debug_index=False):
    
    eye_img_r = cv2.split(eye_img_bgr)[2]

    fast_size_grads = (int((fast_width_grads / eye_img_bgr.shape[0]) * eye_img_bgr.shape[1]), int(fast_width_grads))
    fast_img_grads = cv2.resize(eye_img_r, fast_size_grads)
    
    fast_size_iso = (int(fast_width_iso), int((fast_width_iso / eye_img_r.shape[1]) * eye_img_r.shape[0]))
    fast_img_iso = cv2.resize(eye_img_r, fast_size_iso)
    
    c_map_grads = eye_center_locator_gradients.get_center_map(fast_img_grads)
    c_map_iso = eye_center_locator_isophote.get_center_map(fast_img_iso)
    
    c_map_norm_grads = cv2.normalize(c_map_grads, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    c_map_big_grads = cv2.resize(c_map_norm_grads, (eye_img_bgr.shape[1], eye_img_bgr.shape[0])).astype(np.uint8)
    
    c_map_norm_iso = cv2.normalize(c_map_iso, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    c_map_big_iso = cv2.resize(c_map_norm_iso, (eye_img_bgr.shape[1], eye_img_bgr.shape[0])).astype(np.uint8)
    
    joint_c_map = cv2.addWeighted(c_map_big_grads, w_grads, c_map_big_iso, w_iso, 1.0) 
    
    max_val_index = np.argmax(joint_c_map)
    pupil_y0, pupil_x0 = max_val_index // joint_c_map.shape[1], max_val_index % joint_c_map.shape[1]
    
    max_val_index_2 = np.argmax(c_map_big_grads)
    pupil_y0_2, pupil_x0_2 = max_val_index_2 // joint_c_map.shape[1], max_val_index_2 % joint_c_map.shape[1]
    
    max_val_index_3 = np.argmax(c_map_big_iso)
    pupil_y0_3, pupil_x0_3 = max_val_index_3 // joint_c_map.shape[1], max_val_index_3 % joint_c_map.shape[1]
   
    if debug_index:
        
        debug_img = eye_img_bgr.copy()
        
        joint_c_map = cv2.cvtColor(joint_c_map, cv2.COLOR_GRAY2BGR)
        c_map_big_iso = cv2.cvtColor(c_map_big_iso, cv2.COLOR_GRAY2BGR)
        c_map_big_grads = cv2.cvtColor(c_map_big_grads, cv2.COLOR_GRAY2BGR)
        
        draw_utils.draw_cross(debug_img, (pupil_x0, pupil_y0), (0, 255, 255), 16, 2)
        draw_utils.draw_cross(joint_c_map, (pupil_x0_3, pupil_y0_3), (255, 0, 255), 8, 2)
        draw_utils.draw_cross(joint_c_map, (pupil_x0_2, pupil_y0_2), (255, 0, 255), 8, 2)
        draw_utils.draw_cross(joint_c_map, (pupil_x0, pupil_y0), (255, 0, 0), 16, 2)
        
        draw_utils.draw_cross(c_map_big_iso, (pupil_x0_3, pupil_y0_3), (255, 0, 0), 16, 2)
        draw_utils.draw_cross(c_map_big_grads, (pupil_x0_2, pupil_y0_2), (255, 0, 0), 16, 2)
        
        stacked_imgs = image_utils.stack_imgs_horizontal([debug_img, c_map_big_grads, c_map_big_iso, joint_c_map])
        __debug_imgs[debug_index] = stacked_imgs
        
        if debug_index == 2:
            full_debug_img = image_utils.stack_imgs_vertical([__debug_imgs[1], __debug_imgs[2]]);
            cv2.imshow(__winname, full_debug_img)
        elif debug_index > 2:
            cv2.imshow(__winname, stacked_imgs);
    
    return pupil_x0, pupil_y0

