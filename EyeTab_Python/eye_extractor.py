import cv2, os, numpy as np
import image_utils
import time_profiler
import device_constants

#                        eye_1     eye_2
#                  skin  |    nose |    skin
#                  |     |    |    |    |
eye_part_ratios = [0.05, 0.3, 0.3, 0.3, 0.05]

# Will try to find eye-pairs after rotating by these angles
angles_to_try = [0, 15, -15, 30, -30]

winname = 'Eye Extractor'

classifier_pair = cv2.CascadeClassifier(os.path.join('cascades', 'haarcascade_mcs_eyepair_big.xml'))
classifier_l_eye = cv2.CascadeClassifier(os.path.join('cascades', 'haarcascade_mcs_lefteye.xml'))
classifier_r_eye = cv2.CascadeClassifier(os.path.join('cascades', 'haarcascade_mcs_righteye.xml'))

class NoEyesFound(Exception):
    def __init__(self, msg):
        self.msg = msg

class EyeRoi:
    
    def __init__(self, (roi_x0_param, roi_y0_param), img_param):
        
        self.roi_x0, self.roi_y0 = int(roi_x0_param), int(roi_y0_param)
        self.roi_h, self.roi_w = img_param.shape[:2]
        self.img = img_param
        
        if self.img is None: raise NoEyesFound()    # Prevent future problems with NoneType image

    def refine_pupil(self, (pupil_x0, pupil_y0), full_img):
        
        # Copy pre-processed ROI back into full-frame (specularities removed)
        full_img[self.roi_y0:self.roi_y0 + self.roi_h,
                 self.roi_x0:self.roi_x0 + self.roi_w] = self.img
        
        img_size = min(self.roi_w, self.roi_h)
        self.roi_h, self.roi_w = img_size, img_size
        self.roi_x0 = (self.roi_x0 + pupil_x0) - self.roi_w / 2
        self.roi_y0 = (self.roi_y0 + pupil_y0) - self.roi_h / 2
        
        # Ensure ROI will still lie within image boundaries (x-axis only)
        self.roi_x0 = max(self.roi_x0, 0)
        self.roi_x0 = min(self.roi_x0, full_img.shape[1] - self.roi_w)
        
        self.img = full_img[self.roi_y0:self.roi_y0 + self.roi_h,
                            self.roi_x0:self.roi_x0 + self.roi_w]
        
        if self.img is None: raise NoEyesFound()    # Prevent future problems with NoneType image
        

def choose_best_eye_pair(eye_pair_rects, img):
    
    """ Determine between eyes / eyebrows / nostrils by comparing frequencies in image
    """
    
    return max(eye_pair_rects,
               key=lambda (x0, y0, w, h)
               : image_utils.measure_blurriness_LoG(img[y0:y0 + h, x0:x0 + w]))


# Default behaviour if fail to get eyepair
def get_eye_rois_default(frame_pyr, down_scale=4, debug=False, device=None):

    """ Returns a pair of EyeRois - one for each eye in an eye pair
    """

    pyr_img = frame_pyr[down_scale].copy()
    full_frame = frame_pyr[1]
    
    pyr_img_grey = cv2.cvtColor(pyr_img, cv2.COLOR_BGR2GRAY)
    
    _, img_w = pyr_img_grey.shape[:2]
    roi_r,roi_l = pyr_img_grey.copy(), pyr_img_grey.copy()
    roi_r[:, 0:img_w/2] = 0
    roi_l[:, img_w / 2:img_w] = 0
    min_eye_size = (pyr_img.shape[0] / 6, pyr_img.shape[0] / 9)
    
    eye_l_rects = classifier_l_eye.detectMultiScale(roi_l, scaleFactor=1.1, minSize=min_eye_size)
    eye_r_rects = classifier_r_eye.detectMultiScale(roi_r, scaleFactor=1.1, minSize=min_eye_size)
    
    if len(eye_l_rects) == 0 or len(eye_r_rects) == 0 :
        raise NoEyesFound('Did not find eyes with default behaviour')
    
    if len(eye_l_rects) == 1:
        best_eye_l_rect = eye_l_rects[0]
    else:
        best_eye_l_rect = choose_best_eye_pair(eye_l_rects, pyr_img)
    rect_l_x0, rect_l_y0, rect_l_w, rect_l_h = best_eye_l_rect
    roi_l_x0, roi_l_y0, roi_l_w, roi_l_h = [x * down_scale for x in [rect_l_x0, rect_l_y0, rect_l_w, rect_l_h]]
        
    if len(eye_r_rects) == 1:
        best_eye_r_rect = eye_r_rects[0]
    else:
        best_eye_r_rect = choose_best_eye_pair(eye_r_rects, pyr_img)
    rect_r_x0, rect_r_y0, rect_r_w, rect_r_h = best_eye_r_rect
    roi_r_x0, roi_r_y0, roi_r_w, roi_r_h = [x * down_scale for x in [rect_r_x0, rect_r_y0, rect_r_w, rect_r_h]]
    
    eye_1_img = full_frame[roi_l_y0:(roi_l_y0 + roi_l_h), roi_l_x0:roi_l_x0 + roi_l_w]
    eye_2_img = full_frame[roi_r_y0:(roi_r_y0 + roi_r_h), roi_r_x0:roi_r_x0 + roi_r_w]
    eye_roi_1, eye_roi_2 = EyeRoi((roi_l_x0, roi_l_y0), eye_1_img), EyeRoi((roi_r_x0, roi_r_y0), eye_2_img)
    
    # Draw box around each eye_roi
    if debug:
        debug_img = frame_pyr[down_scale]
        cv2.rectangle(debug_img,
              (eye_roi_1.roi_x0 / down_scale, eye_roi_1.roi_y0 / down_scale),
              ((eye_roi_1.roi_x0 + eye_roi_1.roi_w) / down_scale, (eye_roi_1.roi_y0 + eye_roi_1.roi_h) / down_scale),
              (0, 0, 255))
        cv2.rectangle(debug_img,
              (eye_roi_2.roi_x0 / down_scale, eye_roi_2.roi_y0 / down_scale),
              ((eye_roi_2.roi_x0 + eye_roi_2.roi_w) / down_scale, (eye_roi_2.roi_y0 + eye_roi_2.roi_h) / down_scale),
              (0, 0, 255))
        
        cv2.imshow(winname, debug_img)
    
    return eye_roi_1, eye_roi_2


def get_eye_rois_at_angle(frame_pyr, angle, down_scale=4, debug=False, device=None):

    """ Returns a pair of EyeRois - one for each eye in an eye pair
    """

    pyr_img = frame_pyr[down_scale].copy()
    full_frame = frame_pyr[1]
    
    # Rotate down-scaled frame for potential non-horizontal eye-pairs
    if angle != 0:
        rot_center = (pyr_img.shape[1] / 2, pyr_img.shape[0] / 2)
        rot_mat_fwd = cv2.getRotationMatrix2D(rot_center, angle, 1)
        pyr_img = cv2.warpAffine(pyr_img, rot_mat_fwd, (pyr_img.shape[1], pyr_img.shape[0]))
    
    if device is None or device.device_type == device_constants.WEBCAM:
        min_eye_pair_size = (0, 0)
    elif device.device_type == device_constants.NEXUS_7_INV:
        min_eye_pair_size = (pyr_img.shape[0] / 4, pyr_img.shape[0] / 16)
        
    pyr_img_grey = cv2.cvtColor(pyr_img, cv2.COLOR_BGR2GRAY)
        
    eye_pair_rects = classifier_pair.detectMultiScale(pyr_img_grey,
                                                      scaleFactor=1.1,
                                                      minSize=min_eye_pair_size)
    
    if len(eye_pair_rects) == 0 :
        raise NoEyesFound('Haar classifier found no eye-pairs at angle %d' % angle)
    
    if len(eye_pair_rects) == 1:
        best_eye_pair_rect = eye_pair_rects[0]
    else:
        best_eye_pair_rect = choose_best_eye_pair(eye_pair_rects, pyr_img)
    
    # Expand the ROI by constant pyr_scale
    rect_x0, rect_y0, rect_w, rect_h = best_eye_pair_rect
    roi_x0, roi_y0, roi_w, roi_h = [x * down_scale for x in [rect_x0, rect_y0, rect_w, rect_h]]
        
    # Get origin and sizes of eye-pairs found in image (possibly rotated)
    eye_1_x0 = int(roi_x0 + roi_w * eye_part_ratios[0])
    eye_1_w = int(roi_w * eye_part_ratios[1])
    eye_2_x0 = int(roi_x0 + roi_w * sum(eye_part_ratios[:3]))
    eye_2_w = int(roi_w * eye_part_ratios[3])

    if angle == 0:
        eye_1_img = full_frame[roi_y0:(roi_y0 + roi_h), eye_1_x0:eye_1_x0 + eye_1_w]
        eye_2_img = full_frame[roi_y0:(roi_y0 + roi_h), eye_2_x0:eye_2_x0 + eye_2_w]
        eye_roi_1, eye_roi_2 = EyeRoi((eye_1_x0, roi_y0), eye_1_img), EyeRoi((eye_2_x0, roi_y0), eye_2_img)

    else:
        
        # Find currently rotated middles of eye ROIs
        eye_1_mid_rot = [[eye_1_x0 + eye_1_w / 2], [roi_y0 + roi_h / 2], [down_scale]]
        eye_2_mid_rot = [[eye_2_x0 + eye_2_w / 2], [roi_y0 + roi_h / 2], [down_scale]]
        
        # Un-rotate them to match original frame
        rot_mat_inv = cv2.getRotationMatrix2D(rot_center, -angle, 1)
        eye_1_mid_x, eye_1_mid_y = np.ravel(np.dot(rot_mat_inv, eye_1_mid_rot))
        eye_2_mid_x, eye_2_mid_y = np.ravel(np.dot(rot_mat_inv, eye_2_mid_rot))
        
        # Get ROIs of un-rotated full-frame    
        eye_1_x0, eye_1_y0 = eye_1_mid_x - eye_1_w / 2, eye_1_mid_y - roi_h / 2
        eye_2_x0, eye_2_y0 = eye_2_mid_x - eye_2_w / 2, eye_2_mid_y - roi_h / 2
        eye_1_img = full_frame[eye_1_y0:(eye_1_y0 + roi_h), eye_1_x0:eye_1_x0 + eye_1_w]
        eye_2_img = full_frame[eye_2_y0:(eye_2_y0 + roi_h), eye_2_x0:eye_2_x0 + eye_2_w]
        
        eye_roi_1, eye_roi_2 = EyeRoi((eye_1_x0, eye_1_y0), eye_1_img), EyeRoi((eye_2_x0, eye_2_y0), eye_2_img)
    
    # Draw box around each eye_roi
    if debug:
        debug_img = frame_pyr[down_scale]
        cv2.rectangle(debug_img,
              (eye_roi_1.roi_x0 / down_scale, eye_roi_1.roi_y0 / down_scale),
              ((eye_roi_1.roi_x0 + eye_roi_1.roi_w) / down_scale, (eye_roi_1.roi_y0 + eye_roi_1.roi_h) / down_scale),
              (0, 0, 255))
        cv2.rectangle(debug_img,
              (eye_roi_2.roi_x0 / down_scale, eye_roi_2.roi_y0 / down_scale),
              ((eye_roi_2.roi_x0 + eye_roi_2.roi_w) / down_scale, (eye_roi_2.roi_y0 + eye_roi_2.roi_h) / down_scale),
              (0, 0, 255))
        
        cv2.imshow(winname, debug_img)
        
    return eye_roi_1, eye_roi_2


def get_eye_rois(frame_pyr, down_scale=4, debug=False, device=None):
    
    for angle in angles_to_try:
        try:
            return get_eye_rois_at_angle(frame_pyr, angle, down_scale, debug, device)
        except NoEyesFound:
            continue        # Try next angle
        
    try:
        return get_eye_rois_default(frame_pyr, down_scale, debug, device)
    except NoEyesFound:
        pass

    # Draw red border around debug frame
    if debug:
        debug_img = frame_pyr[down_scale]
        cv2.rectangle(debug_img, (0, 0), (debug_img.shape[1] - 1, debug_img.shape[0] - 1), (0, 0, 255), thickness=4)
        cv2.imshow(winname, debug_img)

    # No eyes have been found at any angle
    raise NoEyesFound('Haar classifiers found no eye-pairs at angles: %s' % str(angles_to_try))

