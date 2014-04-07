import cv2, numpy as np, random

from math import sin, cos, atan2, radians, degrees
from conic_section import Ellipse, BadEllipseShape
from gaze_geometry import get_gaze_point_px
from draw_utils import draw_cross, draw_points, draw_normal, draw_gaze

write_text = lambda x, y : cv2.putText(x, y, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
winname = 'Ransac ellipse fit'
image_aware_support = True

max_axis_ratio = 3 
min_coverage = 25

prin_point_x = 344.629
prin_point_y = 626.738


class NotEnoughInliers(Exception):
    pass
        
class NoEllipseFound(Exception):
    pass

class CoverageTooLow(Exception):
    def __init__(self, msg):
        self.msg = msg


def fit_ellipse(ellipse_points, (img_h, img_w)):
    
    rotated_rect = cv2.fitEllipse(np.array([ellipse_points], np.float32))
    (x0, y0), (min_axis, maj_axis), _ = rotated_rect
    
    if min_axis * maj_axis == 0: 
        raise BadEllipseShape('Min_axis: %0.2f maj_axis: %0.2f' % (min_axis, maj_axis))
    if maj_axis / min_axis > max_axis_ratio: 
        raise BadEllipseShape('Axis ratio: %0.2f' % (maj_axis / min_axis))
    if not (0 < x0 < img_w and 0 < y0 < img_h):
        raise BadEllipseShape('Center out of bounds: %d,%d' % (x0, y0))
    
    return Ellipse(rotated_rect)


def get_err_scale(ellipse):
    
    (x0, y0), (_, maj_axis), angle = ellipse.rotated_rect
    min_axis_x, min_axis_y = -sin(radians(angle)), cos(radians(angle))
    min_axis_plus_1px = x0 + (maj_axis / 2 + 1) * min_axis_x, y0 + (maj_axis / 2 + 1) * min_axis_y
    err_of_1px = ellipse.distance(min_axis_plus_1px)
    
    if err_of_1px == 0:
        raise BadEllipseShape('Error scale = 0')
        
    return 1 / err_of_1px


def calculate_coverage(ellipse, inliers, step=5):
    
    '''Calculates percentage of ellipse edge covered by its inliers
    '''

    coverage = {}
    for angle in range(-180, 180, step):
        coverage[angle] = 0
    
    (x0, y0) = ellipse.rotated_rect[0]
    for (x, y) in inliers:
        dx, dy = x - x0, y - y0
        angle = degrees(atan2(dy, dx))
        angle_rounded = int(angle / step) * step
        coverage[angle_rounded] = 1

    return sum(coverage.values()) / (360.0 / step) * 100


def ransac_ellipse_fit(points, bgr_img, roi_pos, ransac_iters_max=50, refine_iters_max=3, max_err=2, debug=False):
    
    if points.size == 0: raise NoEllipseFound()
    
    blurred_grey_img = cv2.blur(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY), (3, 3))
    
    image_dx = cv2.Sobel(blurred_grey_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
    image_dy = cv2.Sobel(blurred_grey_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
    
    pts_x, pts_y = np.split(points, 2, axis=1)
    pts_x, pts_y = np.squeeze(pts_x), np.squeeze(pts_y)
    
    if debug:
        img_points = np.copy(bgr_img)
        draw_points(img_points, points, (0, 0, 255), 1, 2)
        cv2.imshow(winname, img_points)
        cv2.waitKey()
    
    best_ellipse = None
    best_support = float('-inf')
    best_inliers = None

    # Points on right and left of predicted pupil location (center of ROI-img)
    r_inds, l_inds = np.squeeze([pts_x > (bgr_img.shape[1] / 2)]), np.squeeze([pts_x < (bgr_img.shape[1] / 2)])
    
    # Not enough points to start process 
    if r_inds.size < 3 or l_inds.size < 3:  raise NoEllipseFound()  
        
    points_r = points[r_inds]
    points_l = points[l_inds]
    
    if len(points_r) < 3 or len(points_l) < 3:  raise NoEllipseFound()  
    
    # Perform N RANSAC iterations
    for ransac_iter in range(ransac_iters_max):
        
        try:
        
            sample = random.sample(points_r, 3) + (random.sample(points_l, 3))
            ellipse = fit_ellipse(sample, bgr_img.shape[:2])
            
            if debug:
                img_least_sqs = np.copy(bgr_img)
                draw_points(img_least_sqs, points, (0, 0, 255), 1, 2)
                draw_points(img_least_sqs, sample, (0, 255, 255), 6, 2)
                cv2.ellipse(img_least_sqs, ellipse.rotated_rect, (0, 255, 255), 1)
                print 'initial fit: ' + str(ransac_iter + 1)
                cv2.imshow(winname, img_least_sqs)
                cv2.imwrite("ransac_initial" + str(ransac_iter) + ".png", img_least_sqs)
                cv2.waitKey()
            
            # Image-aware sample rejection
            for (p_x, p_y) in sample:
                grad_x, grad_y = ellipse.algebraic_gradient_dir((p_x, p_y))
                dx, dy = image_dx[p_y][p_x], image_dy[p_y][p_x]
                
                # If sample and ellipse gradients don't agree, move to next set of samples
                if dx * grad_x + dy * grad_y <= 0:
                    if debug: print 'Sample and ellipse gradients do not agree, reject'
                    break
            
            else:   # Only continues for else-block did not break on above line (image-aware sample rejection)
                
                # Iteratively refine inliers further
                for _ in range(refine_iters_max):
                    
                    pts_distances = ellipse.distances(pts_x, pts_y)
                    inlier_inds = np.squeeze([np.abs(get_err_scale(ellipse) * pts_distances) < max_err])
                    inliers = points[inlier_inds]
                    
                    if len(inliers) < 5: raise NotEnoughInliers()
                    
                    ellipse = fit_ellipse(inliers, bgr_img.shape[:2])
                    
                    if debug:
                        img_refined = np.copy(bgr_img)
                        draw_points(img_refined, points, (0, 0, 255), 1, 2)
                        draw_points(img_refined, sample, (0, 255, 255), 6, 2)
                        cv2.ellipse(img_refined, ellipse.rotated_rect, (0, 255, 255), 1)
                        draw_points(img_refined, inliers, (0, 255, 0), 1, 2)
                        cv2.imshow(winname, img_refined)
                        cv2.waitKey()
                
                # Calculate the image aware support of the ellipse
                if image_aware_support:
                    
                    inliers_pts_x, inliers_pts_y = np.split(inliers, 2, axis=1)
                    inliers_pts_x, inliers_pts_y = np.squeeze(inliers_pts_x), np.squeeze(inliers_pts_y)
                    
                    # Ellipse gradients at inlier points
                    grads_x, grads_y = ellipse.algebraic_gradient_dirs(inliers_pts_x, inliers_pts_y)
                    
                    # Image gradients at inlier points
                    dxs = image_dx[inliers_pts_y.astype(int), inliers_pts_x.astype(int)]
                    dys = image_dy[inliers_pts_y.astype(int), inliers_pts_x.astype(int)]
                    
                    support = np.sum(dxs.dot(grads_x) + dys.dot(grads_y))
                        
                else: support = len(inliers)
                
                if support > best_support:
                    best_ellipse = ellipse
                    best_support = support
                    best_inliers = inliers
                
                # Early termination for > 95% inliers
                print len(inliers) / float(len(points))
                if len(inliers) / float(len(points)) > 0.95:
                    print "Early Termination"
                    break
                
        except NotEnoughInliers:
            if debug: print 'Not Enough Inliers'
        
        except BadEllipseShape as e:
            if debug: print 'Bad Ellipse Shape: %s' % e.msg
                
    if best_ellipse == None:
        raise NoEllipseFound()

    coverage = calculate_coverage(best_ellipse, best_inliers)
    if coverage < min_coverage:
        raise CoverageTooLow('Minimum inlier coverage: %d, actual coverage: %d' % (min_coverage, coverage))
    
    if debug:
        img_bgr_img = np.copy(bgr_img)
        cv2.ellipse(img_bgr_img, best_ellipse.rotated_rect, (0, 255, 0), 2)
        cv2.imshow(winname, img_bgr_img)
        cv2.waitKey()
    
    return best_ellipse
