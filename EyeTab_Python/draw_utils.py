import cv2
import numpy as np


def draw_cross(bgr_img, (x, y), color=(255, 255, 255), width=2, thickness=1):
    
    """ Draws an "x"-shaped cross at (x,y)
    """
    
    x, y, w = int(x), int(y), int(width / 2)  # ensure points are ints for cv2 methods
    
    cv2.line(bgr_img, (x - w , y - w), (x + w , y + w), color, thickness)
    cv2.line(bgr_img, (x - w , y + w), (x + w, y - w), color, thickness)
    
    
def draw_points(bgr_img, points, color=(255, 255, 255), width=2, thickness=1):
    
    """ Draws an "x"-shaped cross at each point in a list
    """

    for point in points:
        draw_cross(bgr_img, point, color, width, thickness)


def blank_screen(device=None, screen_height=1280, screen_width=720, scale=0.5):

    """ Creates a blank screen image for further drawing on
    """
    screen_w_px, screen_h_px = (screen_width, screen_height) if device is None else device.screen_size_px
    img_w, img_h = screen_w_px * scale, screen_h_px * scale
    
    size = int(16 * scale)   
    x0, dx = int(img_w / 6), int(img_w / 3)
    y0, dy = int(img_h / 6), int(img_h / 3)
    
    screen_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    targets = [(x0 + i * dx, y0 + j * dy) for j in range(3) for i in range(3)]
    map(lambda x : cv2.circle(screen_img, x, size, (255, 255, 255)), targets)
    
    return screen_img


def draw_gaze(screen_img, gaze_pts, gaze_colors, scale=0.5, return_img=False, cross_size=16, thickness=1):
    
    """ Draws an "x"-shaped cross on a screen for given gaze points, ignoring missing ones
    """
    
    width = int(cross_size * scale)
    
    for i, pt in enumerate(gaze_pts):
        if pt is None: continue
        draw_cross(screen_img, (pt[0] * scale, pt[1] * scale), gaze_colors[i], width, thickness)


def draw_normal(img, limbus, device, arrow_len_mm=10, color=(255, 255, 255), thickness=1, scale=1):
    
    """ Draws an arrow pointing towards screen transformed by matrix
    """
    
    focal_len_x_px, focal_len_y_px, prin_point_x, prin_point_y = device.get_intrisic_cam_params()
    
    long_normal = map(lambda x : x * arrow_len_mm, limbus.normal)
    arrow_pts_mm = [limbus.center_mm, map(sum, zip(limbus.center_mm, long_normal))]
    
    # Mirror the normal in the x direction for drawing onto the video captured as-is by camera
    arrow_trans_x = map(lambda v : int((v[0] / v[2] * -focal_len_x_px + prin_point_x) * scale), arrow_pts_mm)
    arrow_trans_y = map(lambda v : int((v[1] / v[2] * focal_len_y_px + prin_point_y) * scale), arrow_pts_mm)
    
    arrow_trans_tuple = zip(arrow_trans_x, arrow_trans_y)

    cv2.circle(img, arrow_trans_tuple[0][:2], 3, color, -1)
    cv2.line(img, arrow_trans_tuple[0][:2], arrow_trans_tuple[1][:2], color, thickness)
    

def draw_limbus(img, limbus, color=(255, 255, 255), thickness=1, scale=1):
    
    """ Draws the 2d ellipse of the limbus
    """
    
    (ell_x0, ell_y0), (ell_w, ell_h), angle = limbus.ransac_ellipse.rotated_rect
    
    (ell_x0, ell_y0), (ell_w, ell_h) = (ell_x0 * scale, ell_y0 * scale), (ell_w * scale, ell_h * scale)
    
    cv2.ellipse(img, ((ell_x0, ell_y0), (ell_w, ell_h), angle), color, thickness)

    
def draw_eyelids(eyelid_t, eyelid_b, eye_img):
    
    """ Draws the parabola for top eyelid and line for bottom eyelid (if they exist)
    """
    
    if eyelid_t is not None:
        a, b, c = eyelid_t
        lid_xs = [x * eye_img.shape[1] / 20 for x in range(21)]
        lid_ys = [a * x ** 2 + b * x + c for x in lid_xs]
        pts_as_array = np.array([[x, y] for (x, y) in zip(lid_xs, lid_ys)], np.int0)
        cv2.polylines(eye_img, [pts_as_array], False, (0, 255, 0))
    if eyelid_b is not None:
        a, b = eyelid_b
        start_pt, end_pt = (0, int(b)), (eye_img.shape[1], int(a * eye_img.shape[1] + b))
        cv2.line(eye_img, start_pt, end_pt, (0, 255, 0))


def draw_histogram(img, bin_width=4):
    
    """ Calculates and plots a histogram (good for BGR / LAB)
    """
    
    hist_img = np.zeros((300, 256, 3))

    bin_count = 256 / bin_width     
    bins = np.arange(bin_count).reshape(bin_count, 1) * bin_width
    debug_colors = [ (255, 0, 0), (0, 255, 0), (0, 0, 255) ]
    
    for ch, col in enumerate(debug_colors):
        hist_item = cv2.calcHist([img], [ch], None, [bin_count], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(hist_img, [pts], False, col)
     
    hist_img = np.flipud(hist_img)
     
    cv2.imshow('hist', hist_img)
    

def draw_histogram_hsv(hsv_img, bin_width=2):
    
    """ Calculates and plots 2 histograms next to each other: one for hue, and one for saturation and value
    """
    
    sv_hist_img, h_hist_img = np.zeros((300, 256, 3)), np.zeros((300, 360, 3))
    sv_bin_count, h_bin_count = 256 / bin_width    , 180 / bin_width
    
    sv_bins = np.arange(sv_bin_count).reshape(sv_bin_count, 1) * bin_width
    h_bins = np.arange(h_bin_count).reshape(h_bin_count, 1) * bin_width * 2
    
    debug_colors = [ (255, 255, 255), (255, 0, 0), (0, 0, 255) ]
    
    # Use ternary conditional for outputting to 2 different hists - a bit of a hack
    for ch, col in enumerate(debug_colors):
        hist_item = cv2.calcHist([hsv_img], [ch], None, [h_bin_count if ch == 0 else sv_bin_count], [0, 180 if ch == 0 else 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((h_bins if ch == 0 else sv_bins, hist))
        cv2.polylines(h_hist_img if ch == 0 else sv_hist_img, [pts], False, col)
    
    sv_hist_img, h_hist_img = np.flipud(sv_hist_img), np.flipud(h_hist_img)
    h_hist_img[:, 0] = (0, 255, 0)
    
    cv2.imshow('sat / val hist | hue hist', np.concatenate([sv_hist_img, h_hist_img], axis=1))


#----------------------------------------
# EXAMPLE USAGE 
#----------------------------------------
if __name__ == '__main__':
    
    bgr_img = cv2.imread('eye_images/andreas2_l.png', 3)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    
    cv2.imshow("bgr_img", bgr_img)
    
    draw_histogram(bgr_img)
    draw_histogram_hsv(hsv_img)
    
    cv2.waitKey()
