import visual as vpy
import numpy as np
import anatomical_constants

from math import sin, cos, acos, atan, radians, sqrt
from conic_section import Ellipse

cam_mat_n7 = np.array([[1062.348, 0.0     , 344.629],
                       [0.0     , 1065.308, 626.738],
                       [0.0     , 0.0     , 1.0]])

# [Looking at a camera facing the user]
# z points towards user's face
# x points to the left  (same as px coord direction)
# y points downwards    (same as px coord direction)

class Limbus:
    
    def __init__(self, centre_mm_param, normal_param, ransac_ellipse_param):
        
        self.center_mm = centre_mm_param
        self.normal = normal_param
        self.ransac_ellipse = ransac_ellipse_param


def ellipse_to_limbuses_persp_geom(ellipse, device):
    
    limbus_r_mm = anatomical_constants.limbus_r_mm
    focal_len_x_px, focal_len_y_px, prin_point_x, prin_point_y = device.get_intrisic_cam_params()
    focal_len_z_px = (focal_len_x_px + focal_len_y_px) / 2
    
    (x0_px, y0_px), (_, maj_axis_px), _ = ellipse.rotated_rect
    
    # Using iris_r_px / focal_len_px = iris_r_mm / distance_to_iris_mm
    iris_z_mm = (limbus_r_mm * 2 * focal_len_z_px) / maj_axis_px
    
    # Using (x_screen_px - prin_point) / focal_len_px = x_world / z_world
    iris_x_mm = -iris_z_mm * (x0_px - prin_point_x) / focal_len_x_px
    iris_y_mm = iris_z_mm * (y0_px - prin_point_y) / focal_len_y_px
    
    limbus_center = (iris_x_mm, iris_y_mm, iris_z_mm)
    
    (ell_x0, ell_y0), (ell_w, ell_h), angle = ellipse.rotated_rect                           
    new_rotated_rect = (ell_x0 - prin_point_x, ell_y0 - prin_point_y), (ell_w, ell_h), angle 
    ell = Ellipse(new_rotated_rect)                                                         
    
    f = focal_len_z_px;
    
    Z = np.array([[ell.A, ell.B / 2.0, ell.D / (2.0 * f)],
                  [ell.B / 2.0, ell.C, ell.E / (2.0 * f)],
                  [ell.D / (2.0 * f), ell.E / (2.0 * f), ell.F / (f * f)]])
    
    eig_vals, eig_vecs = np.linalg.eig(Z)
    
    idx = eig_vals.argsort()   
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    
    L1, L2, L3 = eig_vals[2], eig_vals[1], eig_vals[0]
    R = np.vstack([eig_vecs[:, 2], eig_vecs[:, 1], eig_vecs[:, 0]])
    
    g = sqrt((L2 - L3) / (L1 - L3))
    h = sqrt((L1 - L2) / (L1 - L3))
    
    poss_normals = [R.dot([h, 0, -g]), R.dot([h, 0, g]), R.dot([-h, 0, -g]), R.dot([-h, 0, g])]
    
    # Constraints
    nx, ny, nz = poss_normals[0 if iris_x_mm > 0 else 1]
    if nz > 0:
        nx, ny, nz = -nx, -ny, -nz
    if ny * nz < 0:
        ny *= -1
    if iris_x_mm > 0:
        if nx > 0: nx *= -1
    elif nx < 0: nx *= -1 

    return Limbus(limbus_center, [nx, ny, nz], ellipse)

def ellipse_to_limbuses_approx(ellipse, device):
    
    """ Returns 2 ambiguous limbuses
    """
    
    limbus_r_mm = anatomical_constants.limbus_r_mm
    focal_len_x_px, focal_len_y_px, prin_point_x, prin_point_y = device.get_intrisic_cam_params()
    focal_len_z_px = (focal_len_x_px + focal_len_y_px) / 2
    
    (x0_px, y0_px), (min_axis_px, maj_axis_px), angle = ellipse.rotated_rect
    
    # Using iris_r_px / focal_len_px = iris_r_mm / distance_to_iris_mm
    iris_z_mm = (limbus_r_mm * 2 * focal_len_z_px) / maj_axis_px
    
    # Using (x_screen_px - prin_point) / focal_len_px = x_world / z_world
    iris_x_mm = -iris_z_mm * (x0_px - prin_point_x) / focal_len_x_px
    iris_y_mm = iris_z_mm * (y0_px - prin_point_y) / focal_len_y_px
    
    limbus_center = (iris_x_mm, iris_y_mm, iris_z_mm)
    
    psi = radians(angle)                        # z-axis rotation (radians)
    tht_1 = acos(min_axis_px / maj_axis_px)     # y-axis rotation (radians)
    tht_2 = -tht_1                              # as acos has 2 ambiguous solutions
    
    # Find 2 possible normals for the limbus (weak perspective)
    normal_1 = vpy.vector(sin(tht_1) * cos(psi), -sin(tht_1) * sin(psi), -cos(tht_1))
    normal_2 = vpy.vector(sin(tht_2) * cos(psi), -sin(tht_2) * sin(psi), -cos(tht_2))
    
    # Now correct for weak perspective by modifying angle by offset between camera axis and limbus
    x_correction = -atan(iris_y_mm / iris_z_mm)
    y_correction = -atan(iris_x_mm / iris_z_mm)
    x_axis, y_axis = vpy.vector(1, 0, 0), vpy.vector(0, -1, 0)  # VPython uses different y axis
    
    normal_1 = vpy.rotate(normal_1, y_correction, y_axis)
    normal_1 = vpy.rotate(normal_1, x_correction, x_axis).astuple()
    normal_2 = vpy.rotate(normal_2, y_correction, y_axis)
    normal_2 = vpy.rotate(normal_2, x_correction, x_axis).astuple()

    return Limbus(limbus_center, normal_1, ellipse)


def get_gaze_point_px(limbus):

    """ Convenience method for getting gaze point on screen in px
    """

    gaze_point_mm = get_gaze_point_mm(limbus)
    return convert_gaze_pt_mm_to_px(gaze_point_mm)


def get_gaze_point_mm(limbus):
    
    """ Returns intersection with z-plane of optical axis vector (mm)
    """

    # Ray-plane intersection
    x0, y0, z0 = limbus.center_mm
    dx, dy, dz = limbus.normal
    t = -z0 / dz
    
    x_screen_mm, y_screen_mm = x0 + dx * t, y0 + dy * t

    return x_screen_mm, y_screen_mm
  
    
def convert_gaze_pt_mm_to_px((x_screen_mm, y_screen_mm), device):

    """ Returns intersection with screen in coordinates (px)
    """
    
    screen_w_mm, screen_h_mm = device.screen_size_mm
    screen_w_px, screen_h_px = device.screen_size_px
    screen_y_offset_px = device.screen_y_offset_px      # height of notification bar
    x_offset, y_offset = device.offset_mm               # screen offset from camera position
    
    x_screen_px = (x_screen_mm + x_offset) / screen_w_mm * screen_w_px
    y_screen_px = (y_screen_mm - y_offset) / screen_h_mm * screen_h_px - screen_y_offset_px
    
    return x_screen_px, y_screen_px
