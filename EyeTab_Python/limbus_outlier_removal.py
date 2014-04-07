import gaze_geometry
import math

# Screen size constants
screen_w_mm_n7, screen_h_mm_n7 = 94, 151
screen_w_px_n7, screen_h_px_n7 = 800, 1280

__last_pos = [0, 0, 0]

def remove_outliers(limbuses):
    
    limbuses_to_return = [limbuses[0], limbuses[1]]
    
    for i, limbus in enumerate(limbuses):
        
        # Remove limbuses that point widly out of frame
        if limbus is None:
            limbuses_to_return[i] = None
            continue
        
        gaze_pts_mm_x, gaze_pts_mm_y = gaze_geometry.get_gaze_point_mm(limbus)
        if gaze_pts_mm_x < -screen_w_mm_n7 or gaze_pts_mm_x > screen_w_mm_n7:
            limbuses_to_return[i] = None
            continue
        elif gaze_pts_mm_y < (-screen_h_mm_n7 * 3 / 2) or (gaze_pts_mm_y > 0):
            limbuses_to_return[i] = None
            continue
        else: limbuses_to_return[i] = limbus
    
        
    # Remove limbuses that are too far from eachother (Pupilary Distance)
    if limbuses_to_return[0] is not None and limbuses_to_return[1] is not None:
        
        l1x, l1y, l1z = limbuses_to_return[0].center_mm
        l2x, l2y, l2z = limbuses_to_return[1].center_mm
        limb_dist = math.sqrt((l1x - l2x) ** 2 + (l1y - l2y) ** 2 + (l1z - l2z) ** 2)
        
        last_x, last_y, last_z = __last_pos

        if limb_dist > 80:
            dist_1 = (last_x - l1x) ** 2 + (last_y - l1y) ** 2 + (last_z - l1z) ** 2
            dist_2 = (last_x - l2x) ** 2 + (last_y - l2y) ** 2 + (last_z - l2z) ** 2
            if dist_1 < dist_2:
                limbuses_to_return[1] = None
            else : limbuses_to_return[0] = None
            
        else: __last_pos[0], __last_pos[1], __last_pos[2] = (l1x + l2x) / 2, (l1y + l2y) / 2, (l1z + l2z) / 2
    
    return limbuses_to_return

