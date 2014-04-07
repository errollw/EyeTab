import cv2, numpy as np, os

board_w, board_h = 6, 9
board_size = (board_w, board_h)

img_w, img_h = 720, 1280
img_size = (img_w, img_h)
square_mm = 21

def get_obect_img_pts(board_img, debug=False):
    
    board_img_grey = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(
      image=board_img_grey,
      patternSize=board_size,
      flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS | cv2.CALIB_CB_FAST_CHECK)
    
    # Refine corner locations
    cv2.cornerSubPix(
      image=board_img_grey,
      corners=corners,
      winSize=(11, 11),
      zeroZone=(-1, -1),
      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1))
    
    if debug:
        chessboard_debug_img = np.copy(board_img)
        chessboard_debug_img = cv2.cvtColor(chessboard_debug_img,cv2.COLOR_BGR2GRAY)
        chessboard_debug_img = cv2.cvtColor(chessboard_debug_img,cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(chessboard_debug_img, board_size, corners, found)
        cv2.imshow("board", chessboard_debug_img)
        cv2.waitKey()
    
    # Correctly arrange arrays of corresponding points
    object_pts = np.zeros((np.prod(board_size), 3), np.float32)
    object_pts[:, :2] = np.indices(board_size).T.reshape(-1, 2) * square_mm
    image_pts = corners.reshape(-1, 2)
    
    return object_pts, image_pts


def get_intrinsic_camera_params(object_pts, image_pts, debug=False):
    
    _, cam_mat, dist_coefs, _, _ = cv2.calibrateCamera(
       objectPoints=object_pts,
       imagePoints=image_pts,
       imageSize=img_size)
    
    if debug:
        print dist_coefs
        print cam_mat
        print 'fx = {}, fy = {}'.format(cam_mat[0][0], cam_mat[1][1])
        print 'cx = {}, cy = {}'.format(cam_mat[0][2], cam_mat[1][2])
    
    return cam_mat, dist_coefs


#----------------------------------------
# FIND INTRINSIC CAMERA PARAMETERS 
#----------------------------------------
if __name__ == '__main__':

    calibration_imgs_path = 'camera_calibration'
    calibration_imgs = map(lambda x: cv2.imread(os.path.join(calibration_imgs_path, x)),
                            os.listdir(calibration_imgs_path))
    
    all_object_pts, all_img_pts = [], []
    for img in calibration_imgs:
        object_pts, image_pts = get_obect_img_pts(img, debug=True)
        all_object_pts.append(object_pts)
        all_img_pts.append(image_pts)
        
    cam_mat, dist_coefs = get_intrinsic_camera_params(all_object_pts, all_img_pts, True)
    
    params_file = open('camera_matrix_N7.txt', 'w')
    params_file.write('fx = {}\nfy = {}\ncx = {}\ncy = {}'.format(cam_mat[0][0], cam_mat[1][1], cam_mat[0][2], cam_mat[1][2]))
    params_file.close()
