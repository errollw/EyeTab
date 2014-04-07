import cv2, numpy as np



def get_contour_centre(contour):
    
    """ Robustly get centre of contour. Returns centroid if possible, otherwise centre of bounding box
    """
    
    moments = cv2.moments(contour)
    if moments['m00'] != 0.0:
        centroid_x = moments['m10'] / moments['m00']
        centroid_y = moments['m01'] / moments['m00']
        return (centroid_x, centroid_y)
    else:
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w / 2, y + h / 2)


def stack_imgs(imgs, vertical):
    
    ws = [img.shape[1] for img in imgs]
    hs = [img.shape[0] for img in imgs]
    max_w, max_h = max(ws), max(hs)

    max_dim = max_w if vertical else max_h
    
    imgs_resized = []
    for img in imgs:
        
        if len(img.shape) == 2:                         # 1-channel image -> greyscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Convert to BGR for mixing with color images
        
        if img.shape[1 if vertical else 0] < max_dim:
            old_h, old_w = img.shape[:2]
            img_resized = np.ones([old_h if vertical else max_dim,      # Choose resized img dimensions based on stack direction
                                   max_dim if vertical else old_w, 3],  # Must have 3 channels
                                  dtype=np.uint8) * 50                  # Background is dark grey (50,50,50) 
            
            img_resized[0:old_h, 0:old_w] = img         # Copy in smaller old image
            imgs_resized.append(img_resized)
        else:
            imgs_resized.append(img)
    
    stacked_imgs = np.concatenate(imgs_resized, axis=0 if vertical else 1)
    return stacked_imgs   

def stack_imgs_vertical(imgs):
    return stack_imgs(imgs, vertical=True)

def stack_imgs_horizontal(imgs):
    return stack_imgs(imgs, vertical=False)
    
    
def explore_img(winname, img_to_show, img_vals=None):
    
    """ Shows an image window which displays image values on mouse roll-over
    """
    
    def write_text(base_img, text):
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(base_img, text, (10, 20), font, 1, (0, 0, 0), thickness=3)       # shadow
        cv2.putText(base_img, text, (10, 20), font, 1, (255, 255, 255)) # text itself
    
    def on_mouse(event, x, y, flag, param):
        img_copy = img_to_show.copy()
        if 0 < x < img_to_show.shape[1] and 0 < y < img_to_show.shape[0]:
            local_val = img_to_show[y][x] if img_vals is None else img_vals[y][x]
            write_text(img_copy, str(local_val))
            cv2.imshow(winname, img_copy)

    cv2.imshow(winname, img_to_show)
    cv2.setMouseCallback(winname, on_mouse)


def pil_to_cv2(pil_img):
    
    """ Converts a PIL image to an OpenCV image
    """
    
    open_cv_img = np.array(pil_img.convert('RGB')) 
    open_cv_img = open_cv_img[:, :, ::-1].copy()
    return open_cv_img


def make_gauss_pyr(full_img, max_depth=4):
    
    """ Constructs a dict of depth:image where pyramid_image * scale = original_image
    """
    
    gauss_pyr = {}
    for i in range(max_depth):
        if i == 0:
            pyr_img = full_img
        else: 
            pyr_img = cv2.pyrDown(gauss_pyr[2 ** (i - 1)])
            
        gauss_pyr[2 ** i] = pyr_img
        
    return gauss_pyr


def measure_blurriness_LoG(img):

    """ Blurriness measure
    """

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(grey_img, (3, 3), 0)
    LoG_img = cv2.Laplacian(blur_img, cv2.CV_16S, ksize=5, scale=15, delta=0)
    
    thresh_val = np.percentile(LoG_img, 90)
    av_edge_strength = np.mean(LoG_img[LoG_img > thresh_val])
    
    return av_edge_strength


def measure_blurriness_DFT(img):
    
    """ More complex blurriness measure averaging top 90% of frequencies in image
    """
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    
    dftHeight = cv2.getOptimalDFTSize(blur_img.shape[0])
    dftWidth = cv2.getOptimalDFTSize(blur_img.shape[1])
    
    complexImg = np.zeros([dftHeight, dftWidth, 2], dtype=float)
    complexImg[0:img.shape[0], 0:img.shape[1], 0] = img / 255.0
            
    dft_img = cv2.dft(complexImg)
    dft_img = cv2.magnitude(dft_img[:, :, 0], dft_img[:, :, 1])
    dft_img = cv2.log(dft_img + 1)
    cv2.normalize(dft_img, dft_img, 0, 1, cv2.NORM_MINMAX)
    
    dft_img_h, dft_img_w = dft_img.shape[:2]
    win_size = dft_img_w * 0.55
    window = dft_img[dft_img_h / 2 - win_size:dft_img_h / 2 + win_size,
                     dft_img_w / 2 - win_size:dft_img_w / 2 + win_size]
 
    return np.mean(np.abs(window))


#----------------------------------------
# EXAMPLE USAGE 
#----------------------------------------
if __name__ == '__main__':

    bgr_img = cv2.imread('eye_images/erroll3_r.png', 3)
    measure_blurriness_DFT(bgr_img)
    measure_blurriness_DFT(cv2.GaussianBlur(bgr_img, (5, 5), 0))
    measure_blurriness_DFT(cv2.GaussianBlur(bgr_img, (7, 7), 0))
    cv2.waitKey()
    
    img = cv2.imread('face_images/erroll6.png', 3)
    
    gauss_pyr = make_gauss_pyr(img)
    
    for depth in gauss_pyr.keys():
        cv2.imshow("Depth %s" % str(depth), gauss_pyr[depth])
    
    cv2.waitKey()
