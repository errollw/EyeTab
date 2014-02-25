import cv2, numpy as np

# initialize the input video
vid = cv2.VideoCapture('input.mp4')
ret, frame = vid.read()

# video to write the flipped input video to
out = cv2.VideoWriter('output.avi', cv2.cv.CV_FOURCC(*'XVID'), 20.0, (1280,720))

while ret:
    frame = cv2.flip(frame,-1)  # flip frame on both axes
    out.write(frame)            # write it out
    cv2.imshow('frame',frame)   # show for debugging

    ret, frame = vid.read()     # read in a new frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break