void gaze_system_init();

void track_gaze(Mat& frame_bgr, Mat& frame_gray);

Rect choose_best_eye_pair(vector<Rect> eye_pairs);