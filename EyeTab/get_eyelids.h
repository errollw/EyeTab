using namespace cv;

vector<Point> guess_eye_corners(Point2i& eye_center, Point2i& eye_cs_vec);

Point find_upper_eyelid_pt(Point2i& eye_center, const Mat& eye_c_col);

Mat get_upper_eyelid(Point2i& eye_center, Point2i& eye_cs_vec, Mat& eye_bgr);

vector<Point2f> filter_poss_limb_pts(vector<Point2f>& poss_limb_pts, Mat& parabola, Point2i& roi_tl);

void draw_eyelid(Mat& img, Mat& parabola, Rect& roi, Scalar color = RED);