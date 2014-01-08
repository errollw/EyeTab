#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;
using namespace cv;

vector<Vector3d> ellipse_to_limbus(RotatedRect& ellipse);

Point2d get_gaze_pt_mm(Vector3d limb_center, Vector3d limb_normal);

Point2d get_gaze_pt_mm(RotatedRect& ellipse);

Point2i convert_gaze_pt_mm_to_px(Point2d gaze_pt_mm);

void show_gaze(Mat& img, vector<Point2i> gaze_pt_px_s, vector<Scalar> colors);
