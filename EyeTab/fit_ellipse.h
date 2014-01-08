using namespace cv;
using namespace std;

RotatedRect fit_ellipse(vector<Point2f> edgePoints, Mat_<float> mPupilSobelX, Mat_<float> mPupilSobelY);