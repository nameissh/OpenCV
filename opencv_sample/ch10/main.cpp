#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

//ColorOp
/*
void color_op();
void color_inverse();
void color_grayscale();
void color_split();

int main(void)
{
	//color_op();
	//color_inverse();
	//color_grayscale();
	color_split();

	return 0;
}

void color_op()
{
	Mat img = imread("butterfly.jpg", IMREAD_COLOR);

	if (img.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Vec3b& pixel = img.at<Vec3b>(0, 0);
	uchar b1 = pixel[0];
	uchar g1 = pixel[1];
	uchar r1 = pixel[2];

	Vec3b* ptr = img.ptr<Vec3b>(0);
	uchar b2 = ptr[0][0];
	uchar g2 = ptr[0][1];
	uchar r2 = ptr[0][2];

	//È®ÀÎ¿ë
	cout << "pixel  B:" << (int)b1 << " G:" << (int)g1 << " R:" << (int)r1 << endl;
	cout << "ptr    B:" << (int)b2 << " G:" << (int)g2 << " R:" << (int)r2 << endl;
}

void color_inverse()
{
	Mat src = imread("butterfly.jpg", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst(src.rows, src.cols, src.type());

	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			Vec3b& p1 = src.at<Vec3b>(j, i);
			Vec3b& p2 = dst.at<Vec3b>(j, i);

			p2[0] = 255 - p1[0]; // B
			p2[1] = 255 - p1[1]; // G
			p2[2] = 255 - p1[2]; // R
		}
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void color_grayscale()
{
	Mat src = imread("butterfly.jpg");

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst;
	cvtColor(src, dst, COLOR_BGR2GRAY);

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void color_split()
{
	Mat src = imread("candies.png");

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	imshow("src", src);
	imshow("B_plane", bgr_planes[0]);
	imshow("G_plane", bgr_planes[1]);
	imshow("R_plane", bgr_planes[2]);

	waitKey();
	destroyAllWindows();
}
*/

//coloreq
 /*
int main(void)
{
	Mat src = imread("pepper.bmp", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat src_ycrcb;
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

	vector<Mat> ycrcb_planes;
	split(src_ycrcb, ycrcb_planes);

	equalizeHist(ycrcb_planes[0], ycrcb_planes[0]); // Y channel

	Mat dst_ycrcb;
	merge(ycrcb_planes, dst_ycrcb);

	Mat dst;
	cvtColor(dst_ycrcb, dst, COLOR_YCrCb2BGR);

	imshow("src", src);
	imshow("dst", dst);

	waitKey(0);
	return 0;
}
*/

//inrange
 /*
int lower_hue = 40, upper_hue = 80;
Mat src, src_hsv, mask;

void on_hue_changed(int, void*);

int main(int argc, char* argv[])
{
	src = imread("candies.png", IMREAD_COLOR);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return -1;
	}

	cvtColor(src, src_hsv, COLOR_BGR2HSV);

	imshow("src", src);

	namedWindow("mask");
	createTrackbar("Lower Hue", "mask", &lower_hue, 179, on_hue_changed);
	createTrackbar("Upper Hue", "mask", &upper_hue, 179, on_hue_changed);
	on_hue_changed(0, 0);

	waitKey(0);
	return 0;
}

void on_hue_changed(int, void*)
{
	Scalar lowerb(lower_hue, 100, 0);
	Scalar upperb(upper_hue, 255, 255);
	inRange(src_hsv, lowerb, upperb, mask);

	imshow("mask", mask);
}
*/

//backProjection_1
/*
int main()
{
	// Calculate CrCb histogram from a reference image

	Mat ref, ref_ycrcb, mask;
	ref = imread("ref.png", IMREAD_COLOR);
	mask = imread("mask.bmp", IMREAD_GRAYSCALE);
	cvtColor(ref, ref_ycrcb, COLOR_BGR2YCrCb);


	Mat hist;
	int channels[] = { 1, 2 };
	int cr_bins = 128; int cb_bins = 128;//int cr_bins = 256; int cb_bins = 256;
	int histSize[] = { cr_bins, cb_bins };
	float cr_range[] = { 0, 256 };
	float cb_range[] = { 0, 256 };
	const float* ranges[] = { cr_range, cb_range };

	calcHist(&ref_ycrcb, 1, channels, mask, hist, 2, histSize, ranges);
	
	imshow("ref", ref);
	imshow("hist", hist);
	imshow("mask", mask);
	// Apply histogram backprojection to an input image

	Mat src, src_ycrcb;
	src = imread("kids.png", IMREAD_COLOR);
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

	Mat backproj;
	calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);

	imshow("src", src);
	imshow("backproj", backproj);
	waitKey(0);

	return 0;
}
*/

//backProjection_2

Mat ref_image,ref_mask, ref_ycrcb;

vector<Point> srcPoint;

void Calhist_BackPro();
void on_mouse(int event, int x, int y, int flags, void* userdata);

int main()
{
	// Calculate CrCb histogram from a reference image
	ref_image = imread("ref.png", IMREAD_COLOR);
	ref_mask = Mat(ref_image.size(), CV_8UC1);
	ref_mask.setTo(0);

	//mask = imread("mask.bmp", IMREAD_GRAYSCALE);
	cvtColor(ref_image, ref_ycrcb, COLOR_BGR2YCrCb);
	
	///
	namedWindow("ref_image");
	setMouseCallback("ref_image", on_mouse);

	imshow("ref_image", ref_image);

	
	waitKey(0);

	return 0;
}

void Calhist_BackPro()
{
	Mat hist;
	int channels[] = { 1, 2 };
	int cr_bins = 128; int cb_bins = 128;//int cr_bins = 256; int cb_bins = 256;
	int histSize[] = { cr_bins, cb_bins };
	float cr_range[] = { 0, 256 };
	float cb_range[] = { 0, 256 };
	const float* ranges[] = { cr_range, cb_range };

	calcHist(&ref_ycrcb, 1, channels, ref_mask, hist, 2, histSize, ranges);

	imshow("hist", hist);
	imshow("ref_mask", ref_mask);
	// Apply histogram backprojection to an input image
	
	Mat src, src_ycrcb;
	src = imread("kids.png", IMREAD_COLOR);
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

	Mat backproj;
	calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);

	imshow("src", src);
	imshow("backproj", backproj);
}


void on_mouse(int event, int x, int y, int flags, void*)
{
	static int cnt = 0;

	if (event == EVENT_LBUTTONDOWN)
	{
		srcPoint.push_back(Point(x, y));
		circle(ref_image, Point(x, y), 2, Scalar(0, 0, 255), -1);
		imshow("ref_image", ref_image);
	}

	if (event == EVENT_RBUTTONDOWN)
	{
		polylines(ref_image, srcPoint, true, Scalar(255, 0, 255), 1);
		imshow("ref_image", ref_image);

		fillPoly(ref_mask, srcPoint, Scalar(255, 255, 255));

		Calhist_BackPro();
	}

	if (event == EVENT_MBUTTONDOWN)
	{
		srcPoint.clear();
		ref_image = imread("ref.png", IMREAD_COLOR);
		ref_mask = Mat(ref_image.size(), CV_8UC1);
		ref_mask.setTo(0);

		//mask = imread("mask.bmp", IMREAD_GRAYSCALE);
		cvtColor(ref_image, ref_ycrcb, COLOR_BGR2YCrCb);

		imshow("ref_image", ref_image);

	}
}
