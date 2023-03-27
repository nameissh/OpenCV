// imread, imwrite
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	cout << "Hello OpenCV" << CV_VERSION << endl;

	Mat img;
	img = imread("lenna.bmp");							// default
	img = imread("lenna.bmp", IMREAD_GRAYSCALE);		// gray

	vector<int> params;
	params.push_back(IMWRITE_JPEG_QUALITY);
	params.push_back(95);
	imwrite("lenna.jpg", img, params);

	imshow("image", img);

	if (img.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}

	namedWindow("image");
	imshow("image", img);

	waitKey(0);
	return 0;
}*/

// keyboard
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(void)
{
	Mat img = imread("lenna.bmp");
	if (img.empty())
	{
		cerr << "Image load failed" << endl;
		return -1;						                 // ������ �ִ� ��� ��ȯ�� -1
	}

	namedWindow("img");
	imshow("img", img);

	while (true)
	{
		int keycode = waitKey();

		if (keycode == 'i' || keycode == 'I')
		{
			img = ~img;
			imshow("img", img);
		}
		else if (keycode == 27 || keycode == 'q' || keycode == 'Q')
		{
			break;
		}
	}

	return 0;
}*/

// mouse
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat img;					// ��������
Point ptOld;				// ��������
void on_mouse(int event, int x, int y, int flags, void*);

int main(void)
{
	img = imread("lenna.bmp");

	if (img.empty())
	{
		cerr << "Image load failed" << endl;
		return -1;
	}

	namedWindow("img");
	setMouseCallback("img", on_mouse);

	imshow("img", img);
	waitKey(0);

	return 0;
}

void on_mouse(int event, int x, int y, int flags, void*)
{
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		ptOld = Point(x, y);										// ���� ��ǥ�� ����
		cout << "EVENT_LBUTTONDOWN: " << x << "," << y << endl;
		break;

	case EVENT_LBUTTONUP:
		cout << "EVENT_LBUTTONUP: " << x << "," << y << endl;
		break;

	case EVENT_MOUSEMOVE:
		if (flags & EVENT_FLAG_LBUTTON)									// L��ư down(flag = 1) + ���콺 move �ΰ��� ���� �����Ǹ� ����
		{
			line(img, ptOld, Point(x, y), Scalar(0, 255, 255), 2);			// ��� ����
			imshow("img", img);
			ptOld = Point(x, y);
		}
		else if (flags & EVENT_FLAG_CTRLKEY)							// ctrl Ű ������(flag = 8) + ���콺 move �ΰ��� ���� �����Ǹ� ����
		{
			line(img, ptOld, Point(x, y), Scalar(0, 0, 255), 2);			// ���� ����
			imshow("img", img);
			ptOld = Point(x, y);
		}
		break;

		if (flags & EVENT_FLAG_LBUTTON)
		{
			if (flags == EVENT_FLAG_LBUTTON)
			{
				line(img, ptOld, Point(x, y), Scalar(0, 255, 255), 2);
			}
			else if (flags & EVENT_FLAG_CTRLKEY)
			{
				line(img, ptOld, Point(x, y), Scalar(0, 0, 255), 2);
			}

			imshow("img", img);
			ptOld = Point(x, y);
		}
		break;

	default:
		break;
	}
}*/

// trackbar
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void on_level_change(int pos, void* userdata);		// ������ݹ��Լ�

int main(void)
{
	Mat img = Mat::zeros(600, 600, CV_8UC1);		// �� �̹��� ���� (zeros �� ����)

	namedWindow("image");
	int value = 0;									// ������ ������ ����
	createTrackbar("level", "image", &value, 16, on_level_change, (void*)&img);			// createTrackbar �Լ�

	imshow("image", img);
	waitKey(0);

	return 0;
}

void on_level_change(int pos, void* userdata)
{
	Mat img = *(Mat*)userdata;		// ���ο� �̹��� ����

	img.setTo(pos * 16);			// Ʈ������ ��ġ(grayscale ����)�� ���� ��ü�� X16���� ����
	imshow("image", img);
}*/

// perspective
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src;
Point2f srcQuad[4], dstQuad[4];									// ��ǥ�� �迭 ����
void on_mouse(int event, int x, int y, int flags, void*userdata);

int main()
{
	src = imread("card.bmp");

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}

	namedWindow("src");
	setMouseCallback("src", on_mouse);						// srcâ�� ���콺 �ݹ� �Լ��� ���

	while (true)
	{
		if (waitKey() == 32)								// �����̽��� ������ �ݺ� ����
		{
			imshow("src", src);
		}

		else if (waitKey() == 27)							// ESC ������ �������� ��������
			break;
	}
	
	return 0;
}

void on_mouse(int event, int x, int y, int flags, void*)
{
	static int cnt = 0;											// ���콺 Ŭ��

	if (event == EVENT_LBUTTONDOWN)
	{
		if (cnt < 4)
		{
			srcQuad[cnt++] = Point2f(x, y);

			circle(src, Point(x, y), 5, Scalar(0, 0, 255), -1);			// �������� 5�� ������ ��
			imshow("src", src);

			if (cnt == 4)
			{
				int w = 200, h = 300;									// ���� ����

				//dstQuad[0] = Point2f(0, 0);								// �ð� �������� Ŭ���ϸ� �۵�
				//dstQuad[1] = Point2f(w - 1, 0);
				//dstQuad[2] = Point2f(w - 1, h - 1);
				//dstQuad[3] = Point2f(0, h - 1);

				vector< Point2f> dstQuad_new;
				dstQuad_new.push_back(Point(0, 0));
				dstQuad_new.push_back(Point(w - 1, 0));
				dstQuad_new.push_back(Point(w - 1, h - 1));
				dstQuad_new.push_back(Point(0, h - 1));

				Mat pers = getPerspectiveTransform(srcQuad, dstQuad);

				Mat dst;
				warpPerspective(src, dst, pers, Size(w, h));

				imshow("dst", dst);
			}
		}
	}
}*/

// inRange
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int lower_hue = 40, upper_hue = 80;								// �������� ����
Mat src, src_hsv, mask;

void on_hue_changed(int, void*);								// �ݹ��Լ�

int main()
{
	Mat src = imread("candies.png", IMREAD_COLOR);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}

	cvtColor(src, src_hsv, COLOR_BGR2HSV);						// src ������ HSV �� �������� ��ȯ�ؼ� ����

	imshow("src", src);

	namedWindow("mask");
	createTrackbar("Lower Hue", "mask", &lower_hue, 179, on_hue_changed);				// ���� �ִ밪 179 �� 360 / 2 = 180, 0 ~ 179
	createTrackbar("Upper Hue", "mask", &upper_hue, 179, on_hue_changed);
	on_hue_changed(0, 0);																// ���� ���� ����� ���� �ݹ��Լ� ����ȣ��

	waitKey();
	return 0;
}
void on_hue_changed(int, void*)
{
	Scalar lowerb(lower_hue, 100, 0);						// s(ä��) = 100 ~ 255, v(���) = 0 ~ 255, v�� ������ 0~255�� �����ؼ� ���� ������ ���� 
	Scalar upperb(upper_hue, 255, 255);
	inRange(src_hsv, lowerb, upperb, mask);					// src_hsv ���󿡼� HSV �� ���� ������ lowerb���� upperb ������ ��ġ�� �ȼ��� ������� ������ mask ���� ����

	imshow("mask", mask);
}*/

// backProjection_ver.1 �� �̹����� ��Ÿ���� �ʴ� ���� �߻�
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat ref_img, ref_ycrcb, ref_mask;
Point ptOld_mouse;
void on_mouse(int event, int x, int y, int flags, void*);

int main()
{
	Mat ref = imread("ref.png", IMREAD_COLOR);

	namedWindow("ref_img");
	setMouseCallback("ref_img", on_mouse);

	waitKey();
	return 0;
}
void on_mouse(int event, int x, int y, int flags, void*)
{
	vector<Point2f> pts_mouse;
	Mat ref_mask = Mat(ref_img.size(), CV_8UC1);

	if (flags && EVENT_FLAG_LBUTTON)
	{

		ptOld_mouse = Point(x, y);
		cout << "Point: " << x << "," << y << endl;
		pts_mouse.push_back(Point(x, y));
	}

	else if (flags && EVENT_FLAG_RBUTTON)
	{
		cout << "EVENT_RBUTTONDOWN!" << endl;

		polylines(ref_img, pts_mouse, true, Scalar(0, 0, 255), 2);
		fillPoly(ref_mask, pts_mouse, Scalar(255, 0, 255), 0);

		Mat hist;
		int channels[] = { 1,2 };
		int cr_bins = 128;
		int cb_bins = 128;
		int histSize[] = { cr_bins,cb_bins };
		float cr_range[] = { 0,256 };
		float cb_range[] = { 0,256 };
		const float* ranges[] = { cr_range, cb_range };

		calcHist(&ref_ycrcb, 1, channels, ref_mask, hist, 2, histSize, ranges);

		Mat src, src_ycrcb;
		src = imread("kids.png", IMREAD_COLOR);
		cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

		Mat backproj;
		calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);

		imshow("src", src);
		imshow("backproj", backproj);
	}

	else if (flags && EVENT_FLAG_MBUTTON)
	{
		cout << "EVENT_MBUTTONDOWN!" << endl;
		pts_mouse.clear();
	}
}*/

// backProjection_ver.2 �� ������ �̹����� ��Ÿ���� �ʴ� ���� �߻�
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat ref_img, ref_ycrcb, ref_mask;
vector<Point> pts_mouse;
void on_mouse(int event, int x, int y, int flags, void*);

int main()
{
	Mat ref = imread("ref.png", IMREAD_COLOR);

	cvtColor(ref, ref_ycrcb, COLOR_BGR2YCrCb);

	namedWindow("ref_img");
	setMouseCallback("ref_img", on_mouse);

	waitKey();
	return 0;
}
void on_mouse(int event, int x, int y, int flags, void*)
{
	vector<Point2f> pts_mouse;
	ref_mask = Mat(ref_img.size(), CV_8UC1);

	if (flags && EVENT_FLAG_LBUTTON)
	{
		cout << "Point: " << x << "," << y << endl;
		pts_mouse.push_back(Point(x, y));
		circle(ref_img, Point(x, y), 2, Scalar(0, 0, 255), -1);
		imshow("ref_img", ref_img);
	}

	else if (flags && EVENT_FLAG_RBUTTON)
	{
		cout << "EVENT_RBUTTONDOWN!" << endl;

		polylines(ref_img, pts_mouse, true, Scalar(0, 0, 255), 2);
		imshow("ref_img", ref_img);
		fillPoly(ref_mask, pts_mouse, Scalar(255, 255, 255), 0);
		imshow("ref_mask", ref_mask);

		Mat hist;
		int channels[] = { 1,2 };
		int cr_bins = 128;
		int cb_bins = 128;
		int histSize[] = { cr_bins,cb_bins };
		float cr_range[] = { 0,256 };
		float cb_range[] = { 0,256 };
		const float* ranges[] = { cr_range, cb_range };

		calcHist(&ref_ycrcb, 1, channels, ref_mask, hist, 2, histSize, ranges);

		imshow("hist", hist);
		
		Mat src, src_ycrcb;
		src = imread("kids.png", IMREAD_COLOR);
		cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

		Mat backproj;
		calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);

		imshow("src", src);
		imshow("backproj", backproj);
	}

	else if (flags && EVENT_FLAG_MBUTTON)
	{
		cout << "EVENT_MBUTTONDOWN!" << endl;
		pts_mouse.clear();
	}
}*/

// backProjection_ver.3 �� ref_img�� ���ο��� �ٽ� Mat���� �ʱ�ȭ���ָ鼭 ���� �߻���
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat ref_img, ref_ycrcb, ref_mask;
vector<Point> pts_mouse;
void on_mouse(int event, int x, int y, int flags, void* userdata);

int main()
{
	ref_img = imread("ref.png", IMREAD_COLOR);								// ���� ���������� Mat�� ����� ������ �� ���� ���� ����� �ȵ�
	ref_mask = Mat(ref_img.size(), CV_8UC1);								// mask = unsigned 8 bit char
	ref_mask.setTo(0);

	if (ref_img.empty() || ref_mask.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}

	cvtColor(ref_img, ref_ycrcb, COLOR_BGR2YCrCb);

	namedWindow("ref_img");
	setMouseCallback("ref_img", on_mouse);									// �ݹ� �Լ� ȣ��

	imshow("ref_img", ref_img);

	waitKey(0);
	return 0;
}
void on_mouse(int event, int x, int y, int flags, void*)
{
	if (flags & EVENT_FLAG_LBUTTON)											// L��ư�� ������
	{
		cout << "Point: " << x << "," << y << endl;
		pts_mouse.push_back(Point(x, y));									// ��ǥ ����
		circle(ref_img, Point(x, y), 2, Scalar(0, 0, 255), -1);
		imshow("ref_img", ref_img);											// ���׶�̰� ǥ�õ� ���� �ٽ� ������
	}

	else if (flags & EVENT_FLAG_RBUTTON)									// R��ư�� ������
	{
		cout << "EVENT_RBUTTONDOWN!" << endl;
		polylines(ref_img, pts_mouse, true, Scalar(0, 0, 255), 2);
		imshow("ref_img", ref_img);											// ��ǥ�� ���� ���� ������ ǥ�õ� ���� �ٽ� ������
		fillPoly(ref_mask, pts_mouse, Scalar(255, 255, 255), 0);			// ������ ���� ���� ������ ������� ĥ�ϰ� mask�� ����

		Mat hist;
		int channels[] = { 1,2 };
		int cr_bins = 128;
		int cb_bins = 128;
		int histSize[] = { cr_bins,cb_bins };
		float cr_range[] = { 0,256 };
		float cb_range[] = { 0,256 };
		const float* ranges[] = { cr_range, cb_range };

		calcHist(&ref_ycrcb, 1, channels, ref_mask, hist, 2, histSize, ranges);

		imshow("hist", hist);
		imshow("ref_mask", ref_mask);

		Mat src, src_ycrcb;
		src = imread("kids.png", IMREAD_COLOR);
		cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

		Mat backproj;
		calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);

		imshow("src", src);
		imshow("backproj", backproj);
	}

	else if (flags & EVENT_FLAG_MBUTTON)									// ��� ��ư�� ������
	{
		cout << "EVENT_MBUTTONDOWN!" << endl;
		pts_mouse.clear();													// �ʱ�ȭ

		Mat ref_img = imread("ref.png", IMREAD_COLOR);
		ref_mask = Mat(ref_img.size(), CV_8UC1);
		ref_mask.setTo(0);

		cvtColor(ref_img, ref_ycrcb, COLOR_BGR2YCrCb);

		imshow("ref_img", ref_img);
	}
}*/

// threshold
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void on_threshold(int pos, void* userdata);								// Ʈ���� �ݹ� �Լ�

int main(int argc, char* argv[])										// ����� ���� ����
{
	Mat src;

	if (argc < 2)
		src = imread("neutrophils.png", IMREAD_GRAYSCALE);

	else
		src = imread(argv[1], IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}

	imshow("src", src);

	namedWindow("dst");
	createTrackbar("Threshold", "dst", 0, 255, on_threshold, (void*)&src);			// Ʈ������ �ִ� = 255
	setTrackbarPos("Threshold", "dst", 124);

	waitKey(0);
	return 0;
}

void on_threshold(int pos, void* userdata)
{
	Mat src = *(Mat*)userdata;											// void* Ÿ���� ���� �̹���(userdata)�� Mat* Ÿ������ ����ȯ �� src ���� ����

	Mat dst;
	threshold(src, dst, pos, 255, THRESH_BINARY);
	int th = (int)threshold(src, dst, 0, 255, THRESH_BINARY | THRESH_OTSU);

	imshow("dst", dst);
}*/

// polygon
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void setLabel(Mat& img, const vector<Point>& pts, const String& label)							// img ���󿡼� pts �ܰ��� �ֺ��� �ٿ���ڽ��� �׸��� label ���ڿ� ���
{
	Rect rc = boundingRect(pts);																// pts �ܰ����� ���δ� �ٿ�� �ڽ�
	rectangle(img, rc, Scalar(0, 0, 255), 1);													// ��Ȳ�� �ڽ�
	putText(img, label, rc.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));						// �ٿ���ڽ� ������ܿ� label ���ڿ� ���
}

int main(int argc, char* argv[])
{
	Mat img = imread("polygon.bmp", IMREAD_COLOR);

	if (img.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);														// grayscale

	Mat bin;
	threshold(gray, bin, 200, 255, THRESH_BINARY_INV | THRESH_OTSU);							// �ڵ�����ȭ

	vector<vector<Point>> contours;
	findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);								// ��� ��ü�� �ٱ��� �ܰ��� ����

	for (vector<Point> pts : contours)															// �ܰ��� ������ŭ �ݺ�, �ܰ��� ��ǥ�� pts ������ ����
	{
		if (contourArea(pts) < 400)																// �ܰ����� ���δ� ������ 400���� ������ ����
			continue;

		vector<Point> approx;
		approxPolyDP(pts, approx, arcLength(pts, true) * 0.02, true);							// �ܰ����� �ٻ�ȭ�ؼ� approx�� ����

		int vtc = (int)approx.size();															// �ܰ��� ���� ������ vtc�� ����

		if (vtc == 3)																	// �ܰ��� ������ ������ 3�̸� �ﰢ��
		{
			setLabel(img, pts, "TRI");
		}

		else if (vtc == 4)																// �ܰ��� ������ ������ 4�̸� �簢��
		{
			setLabel(img, pts, "RECT");
		}

		else if (vtc > 4)
		{
			double len = arcLength(pts, true);
			double area = contourArea(pts);
			double ratio = 4. * CV_PI * area / (len * len);					// ��ü�� ���� �� ���� ����

			if (ratio > 0.8)												// ratio �������� 0.8���� ũ�� ���̶�� �Ǵ�
			{
				setLabel(img, pts, "CIR");
			}
		}
	}

	imshow("img", img);

	waitKey();

	return 0;
}*/

// convex hull
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat src = imread("Paper.bmp");
	Mat dst = src.clone();
	GaussianBlur(src, src, Size(3, 3), 0.0);

	Mat hsv;
	cvtColor(src, hsv, COLOR_BGR2HSV);

	Mat b_img;
	Scalar lowerb(0, 40, 0);
	Scalar upperb(20, 180, 255);
	inRange(hsv, lowerb, upperb, b_img);

	erode(b_img, b_img, Mat());
	dilate(b_img, b_img, cv::Mat(), Point(-1, -1), 2);

	vector<vector<Point>> contours;
	findContours(b_img, contours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	cout << "contours.size()= " << contours.size() << endl;

	if (contours.size() < 1)
		return 0;

	int maxK = 0;
	double maxArea = contourArea(contours[0]);

	for (int k = 1; k < contours.size(); k++)
	{
		double area = contourArea(contours[k]);

		if (area > maxArea)
		{
			maxK = k;
			maxArea = area;
		}
	}

	vector<Point> handContour = contours[maxK];
	vector<int> hull;
	convexHull(handContour, hull);													// points: �Է� ������, hull: ���� ��ü ���, clockwise/returnPoints ���� ����
	cout << "hull.size()= " << hull.size() << endl;

	vector<Point> ptsHull;
	for (int k = 0; k < hull.size(); k++)
	{
		int i = hull[k];
		ptsHull.push_back(handContour[i]);
	}

	drawContours(dst, vector<vector<Point>>(1, ptsHull), 0, Scalar(255, 0, 0), 2);

	imshow("dst_Hull", dst);

	vector<Vec4i> defects;
	convexityDefects(handContour, hull, defects);

	for (int k = 0; k < defects.size(); k++)
	{
		Vec4i v = defects[k];
		Point ptStart = handContour[v[0]];											// start: ������ ���� ���۵Ǵ� �������� �ε���
		Point ptEnd = handContour[v[1]];											// end: ������ ���� ������ �������� �ε���
		Point ptFar = handContour[v[2]];											// farthest: ���� ��ü���� ���� �� ������ ������ ������ �ε���, distance: farthest�� ���� ��ü���� �Ÿ�

		circle(dst, ptStart, 3, Scalar(0, 0, 255), 2);
		circle(dst, ptEnd, 3, Scalar(0, 0, 255), 2);
		circle(dst, ptFar, 3, Scalar(255, 0, 255), 2);
	}

	cout << "defects.size()= " << defects.size() << endl;
	imshow("dst", dst);

	waitKey();
	return 0;
}*/

// rockscissorspaper
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap(0);																					// camera_in()

	if (!cap.isOpened())
	{
		cerr << "Camera open failed!" << endl;
		return -1;
	}

	Scalar lowery = Scalar(20, 100, 100);																	// HSV color ����� ���� ����
	Scalar uppery = Scalar(30, 255, 255);
	Scalar lowerr = Scalar(0, 100, 100);																	// HSV color ������ ���� ����
	Scalar upperr = Scalar(10, 255, 255);

	Mat frame;
	Mat hsv_frame;

	while (true)
	{
		cap >> frame;																						// �ǽð� ���� frame���� �޾ƿ���
		if (frame.empty())
			break;

		cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

		Mat yellow_mask, yellow_frame;
		inRange(hsv_frame, lowery, uppery, yellow_mask);													// src_hsv ���󿡼� HSV �� ���� ������ lowerb���� upperb ������ ��ġ�� �ȼ��� ������� ������ mask ���� ����
		bitwise_and(frame, hsv_frame, yellow_frame, yellow_mask);											// frame + hsv_frame + yellow_mask = yellow_frame

		Mat red_mask, red_frame;
		inRange(hsv_frame, lowerr, upperr, red_mask);
		bitwise_and(frame, frame, red_frame, red_mask);														// frame + hsv_frame + red_mask = red_frame

		Mat labels, stats, centroids;
		int cnt1 = connectedComponentsWithStats(yellow_mask, labels, stats, centroids);						// ��ü ���̺� ������ return, cnt = labels - 1
		int cnt2 = connectedComponentsWithStats(red_mask, labels, stats, centroids);

		int rock = 0;
		int scissors = 0;
		int paper = 0;
		int num_yellow = cnt1 - 1;
		int num_red = cnt2 - 1;

		String text;

		if (num_red > 0)																					// ������ ��ƼĿ�� 0������ ������ (1��)
			text = "Rock!";

		else if (num_yellow == 2)
			text = "Scissors!";

		else if (num_yellow == 5)
			text = "Paper!";

		putText(frame, text, Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

		imshow("frame", frame);
		imshow("yellow", yellow_frame);
		imshow("red", red_frame);

		if (waitKey(1) == 'q')
			break;
	}

	return 0;
}