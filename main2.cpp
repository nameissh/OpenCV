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
		return -1;						                 // 에러가 있는 경우 반환값 -1
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

Mat img;					// 전역변수
Point ptOld;				// 전역변수
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
		ptOld = Point(x, y);										// 기존 좌표값 저장
		cout << "EVENT_LBUTTONDOWN: " << x << "," << y << endl;
		break;

	case EVENT_LBUTTONUP:
		cout << "EVENT_LBUTTONUP: " << x << "," << y << endl;
		break;

	case EVENT_MOUSEMOVE:
		if (flags & EVENT_FLAG_LBUTTON)									// L버튼 down(flag = 1) + 마우스 move 두가지 조건 만족되면 실행
		{
			line(img, ptOld, Point(x, y), Scalar(0, 255, 255), 2);			// 노란 직선
			imshow("img", img);
			ptOld = Point(x, y);
		}
		else if (flags & EVENT_FLAG_CTRLKEY)							// ctrl 키 누르기(flag = 8) + 마우스 move 두가지 조건 만족되면 실행
		{
			line(img, ptOld, Point(x, y), Scalar(0, 0, 255), 2);			// 빨간 직선
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

void on_level_change(int pos, void* userdata);		// 사용자콜백함수

int main(void)
{
	Mat img = Mat::zeros(600, 600, CV_8UC1);		// 빈 이미지 생성 (zeros → 검정)

	namedWindow("image");
	int value = 0;									// 참조할 정수형 변수
	createTrackbar("level", "image", &value, 16, on_level_change, (void*)&img);			// createTrackbar 함수

	imshow("image", img);
	waitKey(0);

	return 0;
}

void on_level_change(int pos, void* userdata)
{
	Mat img = *(Mat*)userdata;		// 새로운 이미지 참조

	img.setTo(pos * 16);			// 트랙바의 위치(grayscale 레벨)에 따라 전체를 X16으로 맞춤
	imshow("image", img);
}*/

// perspective
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src;
Point2f srcQuad[4], dstQuad[4];									// 좌표값 배열 선언
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
	setMouseCallback("src", on_mouse);						// src창에 마우스 콜백 함수를 등록

	while (true)
	{
		if (waitKey() == 32)								// 스페이스바 누르면 반복 실행
		{
			imshow("src", src);
		}

		else if (waitKey() == 27)							// ESC 누르면 루프에서 빠져나옴
			break;
	}
	
	return 0;
}

void on_mouse(int event, int x, int y, int flags, void*)
{
	static int cnt = 0;											// 마우스 클릭

	if (event == EVENT_LBUTTONDOWN)
	{
		if (cnt < 4)
		{
			srcQuad[cnt++] = Point2f(x, y);

			circle(src, Point(x, y), 5, Scalar(0, 0, 255), -1);			// 반지름이 5인 빨간색 원
			imshow("src", src);

			if (cnt == 4)
			{
				int w = 200, h = 300;									// 임의 지정

				//dstQuad[0] = Point2f(0, 0);								// 시계 방향으로 클릭하면 작동
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

int lower_hue = 40, upper_hue = 80;								// 전역변수 선언
Mat src, src_hsv, mask;

void on_hue_changed(int, void*);								// 콜백함수

int main()
{
	Mat src = imread("candies.png", IMREAD_COLOR);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}

	cvtColor(src, src_hsv, COLOR_BGR2HSV);						// src 영상을 HSV 색 공간으로 변환해서 저장

	imshow("src", src);

	namedWindow("mask");
	createTrackbar("Lower Hue", "mask", &lower_hue, 179, on_hue_changed);				// 색상 최대값 179 → 360 / 2 = 180, 0 ~ 179
	createTrackbar("Upper Hue", "mask", &upper_hue, 179, on_hue_changed);
	on_hue_changed(0, 0);																// 영상 정상 출력을 위한 콜백함수 강제호출

	waitKey();
	return 0;
}
void on_hue_changed(int, void*)
{
	Scalar lowerb(lower_hue, 100, 0);						// s(채도) = 100 ~ 255, v(밝기) = 0 ~ 255, v의 범위를 0~255로 지정해서 범위 제약이 없음 
	Scalar upperb(upper_hue, 255, 255);
	inRange(src_hsv, lowerb, upperb, mask);					// src_hsv 영상에서 HSV 색 성분 범위가 lowerb에서 upperb 사이인 위치의 픽셀만 흰색으로 설정한 mask 영상 생성

	imshow("mask", mask);
}*/

// backProjection_ver.1 → 이미지가 나타나지 않는 오류 발생
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

// backProjection_ver.2 → 여전히 이미지가 나타나지 않는 오류 발생
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

// backProjection_ver.3 → ref_img를 메인에서 다시 Mat으로 초기화해주면서 문제 발생함
/*#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat ref_img, ref_ycrcb, ref_mask;
vector<Point> pts_mouse;
void on_mouse(int event, int x, int y, int flags, void* userdata);

int main()
{
	ref_img = imread("ref.png", IMREAD_COLOR);								// 위에 전역변수로 Mat을 써줬기 때문에 또 쓰면 영상 출력이 안됨
	ref_mask = Mat(ref_img.size(), CV_8UC1);								// mask = unsigned 8 bit char
	ref_mask.setTo(0);

	if (ref_img.empty() || ref_mask.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}

	cvtColor(ref_img, ref_ycrcb, COLOR_BGR2YCrCb);

	namedWindow("ref_img");
	setMouseCallback("ref_img", on_mouse);									// 콜백 함수 호출

	imshow("ref_img", ref_img);

	waitKey(0);
	return 0;
}
void on_mouse(int event, int x, int y, int flags, void*)
{
	if (flags & EVENT_FLAG_LBUTTON)											// L버튼이 눌리면
	{
		cout << "Point: " << x << "," << y << endl;
		pts_mouse.push_back(Point(x, y));									// 좌표 저장
		circle(ref_img, Point(x, y), 2, Scalar(0, 0, 255), -1);
		imshow("ref_img", ref_img);											// 동그라미가 표시된 영상 다시 보여줌
	}

	else if (flags & EVENT_FLAG_RBUTTON)									// R버튼이 눌리면
	{
		cout << "EVENT_RBUTTONDOWN!" << endl;
		polylines(ref_img, pts_mouse, true, Scalar(0, 0, 255), 2);
		imshow("ref_img", ref_img);											// 좌표를 따라서 빨간 점선이 표시된 영상 다시 보여줌
		fillPoly(ref_mask, pts_mouse, Scalar(255, 255, 255), 0);			// 점선을 따라서 만든 도형을 흰색으로 칠하고 mask에 저장

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

	else if (flags & EVENT_FLAG_MBUTTON)									// 가운데 버튼이 눌리면
	{
		cout << "EVENT_MBUTTONDOWN!" << endl;
		pts_mouse.clear();													// 초기화

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

void on_threshold(int pos, void* userdata);								// 트랙바 콜백 함수

int main(int argc, char* argv[])										// 명령행 인자 지정
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
	createTrackbar("Threshold", "dst", 0, 255, on_threshold, (void*)&src);			// 트랙바의 최댓값 = 255
	setTrackbarPos("Threshold", "dst", 124);

	waitKey(0);
	return 0;
}

void on_threshold(int pos, void* userdata)
{
	Mat src = *(Mat*)userdata;											// void* 타입의 세포 이미지(userdata)를 Mat* 타입으로 형변환 → src 변수 참조

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

void setLabel(Mat& img, const vector<Point>& pts, const String& label)							// img 영상에서 pts 외곽선 주변에 바운딩박스를 그리고 label 문자열 출력
{
	Rect rc = boundingRect(pts);																// pts 외곽선을 감싸는 바운딩 박스
	rectangle(img, rc, Scalar(0, 0, 255), 1);													// 주황색 박스
	putText(img, label, rc.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));						// 바운딩박스 좌측상단에 label 문자열 출력
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
	threshold(gray, bin, 200, 255, THRESH_BINARY_INV | THRESH_OTSU);							// 자동이진화

	vector<vector<Point>> contours;
	findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);								// 모든 객체의 바깥쪽 외곽선 검출

	for (vector<Point> pts : contours)															// 외곽선 개수만큼 반복, 외곽선 좌표를 pts 변수로 참조
	{
		if (contourArea(pts) < 400)																// 외곽선을 감싸는 면적이 400보다 작으면 무시
			continue;

		vector<Point> approx;
		approxPolyDP(pts, approx, arcLength(pts, true) * 0.02, true);							// 외곽선을 근사화해서 approx에 저장

		int vtc = (int)approx.size();															// 외곽선 점의 개수를 vtc에 저장

		if (vtc == 3)																	// 외곽선 꼭지점 개수가 3이면 삼각형
		{
			setLabel(img, pts, "TRI");
		}

		else if (vtc == 4)																// 외곽선 꼭지점 개수가 4이면 사각형
		{
			setLabel(img, pts, "RECT");
		}

		else if (vtc > 4)
		{
			double len = arcLength(pts, true);
			double area = contourArea(pts);
			double ratio = 4. * CV_PI * area / (len * len);					// 객체의 면적 대 길이 비율

			if (ratio > 0.8)												// ratio 변수값이 0.8보다 크면 원이라고 판단
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
	convexHull(handContour, hull);													// points: 입력 컨투어, hull: 볼록 선체 결과, clockwise/returnPoints 선택 지정
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
		Point ptStart = handContour[v[0]];											// start: 오목한 각이 시작되는 컨투어의 인덱스
		Point ptEnd = handContour[v[1]];											// end: 오목한 각이 끝나는 컨투어의 인덱스
		Point ptFar = handContour[v[2]];											// farthest: 볼록 선체에서 가장 먼 오목한 지점의 컨투어 인덱스, distance: farthest와 볼록 선체와의 거리

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

	Scalar lowery = Scalar(20, 100, 100);																	// HSV color 노란색 범위 지정
	Scalar uppery = Scalar(30, 255, 255);
	Scalar lowerr = Scalar(0, 100, 100);																	// HSV color 빨간색 범위 지정
	Scalar upperr = Scalar(10, 255, 255);

	Mat frame;
	Mat hsv_frame;

	while (true)
	{
		cap >> frame;																						// 실시간 영상 frame으로 받아오기
		if (frame.empty())
			break;

		cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

		Mat yellow_mask, yellow_frame;
		inRange(hsv_frame, lowery, uppery, yellow_mask);													// src_hsv 영상에서 HSV 색 성분 범위가 lowerb에서 upperb 사이인 위치의 픽셀만 흰색으로 설정한 mask 영상 생성
		bitwise_and(frame, hsv_frame, yellow_frame, yellow_mask);											// frame + hsv_frame + yellow_mask = yellow_frame

		Mat red_mask, red_frame;
		inRange(hsv_frame, lowerr, upperr, red_mask);
		bitwise_and(frame, frame, red_frame, red_mask);														// frame + hsv_frame + red_mask = red_frame

		Mat labels, stats, centroids;
		int cnt1 = connectedComponentsWithStats(yellow_mask, labels, stats, centroids);						// 전체 레이블 개수로 return, cnt = labels - 1
		int cnt2 = connectedComponentsWithStats(red_mask, labels, stats, centroids);

		int rock = 0;
		int scissors = 0;
		int paper = 0;
		int num_yellow = cnt1 - 1;
		int num_red = cnt2 - 1;

		String text;

		if (num_red > 0)																					// 빨간색 스티커가 0개보다 많으면 (1개)
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