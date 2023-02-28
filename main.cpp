/*#include "opencv2/opencv.hpp"
#include <iostream>

int main()
{
	std::cout << "Hello OpenCV " << CV_VERSION << std::endl;

	return 0;
}*/

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

String filename = "mydata.xml";					// 데이터 파일 이름 전역 변수 선언
//String filename = "mydata.yml";
//String filename = "mydata.json";

void read_write();
void Point_pt();
void Rect_rc();
void RotatedRect_rrc();
void MatOp1();
void MatOp2();
void MatOp3();
void MatOp4();
void MatOp5();
void MatOp6();
void MatOp7();
void VecOp();
void ScalarOp();
void camera_in();
void video_in();
void camera_in_video_out();
void drawLines();
void drawPolys();
void drawText1();
void drawText2();
void keyboard();
void trackbar();
void on_level_change(int pos, void* userdata);
void writeData();
void readData();
void mask_setTo();
void mask_copyTo();
void time_inverse();
void gray_sum();
void color_sum();
void min_max();
void norm1();
void norm2();
void round();
void brightness1();
void brightness2();
void brightness3();
void brightness4();
void brightness5();
void on_brightness(int pos, void* userdata);
void contrast1();
void contrast2();
void histogram_cameraMan();
Mat calGrayHist(const Mat& img);
Mat getGrayHistImage(const Mat& hist);
Mat getBlueHistImage(const Mat& hist);
Mat getGreenHistImage(const Mat& hist);
Mat getRedHistImage(const Mat& hist);
void histogram_cameraMan2();
void histogram_stretching();
void histogram_equalization();
void arithmetic();
void logical();
void arit_logi();
void filter_embossing();
void blurring_mean();
void blurring_gaussian();
void unsharp_mask();
void noise_gaussian();
void filter_bilateral();
void filter_median();
void affine_transform();
void affine_translation();
void affine_shear();
void affine_scale();
void affine_rotation1();
void affine_rotation2();
void affine_flip();
void sobel_edge();
void canny_edge();
void hough_lines();
void hough_line_segments();
void hough_circles1();
void hough_circles2();
void color_inverse();
void color_split();
void color_equal();
int lower_hue = 40, upper_hue = 80;
Mat candy, candy_hsv, candy_mask;
void color_hue();
void on_hue_changed(int, void*);
void color_backproj1();
Mat ref_img, ref_ycrcb, ref_mask;
vector<Point> pts_mouse;
void on_mouse(int event, int x, int y, int flags, void* userdata);
void color_backproj2();
void adaptive();
void on_trackbar(int pos, void* userdata);
void struct_element();
void erode_dilate();
void open_close();
void labeling_basic1();
void labeling_basic2();
void labeling_stats();
void contours_basic();
void contours_hier();
void bounding_min_rect();


int main()
{
	// cout << "Hello OpenCV " << CV_VERSION << endl;

	//read_write();
	//Point_pt();
	//Rect_rc();
	//RotatedRect_rrc();
	//MatOp1();
	//MatOp2();
	//MatOp3();
	//MatOp4();
	//MatOp5();
	//MatOp6();
	//MatOp7();
	//VecOp();
	//ScalarOp();
	//camera_in();
	//video_in();
	//camera_in_video_out();
	//drawLines();
	//drawPolys();
	//drawText1();
	//drawText2();
	//keyboard();
	//trackbar();
	//writeData();
	//readData();
	//mask_setTo();
	//mask_copyTo();
	//time_inverse();
	//gray_sum();
	//color_sum();
	//min_max();
	//norm1();
	//norm2();
	//round();
	//brightness1();
	//brightness2();
	//brightness3();
	//brightness4();
	//brightness5();
	//contrast1();
	//contrast2();
	//histogram_cameraMan();
	//histogram_cameraMan2();
	//histogram_stretching();
	//histogram_equalization();
	//arithmetic();
	//logical();
	//arit_logi();
	//filter_embossing();
	//blurring_mean();
	//blurring_gaussian();
	//unsharp_mask();
	//noise_gaussian();
	//filter_bilateral();
	//filter_median();
	//affine_transform();
	//affine_translation();
	//affine_shear();
	//affine_scale();
	//affine_rotation1();
	//affine_rotation2();
	//affine_flip();
	//sobel_edge();
	//canny_edge();
	//hough_lines();
	//hough_line_segments();
	//hough_circles1();
	//hough_circles2();
	//color_inverse();
	//color_split();
	//color_equal();
	//color_hue();
	//color_backproj1();
	//color_backproj2();
	//adaptive();
	//struct_element();
	//erode_dilate();
	//open_close();
	//labeling_basic1();
	//labeling_basic2();
	//labeling_stats();
	//contours_basic();
	//contours_hier();
	//bounding_min_rect();
	
	return 0;
}

void read_write()
{
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
		return;
	}

	namedWindow("image");
	imshow("image", img);

	waitKey(0);
}

void Point_pt()
{
	Point pt1;
	pt1.x = 5; pt1.y = 10;
	Point pt2(10, 30);

	Point pt3 = pt1 + pt2;
	Point pt4 = pt1 * 2;
	int d1 = pt1.dot(pt2);
	bool b1 = (pt1 == pt2);

	cout << "pt1: " << pt1 << endl;
	cout << "pt2: " << pt2 << endl;
}

void Rect_rc()
{
	Rect rc1;								// rc1 = [0*0 from (0, 0)]
	Rect rc2(10, 10, 60, 40);				// rc2 = [60*40 from (10,10)]

	Rect rc3 = rc1 + Size(50, 40);			// rc3 = [50*40 from (0, 0)]
	Rect rc4 = rc2 + Point(10, 10);			// rc4 = [60*40 from (20, 20)]

	Rect rc5 = rc3 & rc4;					// rc5 = [30*20 from (10, 10)]
	Rect rc6 = rc3 | rc4;					// rc6 = [80*60 from (0, 0)]

	cout << "rc5: " << rc5 << endl;
	cout << "rc6: " << rc6 << endl;
}

void RotatedRect_rrc()
{
	RotatedRect rr1(Point2f(40, 30), Size2f(40, 20), 30.f);

	Point2f pts[4];
	rr1.points(pts);

	Rect br = rr1.boundingRect();

	cout << "pts[0]: " << pts[0] << endl;
	cout << "pts[1]: " << pts[1] << endl;
	cout << "pts[2]: " << pts[2] << endl;
	cout << "pts[3]: " << pts[3] << endl;
	cout << "br: " << br << endl;
}

void MatOp1()
{
	Mat img1;								// emtpy matrix

	Mat img2(480, 640, CV_8UC1);			// unsigned char, 1-channel
	Mat img3(480, 640, CV_8UC3);			// unsigned char, 3-channels
	Mat img4(Size(640, 480), CV_8UC3);		// Size (width, height)

	Mat img5(480, 640, CV_8UC1, Scalar(128));		// initial values, 128
	Mat img6(480, 640, CV_8UC3, Scalar(0, 0, 255));		// initial values, red

	Mat mat1 = Mat::zeros(3, 3, CV_32SC1);				// 0's matrix
	Mat mat2 = Mat::ones(3, 3, CV_32FC1);				// 1's matrix
	Mat mat3 = Mat::eye(3, 3, CV_32FC1);				// identity matrix

	float data[] = { 1,2,3,4,5,6 };
	Mat mat4(2, 3, CV_32FC1, data);

	Mat mat5 = (Mat_<float>(2, 3) << 1, 2, 3, 4, 5, 6);
	Mat mat6 = Mat_<uchar>({ 2,3 }, { 1,2,3,4,5,6 });

	mat4.create(256, 256, CV_8UC3);						// uchar, 3-channels
	mat5.create(4, 4, CV_32FC1);						// float, 1-channel

	mat4 = Scalar(255, 0, 0);
	mat5.setTo(1.f);
}

void MatOp2()
{
	Mat img1 = imread("dog.bmp");

	Mat img2 = img1;							// 복사 생성자 얕은 복사
	Mat img3;
	img3 = img1;								// 복사 생성자 얕은 복사

	Mat img4 = img1.clone();					// 깊은 복사
	Mat img5;
	img1.copyTo(img5);							// 깊은 복사

	img1.setTo(Scalar(0, 255, 255));			// yellow

	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);
	imshow("img4", img4);
	imshow("img5", img5);

	waitKey();
	destroyAllWindows();
}

void MatOp3()
{
	Mat img1 = imread("cat.bmp");

	if (img1.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat img2 = img1(Rect(220, 120, 340, 240));			// 얕은 복사
	Mat img3 = img1(Rect(220, 120, 340, 240)).clone();			// 깊은 복사

	img2 = ~img2;			// 반전

	imshow("img1", img1);
	imshow("img2", img2);
	imshow("img3", img3);

	waitKey();
	destroyAllWindows();
}

void MatOp4()
{
	Mat mat1 = Mat::zeros(3, 4, CV_8UC1);			// 4by3 행렬 초기화

	for (int j = 0; j < mat1.rows; j++)				// mat1.rows = 3
	{
		for (int i = 0; i < mat1.cols; i++)			// mat1.cols = 4
		{
			mat1.at<uchar>(j, i)++;
		}
	}

	for (int j = 0; j < mat1.rows; j++)				// Mat::ptr()
	{
		uchar* p = mat1.ptr<uchar>(j);
		for (int i = 0; i < mat1.cols; i++)
		{
			p[i]++;
		}
	}

	for (MatIterator_<uchar>it = mat1.begin<uchar>(); it != mat1.end<uchar>(); ++it)		// MatIterator_ 반복자
	{
		(*it)++;
	}
	cout << "mat1:\n" << mat1 << endl;
}

void MatOp5()
{
	Mat img1 = imread("lenna.bmp");							// truecolor
	// Mat img1 = imread("lenna.bmp",IMREAD_GRAYSCALE);		// grayscale

	cout << "Width: " << img1.cols << endl;
	cout << "Height: " << img1.rows << endl;
	cout << "Channels: " << img1.channels() << endl;

	if (img1.type()==CV_8UC1)
		cout << "img5 is a grayscale image." << endl;
	else if (img1.type()==CV_8UC3)
		cout << "img5 is a truecolor image." << endl;

	float data[] = { 2.f, 1.414f, 3.f, 1.732f };
	Mat mat1(2, 2, CV_32FC1, data);
	cout << "mat1:\n " << mat1 << endl;
}

void MatOp6()
{
	float data[] = { 1,1,2,3 };
	Mat mat1(2, 2, CV_32FC1, data);
	cout << "mat1:\n" << mat1 << endl;

	Mat mat2 = mat1.inv();				// 행렬의 역행렬
	cout << "mat2:\n" << mat2 << endl;

	cout << "mat1.t():\n" << mat1.t() << endl;			// 전치 행렬
	cout << "mat1 + 3:\n" << mat1 + 3 << endl;
	cout << "mat1 + mat2:\n" << mat1 + mat2 << endl;
	cout << "mat1 * mat2:\n" << mat1 * mat2 << endl;	// 행렬 * 역행렬 = 단위행렬
}

void MatOp7()
{
	Mat img1 = imread("lenna.bmp", IMREAD_GRAYSCALE);		// unsigned char 기본

	Mat img1f;												// float 행렬 생성
	img1.convertTo(img1f, CV_32FC1);						// 32 bit float

	uchar data1[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	Mat mat1(3, 4, CV_8UC1, data1);								// 4by3 행렬
	Mat mat2 = mat1.reshape(0, 1);								// 0채널, 1행

	cout << "mat1:\n" << mat1 << endl;
	cout << "mat2:\n" << mat2 << endl;

	Mat mat3 = Mat::ones(1, 4, CV_8UC1) * 255;					// 각 원소 * 255
	mat1.push_back(mat3);										// mat1에 mat3 추가
	cout << "mat1:\n" << mat1 << endl;

	mat1.resize(6, 100);										// 6행으로 변경, 나머지 원소 100으로 채우기
	cout << "mat1:\n" << mat1 << endl;

	mat1.pop_back(3);											// 마지막 3행을 제거, 비워두면 default = 1
	cout << "mat1:\n" << mat1 << endl;
}

void VecOp()
{
	Vec3b p1, p2(0, 0, 255);						// p1(0, 0, 0)
	p1[0] = 100;									// p1.val[0] = 100

	cout << "p1: " << p1 << endl;
	cout << "p2: " << p2 << endl;
}

void ScalarOp()
{
	Scalar gray = 128;
	cout << "gray: " << gray << endl;

	Scalar yellow(0, 255, 255);
	cout << "yellow: " << yellow << endl;

	Mat img1(256, 256, CV_8UC3, yellow);				// yellow로 초기화된 img1 생성

	for (int i = 0; i < 4; i++)
	{
		cout << yellow[i] << endl;
	}
}

void camera_in()
{
	VideoCapture cap(0);

	if (!cap.isOpened())									// return T/F
	{
		cerr << "Camera open failed!" << endl;
		return;
	}

	cout << "Frame width: " << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	cout << "Frame heigth: " << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;

	Mat frame, inversed;
	while (true)
	{
		cap >> frame;
		if (frame.empty())							// 프레임을 받아오지 못하는 경우
			break;

		inversed = ~frame;							// 반전

		imshow("frame", frame);
		imshow("inversed", inversed);

		if (waitKey(10) == 27)						// 대기시간 10ms, ASCII(ESC key)
			break;
	}

	destroyAllWindows();
}

void video_in()
{
	VideoCapture cap("stopwatch.avi");

	if (!cap.isOpened())
	{
		cerr << "Video open failed!" << endl;
		return;
	}

	cout << "Frame width: " << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	cout << "Frame heigth: " << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;
	cout << "Frame count: " << cvRound(cap.get(CAP_PROP_FRAME_COUNT)) << endl;						// 전체 동영상 프레임 수

	double fps = cap.get((CAP_PROP_FPS));
	cout << "FPS: " << fps << endl;				// 초당 프레임 수

	int delay = cvRound(1000 / fps);			// 프레임 사이 간격

	Mat frame, inversed;
	while (true)
	{
		cap >> frame;
		if (frame.empty())
			break;

		inversed = ~frame;

		imshow("frame", frame);
		imshow("inversed", inversed);

		if (waitKey(delay) == 27)
			break;
	}

	destroyAllWindows();
}

void camera_in_video_out()
{
	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cerr << "Camera open failed!" << endl;
		return;
	}

	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	double fps = 30;							// cap.get(CAP_PROP_FPS)이 오류가 나는 경우 따로 지정해주기 

	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
	int delay = cvRound(1000 / fps);

	VideoWriter outputVideo("output.avi", fourcc, fps, Size(w, h));

	if (!outputVideo.isOpened())
	{
		cout << "File open failed" << endl;
		return;
	}

	Mat frame, inversed;
	while (true)
	{
		cap >> frame;
		if (frame.empty())
			break;

		inversed = ~frame;
		outputVideo << inversed;					// 반전된 영상을 outputVideo에 저장

		imshow("frame", frame);
		imshow("inversed", inversed);

		if (waitKey(delay) == 27)
			break;
	}

	destroyAllWindows();
}

void drawLines()
{
	Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));						// 버퍼: 컬러 버퍼

	line(img, Point(50, 50), Point(200, 50), Scalar(0, 0, 255));				// 색만 다른 직선
	line(img, Point(50, 100), Point(200, 100), Scalar(255, 0, 255), 3);			// 굵기 3
	line(img, Point(50, 150), Point(200, 150), Scalar(255, 0, 0), 10);

	line(img, Point(250, 50), Point(350, 100), Scalar(0, 0, 255), 1, LINE_4);		// 대각선, 4방향 연결
	line(img, Point(250, 70), Point(350, 120), Scalar(255, 0, 255), 1, LINE_8);		// 8방향 연결
	line(img, Point(250, 90), Point(350, 140), Scalar(255, 0, 0), 1, LINE_AA);		// 안티에일리어싱(부드러운 선처럼 보이도록 테두리 그라데이션)

	arrowedLine(img, Point(50, 200), Point(150, 200), Scalar(0, 0, 255), 1);		// 화살표
	arrowedLine(img, Point(50, 250), Point(350, 250), Scalar(255, 0, 255), 1);		// 전체 직선 길이에 대한 화살표 길이의 비율 = 1
	arrowedLine(img, Point(50, 300), Point(350, 300), Scalar(255, 0, 255), 1);

	drawMarker(img, Point(50, 350), Scalar(0, 0, 255), MARKER_CROSS);				// 십자가
	drawMarker(img, Point(100, 350), Scalar(0, 0, 255), MARKER_TILTED_CROSS);		// 45도 회전 십자가
	drawMarker(img, Point(150, 350), Scalar(0, 0, 255), MARKER_STAR);				// cross + tilted cross
	drawMarker(img, Point(200, 350), Scalar(0, 0, 255), MARKER_DIAMOND);			// 마름모
	drawMarker(img, Point(250, 350), Scalar(0, 0, 255), MARKER_SQUARE);				// 정사각형
	drawMarker(img, Point(300, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_UP);		// 위로 뾰족한 삼각형
	drawMarker(img, Point(350, 350), Scalar(0, 0, 255), MARKER_TRIANGLE_DOWN);		// 아래로 뾰족한 삼각형

	imshow("img", img);
	waitKey(0);

	destroyAllWindows();
}

void drawPolys()
{
	Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));

	rectangle(img, Rect(50, 50, 100, 50), Scalar(0, 0, 255), 2);
	rectangle(img, Rect(50, 150, 100, 50), Scalar(0, 0, 128), -1);

	circle(img, Point(300, 120), 30, Scalar(255, 255, 0), -1, LINE_AA);
	circle(img, Point(300, 120), 60, Scalar(255, 0, 0), 3, LINE_AA);

	ellipse(img, Point(120, 300), Size(60, 30), 20, 0, 270, Scalar(255, 255, 0), -1, LINE_AA);		// 20도에서 시작해서 270도가 채워진 타원
	ellipse(img, Point(120, 300), Size(100, 50), 20, 0, 360, Scalar(0, 255, 0), 2, LINE_AA);

	vector<Point> pts;													// 다각형 꼭지점을 저장하는 배열
	pts.push_back(Point(250, 250)); pts.push_back(Point(300, 250));
	pts.push_back(Point(300, 300)); pts.push_back(Point(350, 300));
	pts.push_back(Point(350, 350)); pts.push_back(Point(250, 350));
	//polylines(img, pts, true, Scalar(255, 0, 255), 2);					// 다각형 색 채움 없음
	fillPoly(img, pts, Scalar(255, 0, 255), 0);							// 다각형 색 채움 있음

	imshow("img", img);
	waitKey(0);

	destroyAllWindows();
}

void drawText1()
{
	Mat img(500, 800, CV_8UC3, Scalar(255, 255, 255));

	putText(img, "FONT_HERSHEY_SIMPLEX", Point(20, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));									// 일반 크기의 산세리프 폰트
	putText(img, "FONT_HERSHEY_PLAIN", Point(20, 100), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));										// 작은 크기의 산세리프 폰트
	putText(img, "FONT_HERSHEY_DUPLEX", Point(20, 150), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));										// 일반 크기의 복잡한 산세리프 폰트
	putText(img, "FONT_HERSHEY_COMPLEX", Point(20, 200), FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0));									// 일반 크기의 세리프 폰트
	putText(img, "FONT_HERSHEY_TRIPLEX", Point(20, 250), FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 0));									// 일반 크기의 복잡한 세리프 폰트
	putText(img, "FONT_HERSHEY_COMPLEX_SMALL", Point(20, 300), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 0, 0));						// FONT_HERSHEY_COMPLEX보다 작은 폰트
	putText(img, "FONT_HERSHEY_SCRIPT_SIMPLEX", Point(20, 350), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(255, 0, 255));					// 필기체 스타일 폰트
	putText(img, "FONT_HERSHEY_SCRIPT_COMPLEX", Point(20, 400), FONT_HERSHEY_SCRIPT_COMPLEX, 1, Scalar(255, 0, 255));					// 필기체 스타일의 복잡한 폰트
	putText(img, "FONT_HERSHEY_COMPLEX | FONT_ITALIC", Point(20, 450), FONT_HERSHEY_COMPLEX | FONT_ITALIC, 1, Scalar(255, 0, 0));		// 이탤릭체를 위한 플래그

	imshow("img", img);
	waitKey(0);
}

void drawText2()
{
	Mat img(200, 640, CV_8UC3, Scalar(255, 255, 255));

	const String text = "Hello, OpenCV";
	int fontFace = FONT_HERSHEY_TRIPLEX;
	double fontScale = 2.0;
	int thickness = 1;

	Size sizeText = getTextSize(text, fontFace, fontScale, thickness, 0);
	Size sizeImg = img.size();

	Point org((sizeImg.width - sizeText.width) / 2, (sizeImg.height + sizeText.height) / 2);			// org = 좌측하단 좌표값
	putText(img, text, org, fontFace, fontScale, Scalar(255, 0, 0), thickness);
	rectangle(img, org, org + Point(sizeText.width, -sizeText.height), Scalar(255, 0, 0), 1);			// 대각점을 이용한 사각형

	imshow("img", img);
	waitKey(0);

	destroyAllWindows();
}

void keyboard()
{
	Mat img = imread("lenna.bmp");
	if (img.empty())
	{
		cerr << "Image load failed" << endl;
		return;						         // 에러가 있는 경우 반환값 -1
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
}

void trackbar()
{
	Mat img = Mat::zeros(600, 600, CV_8UC1);		// 빈 이미지 생성 (zeros → 검정)

	namedWindow("image");
	int value = 0;									// 참조할 정수형 변수
	createTrackbar("level", "image", &value, 16, on_level_change, (void*)&img);			// createTrackbar 함수

	imshow("image", img);
	waitKey(0);
	destroyAllWindows();
}
void on_level_change(int pos, void* userdata)
{
	Mat img = *(Mat*)userdata;		// 새로운 이미지 참조

	img.setTo(pos * 16);			// 트랙바의 위치(grayscale 레벨)에 따라 전체를 X16으로 맞춤
	imshow("image", img);
}

void writeData()
{
	String name = "Jane";							// 데이터 변수 선언
	int age = 10;
	Point pt1(100, 200);
	vector<int> scores = { 80,90,50 };
	Mat mat1 = (Mat_<float>(2, 2) << 1.0f, 1.5f, 2.0f, 3.2f);

	FileStorage fs(filename, FileStorage::WRITE);		// 쓰기 모드로 생성

	if (!fs.isOpened())					// 파일이 정상적으로 열렸는지 확인
	{
		cerr << "File open failed!" << endl;
		return;
	}

	fs << "name" << name;			// 데이터를 파일에 저장
	fs << "age" << age;
	fs << "point" << pt1;
	fs << "scores" << scores;
	fs << "data" << mat1;

	fs.release();					// 사용하던 파일을 닫고 메모리 버퍼 해제
}

void readData()
{
	String name;						// 읽어온 데이터 저장을 위한 선언
	int age;
	Point pt1;
	vector<int> scores;
	Mat mat1;

	FileStorage fs(filename, FileStorage::READ);

	if (!fs.isOpened())
	{
		cerr << "File open failed!" << endl;
		return;
	}

	/*FileNode fn = fs["name"];
	fn >> name;*/						// fn 변수는 임시로 사용한 다음에 필요가 없어지므로 굳이 변수로 선언해서 사용하지 않음
	fs["name"] >> name;					// 해당 항목 찾기
	fs["age"] >> age;
	fs["point"] >> pt1;
	fs["scores"] >> scores;
	fs["data"] >> mat1;

	fs.release();

	cout << "name: " << name << endl;
	cout << "age: " << age << endl;
	cout << "point: " << pt1 << endl;
	cout << "scores: " << Mat(scores).t() << endl;			// scores는 vector → Mat 객체로 형변환 & 전치행렬로 출력
	cout << "data:\n: " << mat1 << endl;
}

void mask_setTo()
{
	Mat src = imread("lenna.bmp", IMREAD_COLOR);
	Mat mask = imread("mask_smile.bmp", IMREAD_GRAYSCALE);			// 흑백영상, 원본 영상과 동일한 사이즈

	if (src.empty() || mask.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	src.setTo(Scalar(0, 255, 255), mask);					// mask 영상에서 픽셀 값이 0이 아닌 위치에서만 src 영상 픽셀을 노란색으로 설정함

	imshow("src", src);
	imshow("mask", mask);

	waitKey(0);
	destroyAllWindows();
}

void mask_copyTo()
{
	Mat src = imread("airplane.bmp", IMREAD_COLOR);
	Mat mask = imread("mask_plane.bmp", IMREAD_GRAYSCALE);
	Mat dst = imread("field.bmp", IMREAD_COLOR);

	if (src.empty() || mask.empty() || dst.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	src.copyTo(dst, mask);							// mask 영상에서 white 영역만 src 영상 픽셀 값을 dst로 복사

	imshow("dst", dst);

	waitKey(0);
	destroyAllWindows;
}

void time_inverse()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst(src.rows, src.cols, src.type());

	TickMeter tm;
	tm.start();																// 시간 측정 시작

	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			dst.at<uchar>(j, i) = ~src.at<uchar>(j, i);						// 좌표에 해당하는 픽셀값을 가져와서 반전, dst.at<uchar>(j, i) = 255 - src.at<uchar>(j, i);
		}
	}

	tm.stop();																	// 시간 측정 끝
	cout << "Image inverse took " << tm.getTimeMilli() << "ms." << endl;		// 밀리초 단위
}

void gray_sum()
{
	Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE);

	cout << "sum: " << (int)sum(img)[0] << endl;					// sum(img)가 Scalar로 리턴됨, Scalar[0]값 참조, int로 형변화
	cout << "mean: " << (int)mean(img)[0] << endl;
}

void color_sum()
{
	Mat img = imread("lenna.bmp", IMREAD_COLOR);

	cout << "b_sum: " << (int)sum(img)[0] << endl;					// (B,G,R)
	cout << "g_sum: " << (int)sum(img)[1] << endl;
	cout << "r_sum: " << (int)sum(img)[2] << endl;

	cout << "b_mean: " << (int)mean(img)[0] << endl;
	cout << "g_mean: " << (int)mean(img)[1] << endl;
	cout << "r_mean: " << (int)mean(img)[2] << endl;
}

void min_max()
{
	Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE);

	double minVal, maxVal;
	Point minPos, maxPos;
	minMaxLoc(img, &minVal, &maxVal, &minPos, &maxPos);						// 최소, 최대 grayscale 값과 좌표 구하는 함수

	cout << "minVal: " << minVal << " at " << minPos << endl;
	cout << "maxVal: " << maxVal << " at " << maxPos << endl;

	Mat img2 = imread("lenna.bmp", IMREAD_COLOR);

	drawMarker(img2, Point(508, 71), Scalar(0, 255, 255), MARKER_STAR);			// minVal 좌표에 yellow로 별표시
	drawMarker(img2, Point(116, 273), Scalar(0, 255, 0), MARKER_STAR);			// maxVal 좌표에 green으로 별표시

	imshow("img2", img2);

	waitKey(0);
	destroyAllWindows;
}

void norm1()
{
	Mat src = Mat_<float>({ 1,5 }, { -1.f, -0.5f, 0.f, 0.5f, 1.f });

	Mat dst;
	normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);					// 0~255 사이로 비율을 유지하면서 정규화

	cout << "src: " << src << endl;
	cout << "dst: " << dst << endl;
}

void norm2()
{
	Mat img = imread("lenna.bmp", IMREAD_GRAYSCALE);					// 이미지 불러오기 

	Mat dst;
	normalize(img, dst, 128, 255, NORM_MINMAX, CV_8UC1);				// 128~255 사이로 비율을 유지하면서 정규화
	//normalize(img, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	imshow("img", img);										// 원본 이미지
	imshow("img2", dst);									// 픽셀값 범위를 조절한 이미지

	waitKey(0);
	destroyAllWindows;
}

void round()
{
	cout << "cvRound(2.5): " << cvRound(2.5) << endl;			// 소수점 아래가 0.5인 경우에는 가장 가까운 짝수로 반올림
	cout << "cvRound(2.51): " << cvRound(2.51) << endl;			// 0.5보다 크면 올림
	cout << "cvRound(3.4999): " << cvRound(3.4999) << endl;		// 0.5보다 작으면 내림
	cout << "cvRound(3.5): " << cvRound(3.5) << endl;
	cout << "cvCeil(4.3): " << cvCeil(4.3) << endl;				// 올림 함수
	cout << "cvCeil(4.68): " << cvCeil(4.68) << endl;
	cout << "cvFloor(5.25): " << cvFloor(5.25) << endl;			// 내림 함수
	cout << "cvFloor(5.7): " << cvFloor(5.7) << endl;
}

void brightness1()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst = src + 100;
	//Mat dst; add(src, 100, dst);					// src에 100을 더한 값을 dst에 저장함

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void brightness2()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst = src - 100;
	//Mat dst; subtract(src, 100, dst);				// src에 100을 뺀 값을 dst에 저장함

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void brightness3()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst(src.rows, src.cols, src.type());			// 입력 영상과 크기, 타입이 같은 결과 영상 생성

	for (int j = 0; j < src.rows; j++)					// 입력 영상의 픽셀 값에 100을 더해서 결과 영상 값 설정
	{
		for (int i = 0; i < src.cols; i++)
		{
			dst.at<uchar>(j, i) = src.at<uchar>(j, i) + 100;
		}
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void brightness4()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst(src.rows, src.cols, src.type());

	for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			dst.at<uchar>(j, i) = saturate_cast<uchar>(src.at<uchar>(j, i) + 100);			// 포화 연산 수행한 후 결과 영상 픽셀 값으로 설정
		}
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void brightness5()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	namedWindow("dst");						// 트랙바 부착할 창 생성
	createTrackbar("Brightness", "dst", 0, 100, on_brightness, (void*)&src);		// 트랙바 부착, 콜백 함수 등록
	on_brightness(0, (void*)&src);			// 프로그램 실행 시 dst 창에 영상이 정상적으로 표시되도록 강제로 함수 호출, 빈 영상 방지

	waitKey();
	destroyAllWindows();
}
void on_brightness(int pos, void* userdata)
{
	Mat src = *(Mat*)userdata;
	Mat dst = src + pos;

	imshow("dst", dst);
}

void contrast1()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	float s = 2.f;									// 입력 영상의 모든 픽셀 값에 2.0을 곱해서 결과 영상 생성
	Mat dst = s * src;

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void contrast2()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	float alpha = 1.f;								// 효과적인 명암비 조절 수식, 입력 영상으로부터 명암비가 증가된 결과 영상을 생성
	Mat dst = src + (src - 128) * alpha;

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void histogram_cameraMan()
{
	Mat src = imread("camera.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat hist = calGrayHist(src);
	Mat hist_img = getGrayHistImage(hist);

	imshow("src", src);
	imshow("srcHist", hist_img);

	waitKey();
	destroyAllWindows();
}

Mat calGrayHist(const Mat& img)				// 외부에서 호출, img 파라미터
{
	CV_Assert(img.type() == CV_8UC1);		// 타입 체크

	Mat hist;
	int channels[] = { 0 };					// grayscale
	int dims = 1;
	const int histSize[] = { 256 };
	float graylevel[] = { 0,256 };
	const float* ranges[] = { graylevel };		// float** ranges

	calcHist(&img, 1, channels, noArray(), hist, dims, histSize, ranges);		// noArray() = mask

	return hist;
}
Mat getGrayHistImage(const Mat& hist)
{
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));

	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC1, Scalar(255));

	for (int i = 0; i < 256; i++)
	{
		line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0));
	}

	return imgHist;
}

Mat getBlueHistImage(const Mat& hist)
{
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));

	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC3, Scalar(255, 255, 255));				// 채널 3개 열어줌

	for (int i = 0; i < 256; i++)
	{
		line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(255, 0, 0));			// blue
	}

	return imgHist;
}
Mat getGreenHistImage(const Mat& hist)
{
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));

	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC3, Scalar(255, 255, 255));

	for (int i = 0; i < 256; i++)
	{
		line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0, 255, 0));
	}

	return imgHist;
}
Mat getRedHistImage(const Mat& hist)
{
	CV_Assert(hist.type() == CV_32FC1);
	CV_Assert(hist.size() == Size(1, 256));

	double histMax;
	minMaxLoc(hist, 0, &histMax);

	Mat imgHist(100, 256, CV_8UC3, Scalar(255, 255, 255));

	for (int i = 0; i < 256; i++)
	{
		line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0, 0, 255));
	}

	return imgHist;
}

void histogram_cameraMan2()
{
	Mat src = imread("lenna.bmp", IMREAD_COLOR);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<Mat> bgr_lenna;
	split(src, bgr_lenna);								// lenna 영상을 분리

	Mat hist_b = calGrayHist(bgr_lenna[0]);				// blue
	Mat hist_ib = getBlueHistImage(hist_b);

	Mat hist_g = calGrayHist(bgr_lenna[1]);				// green
	Mat hist_ig = getGreenHistImage(hist_g);

	Mat hist_r = calGrayHist(bgr_lenna[2]);				// red
	Mat hist_ir = getRedHistImage(hist_r);



	imshow("src", src);
	imshow("B_lenna", hist_ib);
	imshow("G_lenna", hist_ig);
	imshow("R_lenna", hist_ir);

	waitKey();
	destroyAllWindows();
}

void histogram_stretching()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	double gmin, gmax;
	minMaxLoc(src, &gmin, &gmax);					// grayscale 최대,최소값

	Mat dst = (src - gmin) * 255 / (gmax - gmin);		// 이미지 스트레칭

	imshow("src", src);
	imshow("srcHist", getGrayHistImage(calGrayHist(src)));		// 원본 히스토그램 그래프

	imshow("dst", dst);
	imshow("dstHist", getGrayHistImage(calGrayHist(dst)));		// 이미지 스트레칭 히스토그램 그래프

	waitKey();
	destroyAllWindows();
}

void histogram_equalization()
{
	Mat src = imread("hawkes.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst;
	equalizeHist(src, dst);

	imshow("src", src);
	imshow("srcHist", getGrayHistImage(calGrayHist(src)));		// 원본 히스토그램 그래프

	imshow("dst", dst);
	imshow("dstHist", getGrayHistImage(calGrayHist(dst)));		// 히스토그램 평활화 히스토그램 그래프

	waitKey();
	destroyAllWindows();
}

void arithmetic()
{
	Mat src1 = imread("dog.bmp", IMREAD_GRAYSCALE);
	Mat src2 = imread("camera.bmp", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src1", src1);
	imshow("src2", src2);

	Mat dst1, dst2, dst3;
	addWeighted(src1, 0.5, src2, 0.5, 0, dst1);					// 첫번째 입력 행렬, 가중치, 첫번째와 채널이 같은 두번째 입력 행렬, 가중치, 가중치에 추가적으로 더할 값, 출력 행렬
	addWeighted(src1, 1, src2, 1, 0, dst2);						// 가중치의 합이 1보다 크면 결과 영상이 입력 영상보다 밝아지고 포화 현상이 발생함
	addWeighted(src1, 0.3, src2, 0.3, 0, dst3);					// 가중치의 합이 1보다 작으면 입력 영상의 평균 밝기보다 어두워짐

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);

	waitKey();
	destroyAllWindows();
}

void logical()
{
	Mat src1 = imread("lenna256.bmp", IMREAD_GRAYSCALE);
	Mat src2 = imread("square.bmp", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src1", src1);
	imshow("src2", src2);

	Mat dst1, dst2, dst3, dst4;

	bitwise_and(src1, src2, dst1);					// 논리곱
	bitwise_or(src1, src2, dst2);					// 논리합
	bitwise_xor(src1, src2, dst3);					// 배타적 논리합
	bitwise_not(src1, dst4);						// src1 부정

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);
	imshow("dst4", dst4);

	waitKey();
	destroyAllWindows();
}

void arit_logi()
{
	VideoCapture cap(0);										// camera_in()

	if (!cap.isOpened())
	{
		cerr << "Camera open failed!" << endl;
		return;
	}

	Mat frame;
	Mat gray, model, dst;

	while (true)
	{
		cap >> frame;								// 실시간 영상 frame으로 받아오기
		if (frame.empty())
			break;

		cvtColor(frame, gray, COLOR_BGR2GRAY);		// COLOR_BGR2GRAY = 6
		imshow("frame", frame);
		imshow("gray", gray);

		int keycode = waitKey(10);										// event keyboard

		if (keycode == 's' || keycode == 'S')							// 모델 영상 취득
		{
			model = gray.clone();
			imshow("model", model);
		}

		else if (keycode == 'a' || keycode == 'A')
		{
			absdiff(model, gray, dst);							// 입력 영상에서 모델 영상의 차영상 취득
			imshow("dst", dst);
		}

		else if (keycode == 27 || keycode == 'q' || keycode == 'Q')
		{
			break;
		}
	}

	waitKey();
	destroyAllWindows();
}

void filter_embossing()
{
	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}
	float data[] = { -1,-1,0,-1,0,1,0,1,1 };					// 3X3 크기의 엠보싱 필터 마스크 행렬 생성
	Mat emboss(3, 3, CV_32FC1, data);

	Mat dst;
	filter2D(src, dst, -1, emboss, Point(-1, -1), 128);			// ddepth = -1, 필터링 커널 = emboss, 고정점 (-1,-1), 128을 추가로 더해줌, BOARDER_DEFAULT로 세팅

	imshow("src", src);
	imshow("dst", dst);

	waitKey(0);
	destroyAllWindows();
}

void blurring_mean()
{
	/*Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat dst1, dst2, dst3;

	blur(src, dst1, Size(3, 3));			// ksize = 3, 3X3 평균값 필터 마스크를 이용한 블러링
	blur(src, dst2, Size(5, 5));			// ksize = 5
	blur(src, dst3, Size(7, 7));			// ksize = 7

	String desc1 = format("Mean: %dX%d", 3, 3);				// 사용된 필터값을 문자열 형태로 출력
	putText(dst1, desc1, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String desc2 = format("Mean: %dX%d", 5, 5);
	putText(dst2, desc2, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String desc3 = format("Mean: %dX%d", 7, 7);
	putText(dst3, desc3, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);

	waitKey(0);
	destroyAllWindows();*/

	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat dst[3];

	for (int i = 0; i < 3; i++)
	{
		blur(src, dst[i], Size(2 * i + 3, 2 * i + 3));						  // ksize = (2*i) + 3

		String desc = format("Mean: %dX%d", 2 * i + 3, 2 * i + 3);
		putText(dst[i], desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	}

	imshow("dst1", dst[0]);
	imshow("dst2", dst[1]);
	imshow("dst3", dst[2]);

	waitKey(0);
	destroyAllWindows();
}

void blurring_gaussian()
{
	/*Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat dst1, dst2, dst3, dst4, dst5;

	GaussianBlur(src, dst1, Size(), (double)1);							// Size()를 쓰면 자동으로 mask 지정, 시그마값은 더블형
	GaussianBlur(src, dst2, Size(), (double)2);
	GaussianBlur(src, dst3, Size(), (double)3);
	GaussianBlur(src, dst4, Size(), (double)4);
	GaussianBlur(src, dst5, Size(), (double)5);

	String text1 = format("sigma = %d", 1);
	putText(dst1, text1, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String text2 = format("sigma = %d", 2);
	putText(dst2, text2, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String text3 = format("sigma = %d", 3);
	putText(dst3, text3, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String text4 = format("sigma = %d", 4);
	putText(dst4, text4, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String text5 = format("sigma = %d", 5);
	putText(dst5, text5, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);
	imshow("dst4", dst4);
	imshow("dst5", dst5);

	waitKey(0);
	destroyAllWindows();*/

	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat dst[5];

	for (int sigma = 1; sigma <= 5; sigma++)
	{
		GaussianBlur(src, dst[sigma - 1], Size(), (double)sigma);

		String text = format("sigma = %d", sigma);
		putText(dst[sigma - 1], text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	}

	imshow("dst1", dst[0]);
	imshow("dst2", dst[1]);
	imshow("dst3", dst[2]);
	imshow("dst4", dst[3]);
	imshow("dst5", dst[4]);

	waitKey(0);
	destroyAllWindows();
}

void unsharp_mask()
{
	/*Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat blurred1, blurred2, blurred3, blurred4, blurred5;
	GaussianBlur(src, blurred1, Size(), 1);				// sigma = 1
	GaussianBlur(src, blurred2, Size(), 2);				// sigma = 2
	GaussianBlur(src, blurred3, Size(), 3);				// sigma = 3
	GaussianBlur(src, blurred4, Size(), 4);				// sigma = 4
	GaussianBlur(src, blurred5, Size(), 5);				// sigma = 5

	float alpha = 1.f;
	Mat dst1 = (1 + alpha) * src - alpha * blurred1;		// 언샤프 마스크 필터링
	Mat dst2 = (1 + alpha) * src - alpha * blurred2;
	Mat dst3 = (1 + alpha) * src - alpha * blurred3;
	Mat dst4 = (1 + alpha) * src - alpha * blurred4;
	Mat dst5 = (1 + alpha) * src - alpha * blurred5;

	String desc1 = format("sigma: %d", 1);
	putText(dst1, desc1, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String desc2 = format("sigma: %d", 2);
	putText(dst2, desc2, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String desc3 = format("sigma: %d", 3);
	putText(dst3, desc3, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String desc4 = format("sigma: %d", 4);
	putText(dst4, desc4, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String desc5 = format("sigma: %d", 5);
	putText(dst5, desc5, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);
	imshow("dst4", dst4);
	imshow("dst5", dst5);

	waitKey(0);
	destroyAllWindows();*/

	Mat src = imread("rose.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat dst[5];

	for (int sigma = 1; sigma <= 5; sigma++)
	{
		Mat blurred;
		GaussianBlur(src, blurred, Size(), sigma);

		float alpha = 1.f;
		dst[sigma - 1] = (1 + alpha) * src - alpha * blurred;

		String desc = format("sigma: %d", sigma);
		putText(dst[sigma - 1], desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	}

	imshow("dst1", dst[0]);
	imshow("dst2", dst[1]);
	imshow("dst3", dst[2]);
	imshow("dst4", dst[3]);
	imshow("dst5", dst[4]);

	waitKey(0);
	destroyAllWindows();
}
	
void noise_gaussian()
{
	/*Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat noise1(src.size(), CV_32SC1);				// 32bit short형 빈 객체 생성
	randn(noise1, 0, 10);							// 표준 편차 stddev = 10
	Mat noise2(src.size(), CV_32SC1);
	randn(noise2, 0, 20);
	Mat noise3(src.size(), CV_32SC1);
	randn(noise3, 0, 30);

	Mat dst1, dst2, dst3;
	add(src, noise1, dst1, Mat(), CV_8U);			// Mat() = mask, 영상 깊이 = CV_8U
	add(src, noise2, dst2, Mat(), CV_8U);
	add(src, noise3, dst3, Mat(), CV_8U);

	String desc1 = format("stddev = %d", 10);
	putText(dst1, desc1, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String desc2 = format("stddev = %d", 20);
	putText(dst2, desc2, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
	String desc3 = format("stddev = %d", 30);
	putText(dst3, desc3, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);

	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);

	waitKey(0);
	destroyAllWindows();*/

	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat dst;

	for (int i = 1; i <= 3; i++)
	{
		Mat noise(src.size(), CV_32SC1);
		randn(noise, 0, i*10);								// stddev = i * 10

		add(src, noise, dst, Mat(), CV_8U);

		String desc = format("stddev = %d", i*10);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
		
		imshow(format("dst%d", i), dst);					// dst1, dst2, dst3 순차적 출력
	}

	waitKey(0);
	destroyAllWindows();
}
	
void filter_bilateral()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat noise(src.size(), CV_32SC1);						// 노이즈 추가
	randn(noise, 0, 5);
	add(src, noise, src, Mat(), CV_8U);

	TickMeter tm1;
	tm1.start();

	Mat dst1;
	GaussianBlur(src, dst1, Size(), 5);						// 가우시안 블러, 표준편차 = 5

	tm1.stop();
	cout << "gaussianblur took " << tm1.getTimeMilli() << "ms" << endl;					// 가우시안 블러 시간 측정

	TickMeter tm2;
	tm2.start();

	Mat dst2;
	bilateralFilter(src, dst2, -1, 10, 5);					// 양방향 필터, -10 = sigmacolor(색공간 표준편차), 좌표공간 표준편차 = 5

	tm2.stop();
	cout << "bilateralfilter took " << tm2.getTimeMilli() << "ms" << endl;				// 양방향 필터 시간 측정

	imshow("src", src);
	imshow("gaussianblur", dst1);
	imshow("bilateralfilter", dst2);

	waitKey(0);
	destroyAllWindows();
}

void filter_median()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	int num = (int)(src.total() * 0.1);					// 전체 픽셀 X 10%

	for (int i = 0; i < num; i++)						// 소금&후추 잡음 추가
	{
		int x = rand() % src.cols;
		int y = rand() % src.rows;
		src.at<uchar>(y, x) = (i % 2) * 255;			// 짝수는 0, 홀수는 255
	}

	Mat dst1;
	GaussianBlur(src, dst1, Size(), 1);					// 가우시안 블러, 표준편차 = 1

	Mat dst2;
	medianBlur(src, dst2, 3);							// 필터 크기 = 3

	imshow("src", src);
	imshow("gaussianblur", dst1);
	imshow("medianblur", dst2);

	waitKey();
	destroyAllWindows();
}

void affine_transform()
{
	Mat src = imread("tekapo.bmp");						// 3 컬러 영상 저장

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Point2f srcPts[3], dstPts[3];						// 세 점 좌표를 저장할 배열 선언

	srcPts[0] = Point2f(0, 0);							// srcPts 배열에 입력 영상의 좌측상단, 우측상단, 우측하단 좌표 저장
	srcPts[1] = Point2f(src.cols - 1, 0);
	srcPts[2] = Point2f(src.cols - 1, src.rows - 1);

	dstPts[0] = Point2f(50, 50);						// dstPts 배열에 이동할 좌표 지정
	dstPts[1] = Point2f(src.cols - 100, 100);
	dstPts[2] = Point2f(src.cols - 50, src.rows - 50);

	Mat M = getAffineTransform(srcPts, dstPts);			// 2x3 어파인 변환 행렬을 M에 저장

	Mat dst;
	warpAffine(src, dst, M, Size());					// size를 지정해서 영상 2개의 크기가 같아지도록 설정

	imshow("src", src);
	imshow("dst", dst);

	vector<Point2f> src_p = { Point2f(100,20), Point2f(200,50) };
	vector<Point2f> dst_p;

	transform(src_p, dst_p, M);							// 어파인 변환 행렬을 통해서 특정 좌표값이 어떻게 변화하는지 알아내려는 함수
	cout << "dst_point" << endl;
	cout << dst_p << endl;

	waitKey();
	destroyAllWindows();
}

void affine_translation()
{
	Mat src = imread("tekapo.bmp");

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat M = Mat_<double>({ 2,3 }, { 1, 0, 150, 0, 1, 100 });			// 이동변환 어파인 변환 행렬 → a = 150, b = 100

	Mat dst1;
	warpAffine(src, dst1, M, Size());									// boardervalue를 설정해주지 않으면 검정색이 기본값

	Mat dst2;
	warpAffine(src, dst2, M, Size(790, 580));							// 이동한 그림 전체가 모두 나오도록 640X480 사이즈 변환 → dsize(결과 영상 크기) 파라미터 이용

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	destroyAllWindows();
}

void affine_shear()
{
	Mat src = imread("tekapo.bmp");

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	double mx = 0.3;															// 가로 방향으로 밀림 정도 = 0.3
	Mat M1 = Mat_<double>({ 2,3 }, { 1, mx, 0, 0, 1, 0 });						// 전단 변환 행렬 M1 생성

	double my = 0.3;															// 세로 방향으로 밀림 정도 = 0.3
	Mat M2 = Mat_<double>({ 2,3 }, { 1, 0, 0, my, 1, 0 });						// 전단 변환 행렬 M2 생성

	Mat dst1;
	warpAffine(src, dst1, M1, Size(cvRound(src.cols + src.rows * mx), src.rows));				// 결과 영상 가로 크기 → x' = x + mx*y

	Mat dst2;
	warpAffine(src, dst2, M2, Size(src.cols, cvRound(src.cols * my + src.rows)));				// 결과 영상 세로 크기 → y' = my*x + y

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	destroyAllWindows();
}

void affine_scale()
{
	Mat src = imread("rose.bmp");

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst1, dst2, dst3, dst4;
	resize(src, dst1, Size(), 4, 4, INTER_NEAREST);						// 480X320을 4배해서 1920X1280으로 결정, 최근방 이웃 보간법
	resize(src, dst2, Size(1920, 1280));								// 1920X1280으로 확대, 기본값인 양선형 보간법
	resize(src, dst3, Size(1920, 1280), 0, 0, INTER_CUBIC);				// 3차 회선 보간법
	resize(src, dst4, Size(1920, 1280),0,0,INTER_LANCZOS4);				// 란초스 보간법


	imshow("src", src);
	imshow("dst1", dst1(Rect(400, 500, 400, 400)));						// 확대 변환 영상을 (400,500)좌표부터 400X400 크기의 부분 영상을 출력
	imshow("dst2", dst2(Rect(400, 500, 400, 400)));
	imshow("dst3", dst3(Rect(400, 500, 400, 400)));
	imshow("dst4", dst4(Rect(400, 500, 400, 400)));

	waitKey();
	destroyAllWindows();
}

void affine_rotation1()
{
	Mat src = imread("tekapo.bmp");

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Point2f cp(src.cols / 2.f, src.rows / 2.f);						// cp = 영상의 중심 좌표
	Mat M = getRotationMatrix2D(cp, 20, 1);							// M = cp를 기준으로 반시계 방향으로 20도 회전하는 변환 행렬, 크기 변환을 하지 않으려면 scale을 1로 지정

	Mat dst;
	warpAffine(src, dst, M, Size());

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void affine_rotation2()
{
	Mat src = imread("tekapo.bmp");

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat M1 = Mat_<double>({ 2,3 }, { 1, 0, 100, 0, 1, 100 });			// affine_translation(), 이동변환 어파인 변환 행렬 → a = 100, b = 100
	Mat dst1;
	warpAffine(src, dst1, M1, Size(840, 680));							// 640X480 사이즈 + 200

	Point2f cp1(dst1.cols / 2.f, dst1.rows / 2.f);					// cp1 = 영상의 중심 좌표
	Mat M2 = getRotationMatrix2D(cp1, 20, 1);						// M2 = cp1 기준으로 반시계 방향으로 20도 회전하는 변환 행렬, 크기 변환을 하지 않으려면 scale을 1로 지정

	Mat dst2;
	warpAffine(dst1, dst2, M2, Size());

	vector<Point2f> src_p = { Point2f(0, 0), Point2f(src.cols - 1, 0), Point2f(src.cols - 1, src.rows - 1), Point2f(0, src.rows - 1) };
	vector<Point2f> dst1_p;
	vector<Point2f> dst2_p;

	transform(src_p, dst1_p, M1);
	cout << "src_point" << endl;
	cout << src_p << endl;
	cout << "dst1_point" << endl;
	cout << dst1_p << endl;

	transform(dst1_p, dst2_p, M2);
	cout << "dst2_point" << endl;
	cout << dst2_p << endl;

	Mat dst3;
	warpAffine(dst1, dst3, M2 ,Size(805, 675));					// 좌표에서 얻은 크기로 화면 출력

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
	imshow("dst3", dst3);

	waitKey();
	destroyAllWindows();
}

void affine_flip()
{
	Mat src = imread("eastsea.bmp");

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	Mat dst;

	int flipCode[] = { 1,0,-1 };								// flip() 함수에 전달할 flipCode 3개를 정수형 배열에 저장

	Mat d;
	rotate(src, d, ROTATE_180);									// 180도 회전 = 상하 및 좌우 대칭과 동일

	String desc1 = format("rotate: %d", 180);
	putText(d, desc1, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 1, LINE_AA);
	imshow("d", d);

	for (int i = 0; i < 3; i++)
	{
		flip(src, dst, flipCode[i]);							// 배열에 저장된 정수 값으로 대칭 변환 수행

		String desc = format("flipCode: %d", flipCode[i]);
		putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 1, LINE_AA);

		imshow(format("dst%d", i+1), dst);						// dst1, dst2, dst3 순차적 출력
		//imshow(desc, dst);
	}

	waitKey();
	destroyAllWindows();
}

void sobel_edge()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);							// 8 bit unsigned

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dx, dy;
	Sobel(src, dx, CV_32FC1, 1, 0);								// - 값을 가질 수도 있기 때문에 32 bit float, x 방향으로 편미분
	Sobel(src, dy, CV_32FC1, 0, 1);								// 32 bit float, y 방향으로 편미분

	Mat fmag, mag;
	magnitude(dx, dy, fmag);									// dx,dy 행렬로부터 그래디언트 크기를 계산해서 fmag에 저장
	fmag.convertTo(mag, CV_8UC1);								// fmag를 그레이스케일 형식으로 변환해서 mag에 저장 → 영상으로 출력하기 위해 다시 8 bit unsigned

	Mat edge = mag > 150;										// threshold(임계값) 적용, edge 행렬의 원소값은 mag 행렬 원소값이 150보다 작으면 0으로 설정하고 크면 255로 설정

	imshow("src", src);
	imshow("mag", mag);
	imshow("edge", edge);

	waitKey();
	destroyAllWindows();
}

void canny_edge()
{
	Mat src = imread("lenna.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst1, dst2;
	Canny(src, dst1, 50, 100);							// 낮은 임계값: 50, 높은 임계값: 100으로 설정한 캐니 에지 검출을 수행
	Canny(src, dst2, 50, 150);									// 낮은 임계값: 50, 높은 임계값: 150으로 설정한 캐니 에지 검출을 수행

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	destroyAllWindows();
}

void hough_lines()
{
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat edge;																		// 원본 영상이 아닌 에지 영상
	Canny(src, edge, 50, 150);

	vector<Vec2f> lines;
	HoughLines(edge, lines, 1, CV_PI / 180, 250);									// 각도로 계산해야 하기 때문에 180을 pi로 나눔, 축적배열 threshold = 250

	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);											// 라인을 그어주기 위해 컬러영상으로 변환

	for (size_t i = 0; i < lines.size(); i++)										// 라인에 대한 직선을 긋기 위한 연산
	{
		float r = lines[i][0], t = lines[i][1];										// r = ρ, t = θ
		double cos_t = cos(t), sin_t = sin(t);
		double x0 = r * cos_t, y0 = r * sin_t;
		double alpha = 1000;														// 직선의 시작점과 끝점을 모르기 때문에 크게 지정

		Point pt1(cvRound(x0 + alpha * (-sin_t)), cvRound(y0 + alpha * cos_t));			// sinθ = xα / α, cosθ = yα / α
		Point pt2(cvRound(x0 - alpha * (-sin_t)), cvRound(y0 - alpha * cos_t));

		line(dst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey(0);
	destroyAllWindows();
}

void hough_line_segments()
{
	Mat src = imread("building.jpg", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat edge;																		// 원본 영상이 아닌 에지 영상
	Canny(src, edge, 50, 150);

	vector<Vec4i> lines;															// int형 4개 백터
	HoughLinesP(edge, lines, 1, CV_PI / 180, 160, 50, 5);							// 픽셀 단위 = 1, 각도 해상도 = 1도, 임계점 = 160, 검출할 선분 최소 길이 = 50, 끊어짐 허용 간격 = 5

	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);

	for (Vec4i l : lines)															// 라인 개수만큼 for문 돌리기 → line을 대신한 변수 = l
	{
		line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey(0);
	destroyAllWindows();
}

void hough_circles1()
{
	Mat src = imread("coin.jpg", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat blurred;											// 노이즈 제거 → 가우시안 사용 가능함, 생략도 가능함
	blur(src, blurred, Size(3, 3));

	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 30, 200, 50);				// 입력영상 1 : 축적 배열 1 = 크기 비율 1, 인접한 원 중심의 최소 거리 = 30 픽셀, canny 에지 검출기의 임계값 = 200, 축적배열에서 원 중심찾기를 위한 교차점 임계값 = 50

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (Vec3f c : circles)
	{
		Point center(cvRound(c[0]), cvRound(c[1]));
		int radius = cvRound(c[2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey(0);
	destroyAllWindows();
}

void hough_circles2()
{
	VideoCapture cap(0);										// camera_in()
	cap.set(CAP_PROP_FRAME_WIDTH, 1920);						// 해상도 설정
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080);

	if (!cap.isOpened())
	{
		cerr << "Camera open failed!" << endl;
		return;
	}

	Mat frame, dst;
	Mat coin1, coin2;
	int won = 0;
	int sum = 0;

	while (true)
	{
		cap >> frame;								// 실시간 영상 frame으로 받아오기
		if (frame.empty())
			break;

		imshow("frame", frame);
		
		int keycode = waitKey(10);										// event keyboard

		if (keycode == 's' || keycode == 'S')							// 영상 취득
		{
			coin1 = frame.clone();
			imshow("coin", coin1);

			vector<int> params;											// 취득한 영상 저장
			params.push_back(IMWRITE_JPEG_QUALITY);
			params.push_back(100);
			imwrite("coin1.jpg", coin1, params);
		}

		else if (keycode == 'c' || keycode == 'C')
		{
			Mat coin2 = imread("coin1.jpg", IMREAD_GRAYSCALE);			// 영상 불러오기

			if (coin2.empty())
			{
				cerr << "Image load failed!" << endl;
				return;
			}

			Mat blurred;											// 노이즈 제거
			blur(coin2, blurred, Size(3, 3));

			vector<Vec3f> circles;													// 허프 변환 원 검출
			HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 30, 200, 50);

			cvtColor(coin2, dst, COLOR_GRAY2BGR);

			for (Vec3f c : circles)
			{
				Point center(cvRound(c[0]), cvRound(c[1]));
				int radius = cvRound(c[2]);

				if (radius <= 100)															// 10원 yellow
				{
					circle(dst, center, radius, Scalar(255, 255, 0), 2, LINE_AA);
					won += 10;
					String desc2 = format("Radius: %d", radius);
					putText(dst, desc2, center, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 0), 1, LINE_AA);
				}

				else if (radius > 100 && radius <= 120)										// 50원 green
				{
					circle(dst, center, radius, Scalar(0, 255, 0), 2, LINE_AA);
					won += 50;
					String desc3 = format("Radius: %d", radius);
					putText(dst, desc3, center, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 1, LINE_AA);
				}

				else if (radius > 120 && radius <= 130)									   // 100원 blue
				{
					circle(dst, center, radius, Scalar(255, 0, 0), 2, LINE_AA);
					won += 100;
					String desc4 = format("Radius: %d", radius);
					putText(dst, desc4, center, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 1, LINE_AA);
				}

				else if (radius > 130)													// 500원 red
				{
					circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
					won += 500;
					String desc5 = format("Radius: %d", radius);
					putText(dst, desc5, center, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
				}
				
			}

			imshow("dst", dst);

		}

		else if (keycode == 't' || keycode == 'T')							// 전체 금액 화면에 출력
		{
			sum = won;

			String desc1 = format("Total: %dwon", sum);
			putText(dst, desc1, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 1, LINE_AA);
			imshow("dst", dst);
		}

		else if (keycode == 27 || keycode == 'q' || keycode == 'Q')
		{
			break;
		}
	}

	waitKey();
	destroyAllWindows();
}

void color_inverse()
{
	Mat src = imread("rose.bmp", IMREAD_COLOR);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst(src.rows, src.cols, src.type());				// dst 빈 영상으로 생성

	// for문 사용
	for (int j = 0; j < src.rows; j++)						// src, dst 영상의 (i,j) 좌표 픽셀 값을 각각 p1, p2 변수에 참조로 받아옴
	{
		for (int i = 0; i < src.cols; i++)
		{
			Vec3b& p1 = src.at<Vec3b>(j, i);				// 참조
			Vec3b& p2 = dst.at<Vec3b>(j, i);

			p2[0] = 255 - p1[0];						// B
			p2[1] = 255 - p1[1];						// G
			p2[2] = 255 - p1[2];						// R
		}
	}

	// for문 간단히
	/*for (int j = 0; j < src.rows; j++)
	{
		for (int i = 0; i < src.cols; i++)
		{
			dst.at<Vec3b>(j, i) = Vec3b(255, 255, 255) - src.at<Vec3b>(j, i);				// Vec3b 클래스에서 지원하는 - 연산자 재정의를 이용하여 한꺼번에 반전을 수행함
		}
	}*/

	// 간단 & 실용적
	//Mat dst = Scalar(255, 255, 255) - src;

	// inverse ~
	//Mat dst = ~src;

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void color_split()
{
	Mat src = imread("candies.png", IMREAD_COLOR);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<Mat> bgr_planes;						// src 영상을 분할해서 백터에 저장
	split(src, bgr_planes);

	Mat Y_plane;
	addWeighted(bgr_planes[1], 0.7, bgr_planes[2], 0.3, 0.0, Y_plane);			// Y_plane = 0.7 * bgr_planes[1] + 0.3 * bgr_planes[2];

	imshow("src", src);
	imshow("B_plane", bgr_planes[0]);			// B
	imshow("G_plane", bgr_planes[1]);			// G
	imshow("R_plane", bgr_planes[2]);			// R
	imshow("Y_plane", Y_plane);					// yellow
	
	Mat tt = Y_plane > 200;					// Y_plane 이진화
	imshow("tt", tt);

	waitKey();
	destroyAllWindows();
}

void color_equal()
{
	Mat src = imread("pepper.bmp", IMREAD_COLOR);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat src_ycrcb;
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);					// BGR 색 공간의 src 영상을 YCrCb 색 공간으로 변경하여 src_ycrcb에 저장 

	vector<Mat> ycrcb_planes;
	split(src_ycrcb, ycrcb_planes);								// src_ycrcb 영상 채널 분리

	equalizeHist(ycrcb_planes[0], ycrcb_planes[0]);				// Y 성분에 해당하는 ycrcb_planes[0] 영상에 대해서만 히스토그램 평활화

	Mat dst_ycrcb;
	merge(ycrcb_planes, dst_ycrcb);								// ycrcb_planes 백터에 들어있는 세 영상을 합쳐서 dst_ycrcb 생성

	Mat dst;
	cvtColor(dst_ycrcb, dst, COLOR_YCrCb2BGR);					// dst_ycrcb 영상의 색 공간을 BGR 색 공간으로 변환해서 저장

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void color_hue()
{
	Mat candy = imread("candies.png", IMREAD_COLOR);

	if (candy.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	cvtColor(candy, candy_hsv, COLOR_BGR2HSV);						// src 영상을 HSV 색 공간으로 변환해서 저장

	imshow("src", candy);

	namedWindow("mask");
	createTrackbar("Lower Hue", "mask", &lower_hue, 179, on_hue_changed);				// 색상 최대값 179 → 360 / 2 = 180, 0 ~ 179
	createTrackbar("Upper Hue", "mask", &upper_hue, 179, on_hue_changed);
	on_hue_changed(0, 0);																// 영상 정상 출력을 위한 콜백함수 강제호출

	waitKey();
	destroyAllWindows();
}
void on_hue_changed(int, void*)
{
	Scalar lowerb(lower_hue, 100, 0);						// s(채도) = 100 ~ 255, v(밝기) = 0 ~ 255, v의 범위를 0~255로 지정해서 범위 제약이 없음 
	Scalar upperb(upper_hue, 255, 255);
	inRange(candy_hsv, lowerb, upperb, candy_mask);					// src_hsv 영상에서 HSV 색 성분 범위가 lowerb에서 upperb 사이인 위치의 픽셀만 흰색으로 설정한 mask 영상 생성

	imshow("mask", candy_mask);
}

void color_backproj1()
{
	// calculate CrCb histogram from a reference image, 기준 영상으로부터 피부색 영역에 대한 히스토그램을 추출

	Mat ref, ref_ycrcb, mask;

	ref = imread("ref.png", IMREAD_COLOR);
	mask = imread("mask.bmp", IMREAD_GRAYSCALE);
	cvtColor(ref, ref_ycrcb, COLOR_BGR2YCrCb);								// ref를 YCrCb 색 공간으로 변환

	Mat hist;																// 피부색 영역의 CrCb 2차원 히스토그램을 계산해서 저장
	int channels[] = { 1,2 };												// CrCb 채널
	//int cr_bins = 128;														// 256으로 해도 되지만 128이 더 깔끔
	//int cb_bins = 128;
	int cr_bins = 256;														// 256으로 해도 되지만 128이 더 깔끔
	int cb_bins = 256;
	int histSize[] = { cr_bins,cb_bins };
	float cr_range[] = { 0,256 };
	float cb_range[] = { 0,256 };
	const float* ranges[] = { cr_range, cb_range };

	calcHist(&ref_ycrcb, 1, channels, mask, hist, 2, histSize, ranges);


	// apply histogram backprojection to an input image, 히스토그램 정보를 이용하여 입력 영상에서 피부색 영역을 검출함

	Mat src, src_ycrcb;
	src = imread("kids.png", IMREAD_COLOR);
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

	Mat backproj;
	calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);

	imshow("src", src);
	imshow("backproj", backproj);

	waitKey();
	destroyAllWindows();
}

void color_backproj2()
{
	ref_img = imread("ref.png", IMREAD_COLOR);								// 위에 전역변수로 Mat을 써줬기 때문에 또 쓰면 영상 출력이 안됨
	ref_mask = Mat(ref_img.size(), CV_8UC1);								// mask = unsigned 8 bit char
	ref_mask.setTo(0);

	if (ref_img.empty() || ref_mask.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	cvtColor(ref_img, ref_ycrcb, COLOR_BGR2YCrCb);

	namedWindow("ref_img");
	setMouseCallback("ref_img", on_mouse);									// 콜백 함수 호출

	imshow("ref_img", ref_img);

	waitKey(0);
	destroyAllWindows();
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
}

void adaptive()
{
	Mat src = imread("sudoku.jpg", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	namedWindow("dst");
	createTrackbar("Block Size", "dst", 0, 200, on_trackbar, (void*)&src);						// 트랙바 생성
	setTrackbarPos("Block Size", "dst", 11);													// 트랙바 초기 위치 = 11

	waitKey();
	destroyAllWindows();
}
void on_trackbar(int pos, void* userdata)
{
	Mat src = *(Mat*)userdata;

	int bsize = pos;
	if (bsize % 2 == 0) bsize--;																// bsize = 짝수 → 홀수로 만들기
	if (bsize < 3) bsize = 3;																	// bsize 값이 3보다 작으면 3으로 설정

	Mat dst;
	adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, bsize, 5);			// 가우시안 가중 평균, 블록 평균에서 5를 뺀 값을 임계값으로 사용하는 적응형 이진화 수행

	imshow("dst", dst);
}

void struct_element()
{
	int size_x = 5;
	int size_y = 5;

	cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size_x, size_y), cv::Point(int(size_x / 2), int(size_y / 2)));
	cout << "RECT" << endl;
	cout << kernel1 << endl;
	cout << endl;

	cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(size_x, size_y), cv::Point(int(size_x / 2), int(size_y / 2)));
	cout << "CROSS" << endl;
	cout << kernel2 << endl;
	cout << endl;
	
	cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size_x, size_y), cv::Point(int(size_x / 2), int(size_y / 2)));
	cout << "ELLIPSE" << endl;
	cout << kernel3 << endl;
	cout << endl;
}

void erode_dilate()
{
	Mat src = imread("milkdrop.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);						// 오츠 알고리즘으로 src 자동 이진화 수행

	Mat dst1, dst2, dst3, dst4, dst5, dst6;
	erode(bin, dst1, Mat());														// Mat() = 3X3 사각형 구조 요소
	dilate(bin, dst2, Mat());
	erode(bin, dst3, Mat(), Point(-1, -1), 2);										// 2 = iterations, 반복횟수, 2번 침식
	dilate(bin, dst4, Mat(), Point(-1, -1), 2);										// 2번 팽창
	erode(bin, dst5, Mat(), Point(-1, -1), 3);
	dilate(bin, dst6, Mat(), Point(-1, -1), 3);

	imshow("src", src);
	imshow("bin", bin);
	imshow("erode1", dst1);
	imshow("dilate1", dst2);
	imshow("erode2", dst3);
	imshow("dilate2", dst4);
	imshow("erode3", dst5);
	imshow("dilate3", dst6);
	

	waitKey();
	destroyAllWindows();
}

void open_close()
{
	Mat src = imread("milkdrop.bmp", IMREAD_GRAYSCALE);
	
	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat dst1, dst2, dst3, dst4, dst5;
	morphologyEx(bin, dst1, MORPH_OPEN, Mat());										// 열기(침식 다음 팽창) 모폴로지 연산
	morphologyEx(bin, dst2, MORPH_CLOSE, Mat());									// 닫기(팽창 다음 침식) 모폴로지 연산
	morphologyEx(bin, dst3, MORPH_ERODE, Mat());									// 침식 모폴로지 연산
	morphologyEx(bin, dst4, MORPH_DILATE, Mat());									// 팽창 모폴로지 연산
	morphologyEx(bin, dst5, MORPH_GRADIENT, Mat());									// dilate - erode → 객체의 외곽선이 추출됨


	imshow("src", src);
	imshow("bin", bin);
	imshow("open", dst1);
	imshow("close", dst2);
	imshow("erode", dst3);
	imshow("dilate", dst4);
	imshow("gradient", dst5);

	waitKey();
	destroyAllWindows();
}

void labeling_basic1()													// index number 그룹화
{
	uchar data[] = {													// 8X8 이진화된 이미지
		0, 0, 1, 1, 0, 0, 0, 0,
		1, 1, 1, 1, 0, 0, 1, 0,
		1, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 1, 0,
		0, 0, 0, 1, 1, 1, 1, 0,
		0, 0, 0, 1, 0, 0, 1, 0,
		0, 0, 1, 1, 1, 1, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	};

	Mat src = Mat(8, 8, CV_8UC1, data) * 255;							// uchar 자료형 배열 data를 픽셀 데이터로 사용하는 임시 Mat 객체 생성, 모든 원소에 255 곱한 결과 행렬을 src에 저장

	Mat labels;
	int cnt = connectedComponents(src, labels);							// 레이블 맵을 labels 행렬에 저장

	cout << "src:\n" << src << endl;
	cout << endl;
	cout << "labels:\n" << labels << endl;
	cout << endl;
	cout << "number of labels: " << cnt << endl;
}

void labeling_basic2()
{
	Mat src = imread("milkdrop.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);					// 영상 이진화

	Mat labels;
	int cnt = connectedComponents(bin, labels);									// // 레이블 맵을 labels 행렬에 저장

	cout << "number of labels: " << cnt << endl;
}

void labeling_stats()
{
	Mat src = imread("keyboard.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat bin;
	threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat labels, stats, centroids;
	int cnt = connectedComponentsWithStats(bin, labels, stats, centroids);						// 전체 레이블 개수로 return

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);															// grayscale → color

	for (int i = 1; i < cnt; i++)
	{
		int* p = stats.ptr<int>(i);
		double* c = centroids.ptr<double>(i);													// 무게 중심

		if (p[4] < 20)																			// 면적이 20보다 작은 경우 빨간 동그라미
		{
			circle(dst, Point(p[0], p[1]), 2, Scalar(0, 0, 255), -1);
			circle(dst, Point(c[0], c[1]), 1, Scalar(0, 255, 0), -1);
		}

		else if (p[4] >= 20)
		{
			rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 255), 2);				// p[0] = x, p[1] = y, p[2] = width, p[3] = height
			String desc1 = format("%d", i);
			putText(dst, desc1, Point(p[0], p[1]), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 1, LINE_AA);
			circle(dst, Point(c[0], c[1]), 2, Scalar(0, 255, 0), -1);
		}
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void contours_basic()
{
	Mat src = imread("contours.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<vector<Point>> contours;
	findContours(src, contours, RETR_LIST, CHAIN_APPROX_NONE);						// 모든 외곽선을 검출, 계층 정보는 추출하지 않음

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (int i = 0; i < contours.size(); i++)										// 외곽선 개수만큼 반복문 수행, 외곽선을 임의의 색상으로 그림
	{
		Scalar c(rand() & 255, rand() & 255, rand() & 255);							// 랜덤값이 들어와도 하한선 0, 상한선 255
		drawContours(dst, contours, i, c, 2);
		String desc = format("%d", i);
		putText(dst, desc, Point(contours[i][0].x, contours[i][0].y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1, LINE_AA);						// Point(contours[i][0].x, contours[i][0].y) = 외곽선 index번호의 첫번째 x좌표, 첫번째 y좌표
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void contours_hier()
{
	Mat src = imread("contours.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);											// 2단계 계층구조, 근사화

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])																// 다음 외곽선 번호(hierarchy[idx][0]) 없으면 -1 출력하고 break
	{
		Scalar c(rand() & 255, rand() & 255, rand() & 255);
		drawContours(dst, contours, idx, c, -1, LINE_8, hierarchy);														// 선의 두께 -1로 지정 → 외곽선 내부 채우기
		String desc = format("%d", idx);
		putText(dst, desc, Point(contours[idx][0].x, contours[idx][0].y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1, LINE_AA);
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void bounding_min_rect()
{
	Mat src = imread("beta2.bmp", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	vector<vector<Point>> contours;
	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	Rect rc;
	RotatedRect rrc;

	for (int i = 0; i < contours.size(); i++)
	{
		rc = boundingRect(contours[i]);
		rrc = minAreaRect(contours[i]);

		Point2f pts[4];															// 꼭지점 배열 4개
		rrc.points(pts);														// RotatedRect에 대한 배열

		rectangle(dst, rc, Scalar(0, 0, 255), 2);

		for (int j = 0; j < 4; j++)
		{
			line(dst, pts[j], pts[(j + 1) % 4], Scalar(255, 0, 0), 2);			// pts[j] = 시작점, pts[(j + 1) % 4] = 끝점
		}

		Point2f center;
		float radius = 0;
		minEnclosingCircle(contours[i], center, radius);								// 객체 하나로 하면 0번째, contours[0]
		circle(dst, center,cvRound(radius), Scalar(0, 255, 255), 0);
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

