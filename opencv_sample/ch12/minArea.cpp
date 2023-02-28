//minArea 추가

int main()
{
	Mat src = imread("beta2.bmp", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return 0;
	}

	vector<vector<Point>> contours;
	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);

	Rect boundingbox;
	RotatedRect box;

	for (int i = 0; i < contours.size(); i++) {
		boundingbox = boundingRect(contours[i]);
		box = minAreaRect(contours[i]);
	}

	Point2f vertices[4];
	box.points(vertices);

	rectangle(dst, boundingbox, Scalar(0, 0, 255), 2);

	for (int i = 0; i < 4; i++)
		line(dst, vertices[i], vertices[(i + 1) % 4], Scalar(255, 0, 0), 2);

	Point2f center;
	float radius = 0;
	minEnclosingCircle(contours[0], center, radius);
	circle(dst, center, cvRound(radius), Scalar(0, 255, 255), 1, LINE_AA);


	imshow("src", src);
	imshow("dst", dst);

	waitKey(0);
	destroyAllWindows();

	return 0;
}
