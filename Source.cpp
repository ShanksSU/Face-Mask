#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iterator>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <string>
#include <tuple>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const int inWidth = 1920;
const int inHeight = 1080;

//人臉模型_tensorflow
string faceProto = "Resources/model/opencv_face_detector.pbtxt";
string faceModel = "Resources/model/opencv_face_detector_uint8.pb";

//人臉模型_caffe
string caffeConfigFile = "Resources/model/deploy.prototxt";
string caffeWeightFile = "Resources/model/res10_300x300_ssd_iter_140000.caffemodel";

//顏色
Scalar Color_Text = Scalar(0, 0, 0);
Scalar Color_Face = Scalar(183, 0, 255);

int main(int argc, const char **argv) {
	try {
		//caffe
		Net faceNet = readNetFromCaffe(caffeConfigFile, caffeWeightFile);

		//tensorflow
		//Net faceNet = readNetFromTensorflow(faceModel, faceProto);

		//優化相機模塊
		setUseOptimized(1);

		//開啟相機
		VideoCapture cap("Resources/D/C97.mp4");
		cap.set(CAP_PROP_FRAME_WIDTH, inWidth);
		cap.set(CAP_PROP_FRAME_HEIGHT, inHeight);
		cap.set(CAP_PROP_AUTOFOCUS, 1);


		if (cap.isOpened()) {
			cout << "Emmm..." << endl;
		} else {
			cout << "大意了 沒有接" << endl;
			return 0;
		}


		while (1) {

			//人臉檢測結果圖
			Mat frameFace;
			//frameFace = imread("Resources/D/P3.jpeg");
			cap.read(frameFace);
			Mat InvMaskImg;
			Mat logo = imread("Resources/D/icon1.png");
			Mat imageROI, GrayImg, MaskImg;

			//縮放尺寸
			double inScaleFactor = 1.0;

			Mat tmp;
			resize(frameFace, tmp, Size(300, 300));

			//檢測圖大小
			Size size = Size(300, 300);

			Scalar meanVal = Scalar(104.0, 117.0, 123.0);

			//caffe
			Mat inputBlob = blobFromImage(tmp, inScaleFactor, size, meanVal, false, false);

			//tensorflow
			//Mat inputBlob = blobFromImage(FaceDnn, inScaleFactor, size, meanVal, true, false);

			faceNet.setInput(inputBlob, "data");

			//四維矩陣輸出 (4-D矩陣)
			Mat detection = faceNet.forward();

			//提取結果資訊
			Mat detectionMat(detection.size[2], detection.size[3], CV_32F, (float *)detection.data);//detection.ptr<float>()

			for (int i = 0; i < detectionMat.rows; i++) {

				float conf_threshold = 0.6;

				//預測概率
				float confidence = detectionMat.at<float>(i, 2);

				//---辨識率
				if (confidence > conf_threshold) {
					//i是面部數的迭代器
					//左上角點，座標在[0,1]之間被歸一化,坐標應該乘以原始圖像的高度和寬度,以得到圖像上正確的邊界框
					int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameFace.cols);
					int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameFace.rows);

					//右下角點，座標在[0,1]之間被歸一化,坐標應該乘以原始圖像的高度和寬度,以得到圖像上正確的邊界框
					int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameFace.cols);
					int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameFace.rows);

					Rect object(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
					//cout << object << endl;;
					cout << confidence << endl;

					//影象框選
					rectangle(frameFace, object, Color_Face, 2);//Point(x1, y1), Point(x2, y2)
											  //寬      高
					//resize(logo, logo, Size(x2 - x1, y2 - y1));
					resize(logo, logo, Size(x2 - x1, x2 - x1));
					cvtColor(logo, GrayImg, COLOR_BGR2GRAY);
					threshold(GrayImg, MaskImg, 254, 255, THRESH_BINARY);
					bitwise_not(MaskImg, InvMaskImg);

					imageROI = frameFace(Rect(x1, y1, logo.cols, logo.rows));	//(y2 - y1) - ((y2 - y1) / 6)
					//addWeighted(imageROI, 1.0, logo, 0.3, 0, imageROI);
					logo.copyTo(imageROI, InvMaskImg);

					/*
					int baseLine = 0;

					//產生文字
					//string SSIN = "Prob=" + to_string(confidence);
					string SSIN = "face";

					//調整文字矩形大小
					Size labelSize = getTextSize(SSIN, FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, 1, &baseLine);

					//文字矩形生成座標
					Point TSJ(x1-0.5, y1 - labelSize.height + 18);

					//產生文字矩形
					rectangle(frameFace, Rect(Point(TSJ.x, TSJ.y - labelSize.height), Size(labelSize.width, labelSize.height + baseLine)), Scalar(183, 0, 255), -1);

					//文字輸出
					putText(frameFace, SSIN, Point(x1-0.5, y1 - labelSize.height + 18), FONT_HERSHEY_SIMPLEX, 1, Color_Text, 2);
					*/
				}
			}

			//顯示/儲存結果
			imshow("Face_Detection_DNN", frameFace);
			waitKey(1);
		}
		destroyAllWindows();
		return 0;
	}
	catch (exception &e) {
		cout << e.what();
		//system("pause");
		return 0;
	}
}