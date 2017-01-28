#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include "Clock.h"

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"


#include <iostream>
#include <stdio.h>


/*


*/


using namespace std;
using namespace cv;
using namespace cv::ml;

bool activedTraining = false;
bool activedTesting = false;
bool liveTesting = true;


Mat detectAndDisplay( Mat frame );
void createData( Mat& allData, Mat& allClasses, Mat& trainData, Mat& trainClasses, Mat&testData, Mat& testClasses, Ptr<TrainData>& trainingInData, int K, double ratio);
String window_name = "Capture - MNIST";

/** @function main */
int main( int argc, char** argv )
{
    int size = 20;

     int overallSamples = 5000;
     int numTests = 2500;
     int numTrainSamples = 250;
     Mat allData;
     Mat allClasses;

     Mat trainData;
     Mat trainClasses;
     Mat testData;
     Mat testClasses;
     Ptr<TrainData> trainingInData;

    if (activedTesting){

    Mat img = imread("digits.png", CV_LOAD_IMAGE_GRAYSCALE);

    int indexCell = 0;
    int indexDigits = 0;
    int digit = 0;
    int32_t digValue =0;


    Mat cell, tmp;
    for (int j=0; j< img.rows; j+=size)
     {
      for (int i=0; i< img.cols; i+=size)
       {
        Mat tmp = img(Rect(i,j,size,size) );
        Mat floatImg;
        tmp.convertTo(floatImg, CV_32F);

        allData.push_back(floatImg.reshape(1,1) );
        digValue = digit;
        allClasses.push_back(int32_t (digValue));

        indexCell++;
        indexDigits++;
        if (indexDigits==500)
        {
              digit++;
              indexDigits =0;
        }
       }
     }
    cout<<"loaded all data ok!!!" << endl;
    cout << "svm creation..." << endl;


    double allRatio = 0.5;
    createData(allData, allClasses, trainData, trainClasses, testData, testClasses, trainingInData, 10, allRatio);
    }
    Ptr<SVM> svm;

    if (activedTraining)
    {
    svm = SVM::create();
    svm->setType(SVM::C_SVC);  //C_SVC
    svm->setC(2.67);
    svm->setGamma(5.383);
    svm->setKernel(SVM::LINEAR);  //SVM::LINEAR
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
    //svm->setDegree(5);
    Clock C;
    C.start();
    cout << "Starting training process" << endl;
    cout << "ROW_SAMPLE: "<< ROW_SAMPLE << endl;
    svm->train(trainingInData);
    cout << "Finished training process" << endl;
    svm->save("svmOCR_01.dat");
    C.end();
    cout<<"elapsed time: " << C.elapsedTime() << " ms" << endl;
    }
    else
    {
     svm = StatModel::load<SVM>("svmOCRLinear.dat");
    }
    if (activedTesting){
    Mat result;

    cout<<"testData.rows: "<< testData.rows << endl;
    int correct =0;

    for (int k=0; k<testData.rows; k++)
    {
        Mat tmp = testData.row(k)+0;
        float response = svm->predict(tmp);
        cout<<"k: " << k << " response: "<< response <<" testData: " << int(testClasses.at<int32_t>(0,k)) <<endl;
        if ( (int32_t)(response) ==  testClasses.at<int32_t>(0,k))
            correct++;

    }
     cout<<" correct matches: " << correct << endl;
     cout<<" accuracy: " << (correct*100.0/double(testData.rows)) << endl;
    }
    if (liveTesting){

    VideoCapture capture;
    Mat frame;

    capture.open( 0 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        Mat captured;
        captured = detectAndDisplay( frame );
        Size size(20,20);
        resize(captured,captured,size);
        equalizeHist( captured, captured );
        imshow("Resize",captured);
        Mat floatImg;
        captured.convertTo(floatImg, CV_32F);
        Mat allData;
        allData.push_back(floatImg.reshape(1,1) );
        Mat tmp = allData.row(0)+0;
        imshow("Reshape",tmp);

        float response = svm->predict(tmp);
        cout<<"Prediction "<< response << endl;


        int c = waitKey(10);
        if( (char)c == 27 ) { break; }
    }
    }
      return 0;
}

/** @function detectAndDisplay */
Mat detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    Mat src;
//imshow( "Color", frame );
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
//imshow( "Grayscale", frame_gray );
    equalizeHist( frame_gray, frame_gray );
//imshow( "Equalized", frame_gray );

    src = frame_gray.clone();
    Mat imgGaussian;
    GaussianBlur(src, imgGaussian, Size(3,3), 1.0);
//imshow( "Filtered", imgGaussian );

    Mat img;
    img = imgGaussian.clone();

    threshold(img,img, 20,255,CV_THRESH_BINARY_INV);
//imshow( "Threshold", img );
    imshow( window_name, img );
    return img;
}

void createData( Mat& allData, Mat& allClasses, Mat& trainData, Mat& trainClasses, Mat&testData, Mat& testClasses, Ptr<TrainData>& trainingInData, int K, double ratio)
{
    int layout = ROW_SAMPLE;
    int NSamples = allData.rows;
    int maxSamples = NSamples/K;
    int numSamples = maxSamples*ratio;

    int indexClass = 0;
    int numClass = 0;
    int trainIndex = 0;

    for (int i=0; i<NSamples; i++)
    {
      if (indexClass==numSamples)
      {
              numClass++;
              indexClass = 0;
              for (int k=0; k<(maxSamples-numSamples); k++)
              {
                 Mat tmp = allData.row(i++)+0;
                 testData.push_back(tmp.reshape(1,1) );
                 testClasses.push_back(allClasses.at<int32_t>(0,i));
              }
      }

       if (i<NSamples)
       {
          Mat tmp = allData.row(i)+0;
          trainData.push_back(tmp.reshape(1,1) );
          trainClasses.push_back(allClasses.at<int32_t>(0,i));
          indexClass++;
       }
    }

    trainingInData=TrainData::create(trainData, ROW_SAMPLE, trainClasses  );

    layout = trainingInData->getLayout();
    NSamples = trainingInData->getNSamples();
    maxSamples = NSamples/K;
    numSamples = maxSamples*ratio;

}
