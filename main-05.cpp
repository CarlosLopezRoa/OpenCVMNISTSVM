

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

bool debugP = true;   // debug Program
bool debugV = false;   //debug Visual

Mat detectAndDisplay( Mat frame );
String window_name = "Capture - MNIST";

void highpassGaussianFilter(double D, Mat& Filter);

void shiftSpectrum(Mat& input); //ok
void logTransformation(Mat& input);



void highpassGaussianFilter(double D, Mat& Filter)
{
//    order: oder of filter
//    D: diamter of filter cutoff frequency 0.607 of maximum value
//    Filter: matrix of the filter


     double dist;
     double t;

     t = (double)getTickCount();
     const int times = 100;

     for (int i=0; i<Filter.rows; i++)
     {
         for (int j=0; j<Filter.cols; j++)
         {
           dist = sqrt( (i-Filter.rows/2)*(i-Filter.rows/2) + (j-Filter.cols/2)*(j-Filter.cols/2) );

           Filter.at<float>(i,j) = 1 - exp(-dist*dist/(2*D*D));

         }
     }
   t = 1000*((double)getTickCount() - t)/getTickFrequency();
   t /= times;
   //cout << "Time of highpass Gaussian filter with D: "<<D<<" for "<<Filter.rows<<"x"<<Filter.cols<<" image: " << t << " milliseconds."<< endl;
}


/** @function detectAndDisplay */
Mat detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    Mat src;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    src = frame_gray.clone();
    Mat imgGaussian;
    GaussianBlur(src, imgGaussian, Size(3,3), 1.0);

    //Mat dst, detected_edges;
    //int edgeThresh = 1;
    //int lowThreshold = 60;
    //int const max_lowThreshold = 100;
    //int ratio = 3;
    //int kernel_size = 3;

    //Canny( imgGaussian, imgGaussian, lowThreshold, lowThreshold*ratio, kernel_size );

    //dst = Scalar::all(0);

    //src.copyTo( dst, imgGaussian);

    //cvFindContours( imgGaussian, imgGaussian, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0) );


    Mat img;
    img = imgGaussian.clone();
   // imshow("Gaussian image", img);
//    int M = getOptimalDFTSize( img.rows );
//    int N = getOptimalDFTSize( img.cols );
//    Mat padded;
//    copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
//
//
//
//    //namedWindow("Image histogram", CV_WINDOW_FREERATIO);
//    //drawHist(img, "Image histogram");
//    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
//    Mat complexImg;
//    merge(planes, 2 , complexImg);
//
//    dft(complexImg, complexImg); //discrete Fourier Trasform
//    shiftSpectrum(complexImg); //shift the Spectrum in the image centre
//    Mat Filter(complexImg.size(), CV_32F); // create a Mat for the filtering function
//     highpassGaussianFilter(15, Filter);
//     Mat filterOut = Filter.clone();
//    normalize(filterOut, filterOut, 0, 1, CV_MINMAX);
//    Mat complexFilter;
//    Mat planesFilter[] = {Filter, Filter};
//    merge(planesFilter, 2, complexFilter);
//    Mat temp = complexImg.clone();
//    multiply(complexImg, complexFilter, complexImg); //Calculates the per-element scaled product of two arrays.
//    split(complexImg, planes);
//    magnitude(planes[0], planes[1], planes[0]);
//    Mat mag = planes[0];
//    logTransformation(mag);
//     shiftSpectrum(complexImg); //shift the Spectrum in the image centre
//
//    dft(complexImg, complexImg, DFT_INVERSE || DFT_SCALE);
//
//    split(complexImg, planes);
//     Mat outMagt = planes[0];
//    Mat outMag = outMagt(Rect(0,0,img.cols, img.rows));
//    normalize(outMag, outMag, 0, 1, CV_MINMAX);
//    imshow("rebuild image", outMag);
    //img = outMag.clone();
//    Mat hist;
//    int histSize = 256;
//    calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, 0);
//    Mat histImage = Mat::ones(200, 320, CV_8U)*255;
//    normalize(hist, hist, 0, histImage.rows, CV_MINMAX, CV_32F);


      int C= 3;
    threshold(img,img, 20,255,CV_THRESH_BINARY_INV);
    //adaptiveThreshold(img,img, 100, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3,C);
   // namedWindow("thresh image", CV_WINDOW_FREERATIO);
   // imshow("thresh image", img);

    //-- Detect faces

    //-- Show what you got
    imshow( window_name, img );
    return img;
}


void shiftSpectrum(Mat& input)
{

    // crop the spectrum, if it has an odd number of rows or columns
    //complexImg = complexImg(Rect(0, 0, complexImg.cols & -2, complexImg.rows & -2));

    int cx = input.cols/2;
    int cy = input.rows/2;

    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center
    Mat tmp;
    Mat q0(input, Rect(0, 0, cx, cy));
    Mat q1(input, Rect(cx, 0, cx, cy));
    Mat q2(input, Rect(0, cy, cx, cy));
    Mat q3(input, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

}


//-------------------------------------------------------------------
// logTransformation()
//-------------------------------------------------------------------
//This function determines the log() transformation applied
// on the Spectrum determined to enhance its visibility.


void logTransformation(Mat& input)
{
    input += Scalar::all(1);
    log(input, input);
    normalize(input, input, 0, 1, CV_MINMAX);
}




//-------------------------------------------------------------------
// lowpassIdealFilter()
//-------------------------------------------------------------------
//This function evaluates the lowpass ideal filter
// the unique parameter is the diameter of the ideal filter.
// This code is simple but it is naive code! It could be improved
// with usage of pointer approach. Anyway close to 10ms as computational time
// on my PC!




void createData( Mat& allData, Mat& allClasses, Mat& trainData, Mat& trainClasses, Mat&testData, Mat& testClasses, Ptr<TrainData>& trainingInData, int K, double ratio)
{
    int layout = ROW_SAMPLE;
    int NSamples = allData.rows;
    int maxSamples = NSamples/K;
    int numSamples = maxSamples*ratio;

    if (debugP)
    {
     cout<<" Inside createData() " << endl << endl;
     cout<<"layout: " << layout << endl;
     cout<<"NSamples: " << NSamples << endl;
     cout<<"K: " << K << endl;
     cout<<"maxSamples: " <<maxSamples << endl;
     cout<<"numSamples: " <<numSamples << endl;
    }

//    Mat allTrainData;
//    Mat allTrainClasses;

    int indexClass = 0;
    int numClass = 0;
    int trainIndex = 0;

    for (int i=0; i<NSamples; i++)  //main loop
    {
      //for each class read the specified number of Samples and class values
      if (indexClass==numSamples)
      {
              numClass++;
              indexClass = 0;
              for (int k=0; k<(maxSamples-numSamples); k++)
              {
                 Mat tmp = allData.row(i++)+0;   //+0 explanation see row() operator in OpenCV manual
                 testData.push_back(tmp.reshape(1,1) );
                 testClasses.push_back(allClasses.at<int32_t>(0,i));

              }
      }

       if (i<NSamples)
       {
           Mat tmp = allData.row(i)+0;   //+0 explanation see row() operator in OpenCV manual
          // cout<<" tmp: " << endl << tmp << endl;
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

    if (debugP)
    {
     cout<<" after creation of training data... " << endl << endl;
     cout<<"allData size: " << allData.size() << endl;
     cout<<"allClasses size: " << allClasses.size() << endl;
     cout<<"trainData size: " << trainData.size() << endl;
     cout<<"trainClasses size: " << trainClasses.size() << endl;
     cout<<"testData size: " << testData.size() << endl;
     cout<<"testClasses size: " << testClasses.size() << endl;

     cout<<"layout: " << layout << endl;
     cout<<"NSamples: " << NSamples << endl;
     cout<<"NTraining: " <<  trainingInData->getNTrainSamples() << endl;
     cout<<"K: " << K << endl;
     cout<<"ratio: " << ratio << endl;
     cout<<"maxSamples*ratio: " <<maxSamples << endl;
     cout<<"numSamples*ratio: " <<numSamples << endl;
     cout<<" Inside createData() " << endl << endl;
    }
}




/** @function main */
int main( int argc, char** argv )
{
    int size = 20;

     //Create train and test data and classes
     int overallSamples = 5000;
     int numTests = 2500;
     int numTrainSamples = 250;
     Mat allData;
     Mat allClasses;
     bool activedTraining = true;
     bool activedTesting = true;
     bool liveTesting = false;


     Mat trainData;
    Mat trainClasses;
    Mat testData;
    Mat testClasses;
     Ptr<TrainData> trainingInData;


    if (activedTesting){

    //load the image and transform it into gray scale image
     Mat img = imread("digits.png", CV_LOAD_IMAGE_GRAYSCALE);

    //Now we split the image to 5000 cells, each 20x20 size
    //image size 2000 x 1000 therefore  100 images for row and 500 images for digits
    // Reading and processing all possible small images
    int indexCell = 0; //indexCell is the unique index between 0 to 5000 images
    int indexDigits = 0; //index concerning the digits from 0 to 500
    int digit = 0;       //digit
    int32_t digValue =0;


    Mat cell, tmp;
    for (int j=0; j< img.rows; j+=size)   // rows
     {
      for (int i=0; i< img.cols; i+=size) // cols
       {
        //data
        //ROI of 20x20 pixels
        Mat tmp = img(Rect(i,j,size,size) );
        Mat floatImg;
        tmp.convertTo(floatImg, CV_32F);

       if (debugP) cout<<"indexCell: " << indexCell <<" indexDigits: " << indexDigits << "  " << " digit: " << digValue << "  (x,y)= "<< i<<"x"<<j << endl;

        allData.push_back(floatImg.reshape(1,1) );

//        if (digit == 0)
//            digValue = 1;
//        else
//            digValue = -1;
        digValue = digit;
        allClasses.push_back(int32_t (digValue));


        indexCell++;
        indexDigits++;
        if (indexDigits==500)
        {
              digit++;
              indexDigits =0;
        }

        //visualization:
         if (debugV)
         {
            string cellTitle = format("digit %d   cell(%d) %dx%d", digit, indexCell, i, j);
            namedWindow(cellTitle, CV_WINDOW_KEEPRATIO);
            imshow(cellTitle,tmp );
            waitKey(200);
            destroyWindow(cellTitle);
         }
       }
     }



    cout<<"loaded all data ok!!!" << endl;


    //------------------------ 2. Set up the support vector machines parameters --------------------
    //------------------------ 3. Train the svm ----------------------------------------------------
    cout << "svm creation..." << endl;




    double allRatio = 0.5;  // 0.7 means 70% Training and 30% Test
    //It is a little bit slowly with respect to the direct approach...
    createData(allData, allClasses, trainData, trainClasses, testData, testClasses, trainingInData, 10, allRatio);
    }
    Ptr<SVM> svm;

    if (activedTraining)
    {
    svm = SVM::create();
    svm->setType(SVM::C_SVC);  //C_SVC
    svm->setC(2.67);
    svm->setGamma(5.383);
    svm->setKernel(SVM::POLY);  //SVM::LINEAR
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
    svm->setDegree(5);
    //training data
    Clock C;
    C.start();
    cout << "Starting training process" << endl;
    cout << "ROW_SAMPLE: "<< ROW_SAMPLE << endl;
    svm->train(trainingInData);  //In this case we use Ptr<TrainData>
    ///svm->train(trainData, ROW_SAMPLE, trainClasses );
    cout << "Finished training process" << endl;
    svm->save("svmOCR_01.dat");
    C.end();
    cout<<"elapsed time: " << C.elapsedTime() << " ms" << endl;
    }
    else
    {
     //It is possible to create a SVM already trained loading the "svm.dat" file
     svm = StatModel::load<SVM>("svmOCR_02.dat");
    }
    if (activedTesting){
     //test data
     Mat result;

    cout<<"testData.rows: "<< testData.rows << endl;
    int correct =0;

    for (int k=0; k<testData.rows; k++)
    {
        Mat tmp = testData.row(k)+0;   //+0 explanation see row() operator in OpenCV manual
        float response = svm->predict(tmp);
//        imshow("new",tmp);
//
//        int rows = tmp.rows;
//        int cols = tmp.cols;
//
//        cv::Size s = tmp.size();
//        rows = s.height;
//        cols = s.width;
//        fprintf(stdout,"%d\n",rows);
//        fprintf(stdout,"%d\n",cols);
//
//        waitKey();
        cout<<"k: " << k << " response: "<< response <<" testData: " << int(testClasses.at<int32_t>(0,k)) <<endl;
        if ( (int32_t)(response) ==  testClasses.at<int32_t>(0,k))
            correct++;

    }

    //it is possible to write... but it doesn't work
    //result = svm->predict(testData);


     cout<<" correct matches: " << correct << endl;
     cout<<" accuracy: " << (correct*100.0/double(testData.rows)) << endl;
    }
    if (liveTesting){

    VideoCapture capture;
    Mat frame;

    //-- 1. Load the cascades

    //-- 2. Read the video stream
    capture.open( 0 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        //-- 3. Apply the classifier to the frame
        Mat captured;
        captured = detectAndDisplay( frame );
        Size size(20,20);
        resize(captured,captured,size);
        Mat floatImg;
        captured.convertTo(floatImg, CV_32F);
        Mat allData;
        allData.push_back(floatImg.reshape(1,1) );
        Mat tmp = allData.row(0)+0;
        imshow("new",tmp);

//        int rows;
//        int cols;
//        cv::Size s = tmp.size();
//        rows = s.height;
//        cols = s.width;
//        fprintf(stdout,"%d\n",rows);
//        fprintf(stdout,"%d\n",cols);
//
//        waitKey();


        float response = svm->predict(tmp);
        cout<<"Prediction "<< response << endl;


        int c = waitKey(10);
        if( (char)c == 27 ) { break; } // escape
    }
    }
      return 0;
}


