#include <iostream>
#include <vector>
#include <opencv/cv.hpp>
#include <ctime>
#include <string>

#include "lut.h"

#define ESCAPE 27
#define PI 3.1416

double calcTime(clock_t start_t, clock_t end_t)
{

  return(double)(end_t - start_t)/CLOCKS_PER_SEC;
}

int calcZeta(int r)
{
    if(r < 64)
        return(int)distances[r-21];
    else
        return (226 - r)/4.05;

}
using namespace cv;
using namespace std;

int main()
{
    Mat img;
    VideoCapture webcam = VideoCapture(0);
    bool success = false;
    char keypressed = 0;
    double tim = 0;
    clock_t init;

    namedWindow("Image", CV_WINDOW_AUTOSIZE);
    namedWindow("Thresholding", CV_WINDOW_AUTOSIZE);
    namedWindow("Settings", CV_WINDOW_AUTOSIZE);


    resizeWindow("Settings", 640,240);

    moveWindow("Image" ,600,100); //
    moveWindow("Thresholding" ,1240,100);
    moveWindow("Settings" ,600,640);

    if(!webcam.isOpened())
    {
        cout << "Error opening the camera" << endl;
        return 1;
    }

    //We capture a sample image to check and to obtain dimensions automatically in order to resize components

    success = webcam.read(img);
    if(success == false)
    {
        cout << "Unable to capture frame" << endl;
        return 1;
    }

    Mat featu = Mat(img.size(),img.type());
    Mat canva = Mat(img.size(), img.type());
    int number_of_labels;
    // number_of_labels - Is included the background as label 0.
    Mat labels;
    // destination labeled image (not equalized)

    Mat stats;
    // STATS:
    // CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box.
    // CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box.
    // CC_STAT_WIDTH The horizontal size of the bounding box
    // CC_STAT_HEIGHT The vertical size of the bounding box
    // CC_STAT_AREA The total area (in pixels) of the connected component

    // In the bounding box is included the shape of the binary object.

    Mat centroids;
    // Centroids floating point centroid (x,y) output for each label, including the background label
    int connectivity = 8;
    // Connectivity - Determine the neighbors of the image
    // 4 connectivity  -  8 connectivity
    //	0 1 0				1 1 1
    //	1 X 1				1 X 1
    //	0 1 0               1 1 1
    RNG rng;
    int lvalue = 10;
    int avalue = 160;
    int negrest = 28;
    int posres = 14;


    createTrackbar("b_THRESHOLDING", "Settings", &lvalue, 180);
    createTrackbar("a_THRESHOLDING", "Settings", &avalue, 255);
    createTrackbar("- restrictiveness", "Settings", &negrest, 180);
    createTrackbar("+ restrictiveness", "Settings", &posres, 180);

    while(keypressed != ESCAPE)
    {

        webcam.read(img);
        //cvtColor(canva,featu,CV_BGR2GRAY);
        init = clock();

        GaussianBlur(img,img,Size(7,7),0,0);
       // medianBlur(img,img,5);
        cvtColor(img,canva, COLOR_BGR2Lab);
        inRange(canva, Scalar(0,avalue,lvalue),Scalar(255,255,255),featu);
        //floodFill(im_thresholded, Point(0,0), Scalar(255,255,255));
        //morphologyEx( im_thresholded, im_thresholded,CV_MOP_ERODE , Mat() );
      //  morphologyEx( im_thresholded, im_thresholded,CV_MOP_ERODE , Mat() );
        //morphologyEx( featu,featu,CV_MOP_OPEN , Mat() );
        /*for(int a = 0; a < 200; a++)
        {
            morphologyEx( featu,featu, CV_MOP_OPEN , Mat() );
        }*/

        morphologyEx( featu,featu, CV_MOP_DILATE , Mat() );
        morphologyEx( featu,featu, CV_MOP_DILATE , Mat() );
        morphologyEx( featu,featu, CV_MOP_DILATE , Mat() );
        morphologyEx( featu,featu, CV_MOP_DILATE , Mat() );

        number_of_labels = connectedComponentsWithStats(featu, labels, stats, centroids, connectivity);

        cout << "number_of_labels: " << number_of_labels << endl;
        cout << "stats:" << endl << stats << endl;
        cout << "centroids:" << endl << centroids << endl;

        for(int i = 1; i < stats.rows ; i++)
        {

            int radius = (int) ( stats.at<int>(i,3))/2;
            int diffdiam = (2*sqrt(stats.at<int>(i,4)/PI))-2*radius;

            if(stats.at<int>(i,4) > 1750 && diffdiam > (0-negrest) && diffdiam < posres)
            {
                Point cornerTopLeft(stats.at<int>(i,0), stats.at<int>(i,1));
                Point cornerBottomRight(stats.at<int>(i,0) + stats.at<int>(i,2), stats.at<int>(i,1) + stats.at<int>(i,3));
                Scalar color(87,87,255);

                Point center(centroids.at<double>(i,0), centroids.at<double>(i,1));

                cout << "center x:" << center.x << "y: " << center.y << "radius: " << radius << endl;

                //rectangle(img,cornerTopLeft, cornerBottomRight, color, 2);
                circle(img,center, radius, color,3);
                circle(img,center, 1, Scalar(215,215,215),6);
                putText(img,"X: " + to_string(center.x) + "px", Point(centroids.at<double>(i,0), centroids.at<double>(i,1)+70) , CV_FONT_HERSHEY_PLAIN,1, Scalar(0,215,0),2);
                putText(img,"Y: " + to_string(center.y) + "px", Point(centroids.at<double>(i,0), centroids.at<double>(i,1)+86) , CV_FONT_HERSHEY_PLAIN ,1, Scalar(0,215,0),2);
                putText(img,"Z: " + to_string(calcZeta(radius)) + "cm", Point(centroids.at<double>(i,0), centroids.at<double>(i,1)+102) , CV_FONT_HERSHEY_PLAIN ,1, Scalar(215,215,0),2);
                putText(img,"Circ: " + to_string(diffdiam), Point(centroids.at<double>(i,0)+ 15, centroids.at<double>(i,1)) , CV_FONT_HERSHEY_PLAIN ,1, Scalar(215,215,215)),1.4;
            }
        }

        tim = calcTime(init, clock());

        putText(img,"FPS :" + to_string((int)1/tim) + " / " +to_string(tim*1000) + " ms", Point(2,20), CV_FONT_HERSHEY_PLAIN ,1, Scalar(215,215,215));

        imshow("Image", img);
        imshow("Thresholding", featu);

        keypressed = waitKey(1); //waiting 3ms to key

    }

    //Free memory
    webcam.release();
    img.release();
    featu.release();
    labels.release();
    stats.release();
    canva.release();
    centroids.release();
    destroyAllWindows();

    return 0;
}
