#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

int mouseX[4] = {0, 0, 0, 0};
int mouseY[4] = {0, 0, 0, 0};
int idx = 0;

void drawCircle(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        mouseX[idx] = x;
        mouseY[idx] = y;
        idx++;
    }
}

int main(int argc, char** argv) {

    cv::Mat org;
    cv::VideoCapture cap;
    // open selected camera using selected API
    cap.open(std::string("/media/bkhti4/External/Trajectorix/cityscape.mp4"));

    cv::namedWindow("Mark points");
    cap.read(org);
    cv::resize(org, org, cv::Size(1024, 512));
    cv::setMouseCallback("Mark points", drawCircle);

    std::cout << "Mark the points of an area to generate birds-eye view " << std::endl;
    while (true) {
        cv::imshow("Mark points", org);
        if (idx > 0 && idx <= 5){
            cv::circle(org, cv::Point(mouseX[idx-1], mouseY[idx-1]), 4, cv::Scalar( 0, 0, 255 ), 1, 2);
        } 
        if (cv::waitKey(1) == 81 || cv::waitKey(1) == 113){
            break;
        }
    }
    cv::destroyWindow("Mark points");

    std::cout << "Points collected" << std::endl;

    cv::Point2f srcVertices[4];
    
    //Define points that are used for generating bird's eye view. This was done by trial and error. Best to prepare sliders and configure for each use case.
    srcVertices[0] = cv::Point(mouseX[0], mouseY[0]);
    srcVertices[1] = cv::Point(mouseX[1], mouseY[1]);
    srcVertices[2] = cv::Point(mouseX[2], mouseY[2]);
    srcVertices[3] = cv::Point(mouseX[3], mouseY[3]);

    cv::Point2f dstVertices[4];
    //Destination vertices. Output is 640 by 480px Point2f dstVertices[4];
    dstVertices[0] = cv::Point(0, 0);
    dstVertices[1] = cv::Point(640, 0);
    dstVertices[2] = cv::Point(640, 480);
    dstVertices[3] = cv::Point(0, 480);

    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcVertices, dstVertices);
    cv::Mat dst(480, 640, CV_8UC3);

    cv::Mat invertedPerspectiveMatrix;
    cv::invert(perspectiveMatrix, invertedPerspectiveMatrix);

    while (true)
    {
        cap.read(org);
        cv::resize(org, org, cv::Size(1024, 512));
        cv::warpPerspective(org, dst, perspectiveMatrix, dst.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::imshow("Result", dst); //Show the image
        if (cv::waitKey(1) == 81 || cv::waitKey(1) == 113){
            break;
        }
    }
    cap.release();    
}