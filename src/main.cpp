#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/ColorSegmenter.h"
#include "../include/ShapeAnalyzer.h"

int main()
{
    ColorSegmenter segmenter;
    ShapeAnalyzer analyzer;
    cv::Mat frame, redMask, yellowMask, blueMask;

    std::cout << "--- STARTING STATIC IMAGE TESTS ---" << std::endl;
    std::cout << "Press any key on the image window to move to the next one." << std::endl;

    // --- 1. STATIC RED SIGN ---
    frame = cv::imread("data/stop_sign_2.jpg");
    if (!frame.empty())
    {
        cv::resize(frame, frame, cv::Size(600, (int)(frame.rows * (600.0 / frame.cols))));
        redMask = segmenter.getStaticRedMask(frame);
        analyzer.detectRedSigns(redMask, frame);
        cv::imshow("Static Demo - Red Sign", frame);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // --- 2. STATIC YELLOW SIGN ---
    frame = cv::imread("data/pedestrian_crossing_sign_2.jpg");
    if (!frame.empty())
    {
        cv::resize(frame, frame, cv::Size(600, (int)(frame.rows * (600.0 / frame.cols))));
        yellowMask = segmenter.getStaticYellowMask(frame);
        analyzer.detectWarningSign(yellowMask, frame);
        cv::imshow("Static Demo - Yellow Sign", frame);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // --- 3. STATIC BLUE SIGN ---
    frame = cv::imread("data/disabled_parking_sign_1.jpg");
    if (!frame.empty())
    {
        cv::resize(frame, frame, cv::Size(600, (int)(frame.rows * (600.0 / frame.cols))));
        blueMask = segmenter.getStaticBlueMask(frame);
        analyzer.detectInfoSign(blueMask, frame);
        cv::imshow("Static Demo - Blue Sign", frame);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::cout << "--- STARTING VIDEO DEMO ---" << std::endl;
    std::cout << "Press ESC on the video window to exit." << std::endl;

    // --- 4. DASHCAM VIDEO ---
    // Make sure your video is named this, or change the path!
    cv::VideoCapture cap("data/dashcam.mp4");

    if (!cap.isOpened())
    {
        std::cerr << "Video failed to load. Make sure the path is correct!" << std::endl;
        return -1;
    }

    double fps = 0.0;
    while (true)
    {
        int64 startTick = cv::getTickCount();
        cap >> frame;

        if (frame.empty())
        {
            std::cout << "Video finished." << std::endl;
            break;
        }

        cv::resize(frame, frame, cv::Size(800, (int)(frame.rows * (800.0 / frame.cols))));

        // Get Masks
        redMask = segmenter.getStaticRedMask(frame);
        yellowMask = segmenter.getStaticYellowMask(frame);
        blueMask = segmenter.getStaticBlueMask(frame);

        // Analyze Shapes
        analyzer.detectRedSigns(redMask, frame);
        analyzer.detectWarningSign(yellowMask, frame);
        analyzer.detectInfoSign(blueMask, frame);

        // Telemetry HUD
        int64 endTick = cv::getTickCount();
        fps = cv::getTickFrequency() / (endTick - startTick);
        cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Video Demo - Live Dashcam", frame);

        // 30ms delay plays it at normal speed. ESC (27) to exit.
        if (cv::waitKey(30) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}