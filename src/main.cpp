#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/ColorSegmenter.h"
#include "../include/ShapeAnalyzer.h"

int main()
{
    ColorSegmenter segmenter;
    ShapeAnalyzer analyzer;

    // --- THE ONLY MAJOR CHANGE ---
    // Instead of '0' for webcam, pass the path to your video file!
    std::string videoPath = "data/dashcam.mp4";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the video file!" << std::endl;
        return -1;
    }

    std::cout << "Video Feed started. Press 'ESC' to exit early." << std::endl;

    cv::Mat frame, redMask, yellowMask, blueMask;
    double fps = 0.0;

    while (true)
    {
        int64 startTick = cv::getTickCount();

        cap >> frame;

        // If the video reaches the end, frame.empty() becomes true and it gracefully exits
        if (frame.empty())
        {
            std::cout << "End of video reached." << std::endl;
            break;
        }

        // Resize for performance so the HUD fits on your screen
        cv::resize(frame, frame, cv::Size(800, (int)(frame.rows * (800.0 / frame.cols))));

        // 1. Get Masks
        redMask = segmenter.getStaticRedMask(frame);
        yellowMask = segmenter.getStaticYellowMask(frame);
        blueMask = segmenter.getStaticBlueMask(frame);

        // 2. Analyze Shapes
        analyzer.detectRedSigns(redMask, frame);
        analyzer.detectWarningSign(yellowMask, frame);
        analyzer.detectInfoSign(blueMask, frame);

        // 3. Telemetry HUD
        int64 endTick = cv::getTickCount();
        fps = cv::getTickFrequency() / (endTick - startTick);
        cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // Display the result
        cv::imshow("Road Sign Detection System - DASHCAM", frame);

        // --- PLAYBACK SPEED FIX ---
        // cv::waitKey(1) plays the video as fast as your CPU allows (fast-forward).
        // cv::waitKey(30) adds a 30ms delay, making it play at a normal ~30fps speed!
        if (cv::waitKey(30) == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}