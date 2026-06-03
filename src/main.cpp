/**
 * @file main.cpp
 * @brief Sequential testing script for Road Sign Detection masks & shape logic.
 */
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "../include/ColorSegmenter.h"
#include "../include/ShapeAnalyzer.h"

int main()
{
    ColorSegmenter segmenter;
    ShapeAnalyzer analyzer;

    // --- TEST 1: RED STOP SIGN ---
    std::string redImagePath = "data/stop_sign_1.jpg";
    cv::Mat redImg = cv::imread(redImagePath);
    if (!redImg.empty())
    {
        std::cout << "Testing Red Sign... (Press key to continue)" << std::endl;
        cv::resize(redImg, redImg, cv::Size(600, (int)(redImg.rows * (600.0 / redImg.cols))));

        cv::Mat redMask = segmenter.getStaticRedMask(redImg);
        analyzer.detectRedSigns(redMask, redImg); // Run the shape logic!

        cv::imshow("1 - Mask", redMask);
        cv::imshow("1 - Final Output", redImg);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // --- TEST 2: YELLOW WARNING SIGN ---
    std::string yellowImagePath = "data/pedestrian_crossing_sign_2.jpg";
    cv::Mat yellowImg = cv::imread(yellowImagePath);
    if (!yellowImg.empty())
    {
        std::cout << "Testing Yellow Sign... (Press key to continue)" << std::endl;
        cv::resize(yellowImg, yellowImg, cv::Size(600, (int)(yellowImg.rows * (600.0 / yellowImg.cols))));

        cv::Mat yellowMask = segmenter.getStaticYellowMask(yellowImg);
        analyzer.detectWarningSign(yellowMask, yellowImg); // Run the shape logic!

        cv::imshow("2 - Mask", yellowMask);
        cv::imshow("2 - Final Output", yellowImg);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // --- TEST 3: BLUE INFO SIGN ---
    std::string blueImagePath = "data/disabled_parking_sign_1.jpg";
    cv::Mat blueImg = cv::imread(blueImagePath);
    if (!blueImg.empty())
    {
        std::cout << "Testing Blue Sign... (Press key to close)" << std::endl;
        cv::resize(blueImg, blueImg, cv::Size(600, (int)(blueImg.rows * (600.0 / blueImg.cols))));

        cv::Mat blueMask = segmenter.getStaticBlueMask(blueImg);
        analyzer.detectInfoSign(blueMask, blueImg); // Run the shape logic!

        cv::imshow("3 - Mask", blueMask);
        cv::imshow("3 - Final Output", blueImg);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::cout << "All tests complete!" << std::endl;
    return 0;
}