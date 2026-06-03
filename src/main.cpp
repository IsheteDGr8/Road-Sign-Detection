/**
 * @file main.cpp
 * @brief Sequential testing script for Road Sign Detection masks.
 */
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "../include/ColorSegmenter.h"

int main()
{
    ColorSegmenter segmenter;

    // --- TEST 1: RED MASK ---
    std::string redImagePath = "data/stop_sign_2.jpg"; // Update if your filename is different
    cv::Mat redImg = cv::imread(redImagePath);

    if (!redImg.empty())
    {
        std::cout << "Testing Red Mask... (Press any key on the image window to continue)" << std::endl;
        cv::resize(redImg, redImg, cv::Size(600, (int)(redImg.rows * (600.0 / redImg.cols)))); // Resize for screen
        cv::Mat redMask = segmenter.getStaticRedMask(redImg);

        cv::imshow("1 - Original Red Image", redImg);
        cv::imshow("1 - Static Red Mask", redMask);
        cv::waitKey(0);          // Pause execution until you press a key
        cv::destroyAllWindows(); // Clean up before the next test
    }
    else
    {
        std::cerr << "Could not load red test image." << std::endl;
    }

    // --- TEST 2: YELLOW MASK ---
    std::string yellowImagePath = "data/pedestrian_crossing_sign_2.jpg";
    cv::Mat yellowImg = cv::imread(yellowImagePath);

    if (!yellowImg.empty())
    {
        std::cout << "Testing Yellow Mask... (Press any key on the image window to continue)" << std::endl;
        cv::resize(yellowImg, yellowImg, cv::Size(600, (int)(yellowImg.rows * (600.0 / yellowImg.cols))));
        cv::Mat yellowMask = segmenter.getStaticYellowMask(yellowImg);

        cv::imshow("2 - Original Yellow Image", yellowImg);
        cv::imshow("2 - Static Yellow Mask", yellowMask);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    else
    {
        std::cerr << "Could not load yellow test image." << std::endl;
    }

    // --- TEST 3: BLUE MASK ---
    std::string blueImagePath = "data/disabled_parking_sign_1.jpg";
    cv::Mat blueImg = cv::imread(blueImagePath);

    if (!blueImg.empty())
    {
        std::cout << "Testing Blue Mask... (Press any key on the image window to close)" << std::endl;
        cv::resize(blueImg, blueImg, cv::Size(600, (int)(blueImg.rows * (600.0 / blueImg.cols))));
        cv::Mat blueMask = segmenter.getStaticBlueMask(blueImg);

        cv::imshow("3 - Original Blue Image", blueImg);
        cv::imshow("3 - Static Blue Mask", blueMask);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    else
    {
        std::cerr << "Could not load blue test image." << std::endl;
    }

    std::cout << "All static mask tests complete!" << std::endl;
    return 0;
}