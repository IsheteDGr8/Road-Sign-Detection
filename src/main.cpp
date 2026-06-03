/**
 * @file main.cpp
 * @brief Entry point for the Road Sign Detection system.
 */
#include <iostream>
#include "../include/ColorSegmenter.h"
#include "../include/ShapeAnalyzer.h"

int main()
{
    std::string imagePath = "data/stop_sign_2.jpg";
    cv::Mat rawImage = cv::imread(imagePath);

    if (rawImage.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Instantiate our two modular classes
    ColorSegmenter segmenter;
    ShapeAnalyzer analyzer;

    // Step 1: Isolate the color
    cv::Mat redMask = segmenter.getStaticRedMask(rawImage);

    // Step 2: Analyze the shape and draw on the original image
    analyzer.detectStopSign(redMask, rawImage);

    // Step 3: Show the final results
    cv::imshow("Binary Mask (What the computer sees)", redMask);
    cv::imshow("Final Output", rawImage);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}