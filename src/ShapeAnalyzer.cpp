/**
 * @file ShapeAnalyzer.cpp
 * @brief Implements contour extraction and polygon approximation logic.
 * @author Manish & Ishaan
 */
#include "../include/ShapeAnalyzer.h"

void ShapeAnalyzer::detectStopSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const
{
    // 1. Find the boundaries of all white blobs in the mask
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // RETR_EXTERNAL ignores holes inside the shape; CHAIN_APPROX_SIMPLE compresses straight lines
    cv::findContours(binaryMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto &contour : contours)
    {
        // 2. Area Filter: Ignore tiny specks of white noise (dust, distant red cars)
        double area = cv::contourArea(contour);
        if (area < 1000.0)
        {
            continue;
        }

        // 3. Mathematical Shape Approximation
        std::vector<cv::Point> approxPolygon;
        double perimeter = cv::arcLength(contour, true);

        // Epsilon is the maximum distance between the original curve and its approximation.
        // 2% of the perimeter is the industry standard for geometric shapes.
        double epsilon = 0.02 * perimeter;
        cv::approxPolyDP(contour, approxPolygon, epsilon, true);

        // 4. The Heuristic: A perfect stop sign has 8 sides.
        // We accept 7 to 9 to account for pixelation or slight angles in the camera.
        int vertices = (int)approxPolygon.size();

        if (vertices >= 7 && vertices <= 9)
        {
            // 5. Success! Calculate the bounding box for the UI
            cv::Rect boundingBox = cv::boundingRect(approxPolygon);

            // Draw a thick green box around the sign on the ORIGINAL image
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 255, 0), 3);

            // Add the text label just above the box
            cv::putText(outputImage, "STOP SIGN",
                        cv::Point(boundingBox.x, boundingBox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }
    }
}

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

cv::Mat ColorSegmenter::getStaticRedMask(const cv::Mat &inputImage) const
{
    cv::Mat blurredImage, hsvImage;
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(5, 5), 0);
    cv::cvtColor(blurredImage, hsvImage, cv::COLOR_BGR2HSV);

    // USE YOUR HARDCODED VALUES HERE!
    cv::Mat mask1, mask2, combinedMask;
    cv::inRange(hsvImage, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsvImage, cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255), mask2);

    cv::bitwise_or(mask1, mask2, combinedMask);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, element);
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_CLOSE, element);

    return combinedMask; // Handing the clean mask over to the ShapeAnalyzer!
}