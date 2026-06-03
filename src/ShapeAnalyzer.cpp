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