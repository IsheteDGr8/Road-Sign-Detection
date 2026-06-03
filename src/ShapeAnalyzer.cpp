/**
 * @file ShapeAnalyzer.cpp
 * @brief Implements contour extraction and polygon approximation logic.
 * @author Manish & Ishaan
 */
#include "../include/ShapeAnalyzer.h"

void ShapeAnalyzer::detectRedSigns(const cv::Mat &binaryMask, cv::Mat &outputImage) const
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Calculate maximum allowed area (e.g., 20% of the total image)
    double maxSignArea = outputImage.rows * outputImage.cols * 0.20;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);

        // --- FIX 1: Min and Max Area ---
        if (area < 1000.0 || area > maxSignArea)
            continue;

        cv::Rect boundingBox = cv::boundingRect(contour);

        // --- FIX 2: The Horizon Filter ---
        // Ignore anything in the bottom 30% of the screen (the car's hood / road surface)
        if (boundingBox.y > outputImage.rows * 0.70)
            continue;

        // --- FIX 3: Aspect Ratio ---
        // Width divided by height. Stop signs and Do Not Enter signs are basically 1:1 squares.
        double aspectRatio = (double)boundingBox.width / boundingBox.height;
        if (aspectRatio < 0.75 || aspectRatio > 1.25)
            continue; // Ignore wide trucks or tall poles

        // If it passes the real-world physical filters, do the math:
        double perimeter = cv::arcLength(contour, true);
        double epsilon = 0.01 * perimeter;

        std::vector<cv::Point> approxPolygon;
        cv::approxPolyDP(contour, approxPolygon, epsilon, true);

        int vertices = (int)approxPolygon.size();

        // Branch 1: DO NOT ENTER (Circle -> Many vertices due to tight epsilon)
        if (vertices > 10)
        {
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 0, 255), 3);
            cv::putText(outputImage, "DO NOT ENTER",
                        cv::Point(boundingBox.x, boundingBox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
        // Branch 2: STOP SIGN (Octagon)
        else if (vertices >= 7 && vertices <= 9)
        {
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 255, 0), 3);
            cv::putText(outputImage, "STOP SIGN",
                        cv::Point(boundingBox.x, boundingBox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }
    }
}

void ShapeAnalyzer::detectWarningSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Maximum allowed area (20% of the screen)
    double maxSignArea = outputImage.rows * outputImage.cols * 0.20;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);

        // 1. Min/Max Area Filter
        if (area < 1000.0 || area > maxSignArea)
            continue;

        cv::Rect boundingBox = cv::boundingRect(contour);

        // 2. Horizon Filter: Ignore bottom 30% of the screen (car hood/road)
        if (boundingBox.y > outputImage.rows * 0.70)
            continue;

        // 3. Aspect Ratio: Diamonds fit in a 1:1 square
        double aspectRatio = (double)boundingBox.width / boundingBox.height;
        if (aspectRatio < 0.75 || aspectRatio > 1.25)
            continue;

        double perimeter = cv::arcLength(contour, true);
        double epsilon = 0.03 * perimeter; // 3% smoothing for sharp 4-point corners
        std::vector<cv::Point> approxPolygon;
        cv::approxPolyDP(contour, approxPolygon, epsilon, true);

        int vertices = (int)approxPolygon.size();

        // 4. Shape Heuristic: 4 sides for a diamond
        if (vertices >= 4 && vertices <= 5)
        {
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 255, 255), 3);
            cv::putText(outputImage, "WARNING SIGN",
                        cv::Point(boundingBox.x, boundingBox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        }
    }
}

void ShapeAnalyzer::detectInfoSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Maximum allowed area (20% of the screen)
    double maxSignArea = outputImage.rows * outputImage.cols * 0.20;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);

        // 1. Min/Max Area Filter
        if (area < 300.0 || area > maxSignArea)
            continue;

        cv::Rect boundingBox = cv::boundingRect(contour);

        // 2. Horizon Filter: Ignore bottom 30% of the screen
        if (boundingBox.y > outputImage.rows * 0.70)
            continue;

        // 3. Aspect Ratio: Info signs are wide or tall rectangles
        double aspectRatio = (double)boundingBox.width / boundingBox.height;
        if (aspectRatio < 0.5 || aspectRatio > 4.0)
            continue;

        double perimeter = cv::arcLength(contour, true);
        double epsilon = 0.03 * perimeter;
        std::vector<cv::Point> approxPolygon;
        cv::approxPolyDP(contour, approxPolygon, epsilon, true);

        int vertices = (int)approxPolygon.size();

        // 4. Shape Heuristic: 4 sides for a rectangle
        if (vertices >= 4 && vertices <= 5)
        {
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(255, 100, 0), 3);
            cv::putText(outputImage, "INFO SIGN",
                        cv::Point(boundingBox.x, boundingBox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 0), 2);
        }
    }
}
