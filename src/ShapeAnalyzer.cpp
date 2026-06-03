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

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area < 1000.0)
            continue;

        double perimeter = cv::arcLength(contour, true);

        // --- THE FIX: Tighter Epsilon ---
        // 1% of perimeter forces circles to map to 12+ vertices, while octagons stay at 8.
        double epsilon = 0.01 * perimeter;

        std::vector<cv::Point> approxPolygon;
        cv::approxPolyDP(contour, approxPolygon, epsilon, true);

        int vertices = (int)approxPolygon.size();
        cv::Rect boundingBox = cv::boundingRect(approxPolygon);

        // Branch 1: DO NOT ENTER (Circle)
        // With a 0.01 epsilon, circles will break down into many small lines (usually 12+)
        if (vertices > 10)
        {
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 0, 255), 3); // Red Box
            cv::putText(outputImage, "DO NOT ENTER",
                        cv::Point(boundingBox.x, boundingBox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
        // Branch 2: STOP SIGN (Octagon)
        // Straight-edged octagons will reliably snap to 7-9 sides even with tight epsilon
        else if (vertices >= 7 && vertices <= 9)
        {
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 255, 0), 3); // Green Box
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

    for (const auto &contour : contours)
    {
        if (cv::contourArea(contour) < 1000.0)
            continue;

        std::vector<cv::Point> approxPolygon;
        double perimeter = cv::arcLength(contour, true);

        // 3% epsilon to aggressively snap rounded diamond corners into 4 sharp vertices
        double epsilon = 0.03 * perimeter;
        cv::approxPolyDP(contour, approxPolygon, epsilon, true);

        int vertices = (int)approxPolygon.size();

        // Heuristic: Diamond warning signs have exactly 4 sides (accepting 4-5 for pixelation)
        if (vertices >= 4 && vertices <= 5)
        {
            cv::Rect boundingBox = cv::boundingRect(approxPolygon);
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 255, 255), 3); // Yellow box
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

    for (const auto &contour : contours)
    {
        if (cv::contourArea(contour) < 1000.0)
            continue;

        std::vector<cv::Point> approxPolygon;
        double perimeter = cv::arcLength(contour, true);
        double epsilon = 0.03 * perimeter;
        cv::approxPolyDP(contour, approxPolygon, epsilon, true);

        int vertices = (int)approxPolygon.size();

        // Heuristic: Rectangular info signs have exactly 4 sides
        if (vertices >= 4 && vertices <= 5)
        {
            cv::Rect boundingBox = cv::boundingRect(approxPolygon);
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(255, 100, 0), 3); // Blue box
            cv::putText(outputImage, "INFO SIGN",
                        cv::Point(boundingBox.x, boundingBox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 0), 2);
        }
    }
}