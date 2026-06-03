/**
 * @file ShapeAnalyzer.cpp
 * @brief Implements contour extraction and polygon approximation logic.
 */
#include "../include/ShapeAnalyzer.h"

void ShapeAnalyzer::detectRedSigns(const cv::Mat &binaryMask, cv::Mat &outputImage) const
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // FIX: Allow signs to take up to 95% of the screen (for close-up static images)
    double maxSignArea = outputImage.rows * outputImage.cols * 0.95;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area < 1000.0 || area > maxSignArea)
            continue;

        cv::Rect boundingBox = cv::boundingRect(contour);

        // Horizon filter: Ignore objects touching the absolute bottom edge
        if (boundingBox.y > outputImage.rows * 0.85)
            continue;

        double aspectRatio = (double)boundingBox.width / boundingBox.height;
        if (aspectRatio < 0.75 || aspectRatio > 1.25)
            continue;

        double perimeter = cv::arcLength(contour, true);
        double epsilon = 0.01 * perimeter;

        std::vector<cv::Point> approxPolygon;
        cv::approxPolyDP(contour, approxPolygon, epsilon, true);

        int vertices = (int)approxPolygon.size();

        if (vertices > 10)
        {
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 0, 255), 3);
            cv::putText(outputImage, "DO NOT ENTER",
                        cv::Point(boundingBox.x, boundingBox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
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

    double maxSignArea = outputImage.rows * outputImage.cols * 0.99;

    // --- NEW LOGIC: The Relative Scale Filter ---
    // First pass: Find the absolute largest yellow shape in the frame
    double largestArea = 0.0;
    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area > largestArea && area < maxSignArea)
        {
            largestArea = area;
        }
    }

    // Second pass: Draw bounding boxes
    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);

        // FIX: Ignore shapes smaller than 500px OR shapes that are less than 30% the size of the main sign!
        // This effortlessly filters out the small supplementary arrow plates.
        if (area < 500.0 || area > maxSignArea || area < (largestArea * 0.30))
            continue;

        cv::Rect boundingBox = cv::boundingRect(contour);

        if (boundingBox.y > outputImage.rows * 0.90)
            continue;

        double aspectRatio = (double)boundingBox.width / boundingBox.height;
        if (aspectRatio < 0.2 || aspectRatio > 3.0)
            continue;

        // Warning signs are convex diamonds; supplementary arrow plates can
        // merge into one blob — approximate the hull so vertex count stays stable.
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hullPerimeter = cv::arcLength(hull, true);
        double epsilon = 0.03 * hullPerimeter;
        std::vector<cv::Point> approxPolygon;
        cv::approxPolyDP(hull, approxPolygon, epsilon, true);

        int vertices = (int)approxPolygon.size();

        if (vertices >= 3 && vertices <= 10)
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

    double maxSignArea = outputImage.rows * outputImage.cols * 0.95;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area < 1000.0 || area > maxSignArea)
            continue;

        cv::Rect boundingBox = cv::boundingRect(contour);
        if (boundingBox.y > outputImage.rows * 0.85)
            continue;

        // FIX: Allow wide highway signs and tall parking signs
        double aspectRatio = (double)boundingBox.width / boundingBox.height;
        if (aspectRatio < 0.4 || aspectRatio > 4.0)
            continue;

        double perimeter = cv::arcLength(contour, true);
        double epsilon = 0.03 * perimeter;
        std::vector<cv::Point> approxPolygon;
        cv::approxPolyDP(contour, approxPolygon, epsilon, true);

        int vertices = (int)approxPolygon.size();

        if (vertices >= 4 && vertices <= 5)
        {
            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(255, 100, 0), 3);
            cv::putText(outputImage, "INFO SIGN",
                        cv::Point(boundingBox.x, boundingBox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 0), 2);
        }
    }
}