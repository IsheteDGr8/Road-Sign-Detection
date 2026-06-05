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
        if (aspectRatio < 0.3 || aspectRatio > 1.8)
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
            // Ask the Micro-Detector what icon is inside!
            int iconID = classifyWarningIcon(outputImage, boundingBox);

            std::string signLabel = "WARNING SIGN";
            if (iconID == 1)
                signLabel = "PEDESTRIAN";
            else if (iconID == 2)
                signLabel = "ANIMAL";
            else if (iconID == 3)
                signLabel = "BICYCLE";

            cv::rectangle(outputImage, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 255, 255), 3);
            cv::putText(outputImage, signLabel,
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

int ShapeAnalyzer::classifyWarningIcon(const cv::Mat &bgrImage, const cv::Rect &boundingBox) const
{
    // --- 1. THE PLAQUE CHOPPER ---
    cv::Rect strictBox = boundingBox;
    if (strictBox.height > strictBox.width * 1.1)
    {
        strictBox.height = strictBox.width;
    }

    // --- 2. GENTLE CROP ---
    // Dropped to just 5% to remove stray background pixels without chopping the icon
    int margin = strictBox.width * 0.05;
    cv::Rect innerBox(
        strictBox.x + margin,
        strictBox.y + margin,
        strictBox.width - (2 * margin),
        strictBox.height - (2 * margin));
    innerBox &= cv::Rect(0, 0, bgrImage.cols, bgrImage.rows);
    if (innerBox.width < 10 || innerBox.height < 10)
        return 0;

    // --- 3. ISOLATE DARK PIXELS ---
    cv::Mat roi = bgrImage(innerBox);
    cv::Mat hsvRoi, darkMask;
    cv::cvtColor(roi, hsvRoi, cv::COLOR_BGR2HSV);
    cv::inRange(hsvRoi, cv::Scalar(0, 0, 0), cv::Scalar(180, 140, 135), darkMask);

    double darkAreaRatio = (double)cv::countNonZero(darkMask) / (darkMask.rows * darkMask.cols);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(darkMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a brand new, empty mask to hold ONLY the true icons
    cv::Mat cleanMask = cv::Mat::zeros(darkMask.size(), CV_8UC1);
    cv::Rect combinedDarkBox;

    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Rect box = cv::boundingRect(contours[i]);
        if (box.area() < 15)
            continue;

        // --- THE BORDER ASSASSIN ---
        // If the black shape spans > 75% of the sign, it is the diagonal border ring. Ignore it!
        if (box.width > innerBox.width * 0.75 || box.height > innerBox.height * 0.75)
            continue;

        // Draw ONLY the valid internal icons onto our clean mask
        cv::drawContours(cleanMask, contours, (int)i, cv::Scalar(255), cv::FILLED);
        combinedDarkBox = combinedDarkBox.empty() ? box : (combinedDarkBox | box);
    }

    if (combinedDarkBox.empty())
        return 0;

    // --- 4. GEOMETRY MATH (Now 100% isolated from the border!) ---
    double shapeRatio = (double)combinedDarkBox.height / std::max(1, combinedDarkBox.width);
    double wideRatio = (double)combinedDarkBox.width / std::max(1, combinedDarkBox.height);

    // TEST 1: Pedestrian (Tall)
    if (shapeRatio > 1.3)
        return 1;

    // TEST 2: Animal (Wide)
    if (wideRatio > 1.2 && darkAreaRatio > 0.05)
        return 2;

    // TEST 3: Bicycle (Run HoughCircles on the CLEAN mask, not the raw dark mask)
    cv::Mat blurred;
    cv::GaussianBlur(cleanMask, blurred, cv::Size(7, 7), 1.5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1.2,
                     std::max(12, cleanMask.cols / 5), 100.0, 22.0,
                     std::max(3, cleanMask.cols / 20), std::max(8, cleanMask.cols / 4));

    if (circles.size() >= 2)
        return 3;

    return 0; // Generic Warning Sign
}