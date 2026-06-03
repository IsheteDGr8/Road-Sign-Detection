/**
 * @file ColorSegmenter.cpp
 * @brief Implements HSV color masking and interactive tuning logic.
 * @author Ishaan
 */
#include "../include/ColorSegmenter.h"
#include <iostream>

// Anonymous namespace to hide internal structs and callbacks from the global scope
namespace
{
    struct TunerContext
    {
        cv::Mat hsvImage;
        int lowerH1 = 0, upperH1 = 10;
        int lowerH2 = 170, upperH2 = 180;
        int lowerS = 120, upperS = 255;
        int lowerV = 70, upperV = 255;
        int morphSize = 3;
    };

    // Callback function for the trackbars
    void updateMask(int, void *userdata)
    {
        TunerContext *ctx = static_cast<TunerContext *>(userdata);
        cv::Mat mask1, mask2, combinedMask;

        // Lower and upper red wraps
        cv::inRange(ctx->hsvImage, cv::Scalar(ctx->lowerH1, ctx->lowerS, ctx->lowerV), cv::Scalar(ctx->upperH1, ctx->upperS, ctx->upperV), mask1);
        cv::inRange(ctx->hsvImage, cv::Scalar(ctx->lowerH2, ctx->lowerS, ctx->lowerV), cv::Scalar(ctx->upperH2, ctx->upperS, ctx->upperV), mask2);

        // Combine
        cv::bitwise_or(mask1, mask2, combinedMask);

        // Morphological cleanup
        int kSize = ctx->morphSize;
        if (kSize % 2 == 0)
            kSize += 1; // Enforce odd kernel size
        if (kSize < 1)
            kSize = 1;

        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kSize, kSize));
        cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, element);
        cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_CLOSE, element);

        cv::imshow("Filtered Mask Result", combinedMask);
    }
}

void ColorSegmenter::tuneRedMask(const std::string &imagePath) const
{
    cv::Mat rawImage = cv::imread(imagePath);

    if (rawImage.empty())
    {
        std::cerr << "Error: Could not load image from " << imagePath << std::endl;
        return;
    }

    cv::Mat blurredImage;
    cv::GaussianBlur(rawImage, blurredImage, cv::Size(5, 5), 0);

    // Static so it persists in memory for the callback to access
    static TunerContext context;
    cv::cvtColor(blurredImage, context.hsvImage, cv::COLOR_BGR2HSV);

    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Red Mask Control Panel", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Filtered Mask Result", cv::WINDOW_AUTOSIZE);

    cv::imshow("Original Image", rawImage);

    cv::createTrackbar("Lower Hue 1", "Red Mask Control Panel", &context.lowerH1, 180, updateMask, &context);
    cv::createTrackbar("Upper Hue 1", "Red Mask Control Panel", &context.upperH1, 180, updateMask, &context);
    cv::createTrackbar("Lower Hue 2", "Red Mask Control Panel", &context.lowerH2, 180, updateMask, &context);
    cv::createTrackbar("Upper Hue 2", "Red Mask Control Panel", &context.upperH2, 180, updateMask, &context);
    cv::createTrackbar("Min Saturation", "Red Mask Control Panel", &context.lowerS, 255, updateMask, &context);
    cv::createTrackbar("Max Saturation", "Red Mask Control Panel", &context.upperS, 255, updateMask, &context);
    cv::createTrackbar("Min Brightness (V)", "Red Mask Control Panel", &context.lowerV, 255, updateMask, &context);
    cv::createTrackbar("Max Brightness (V)", "Red Mask Control Panel", &context.upperV, 255, updateMask, &context);
    cv::createTrackbar("Morph Kernel", "Red Mask Control Panel", &context.morphSize, 15, updateMask, &context);

    updateMask(0, &context);

    std::cout << "Press 'ESC' or 'q' in any image window to exit." << std::endl;

    while (true)
    {
        char key = (char)cv::waitKey(10);
        if (key == 27 || key == 'q' || key == 'Q')
            break;
    }
    cv::destroyAllWindows();
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