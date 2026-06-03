/**
 * @file ColorSegmenter.cpp
 * @brief Implements HSV color masking and interactive tuning logic.
 * @author Ishaan
 */
#include "../include/ColorSegmenter.h"
#include <iostream>

namespace
{
    // --- RED CONTEXT (Dual Mask) ---
    struct TunerContext
    {
        cv::Mat hsvImage;
        int lowerH1 = 0, upperH1 = 10;
        int lowerH2 = 170, upperH2 = 180;
        int lowerS = 120, upperS = 255;
        int lowerV = 70, upperV = 255;
        int morphSize = 3;
    };

    void updateRedMask(int, void *userdata)
    {
        TunerContext *ctx = static_cast<TunerContext *>(userdata);
        cv::Mat mask1, mask2, combinedMask;

        cv::inRange(ctx->hsvImage, cv::Scalar(ctx->lowerH1, ctx->lowerS, ctx->lowerV), cv::Scalar(ctx->upperH1, ctx->upperS, ctx->upperV), mask1);
        cv::inRange(ctx->hsvImage, cv::Scalar(ctx->lowerH2, ctx->lowerS, ctx->lowerV), cv::Scalar(ctx->upperH2, ctx->upperS, ctx->upperV), mask2);
        cv::bitwise_or(mask1, mask2, combinedMask);

        int kSize = ctx->morphSize;
        if (kSize % 2 == 0)
            kSize += 1;
        if (kSize < 1)
            kSize = 1;

        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kSize, kSize));
        cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, element);
        cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_CLOSE, element);

        cv::imshow("Filtered Mask Result", combinedMask);
    }

    // --- YELLOW/BLUE CONTEXT (Single Mask) ---
    struct SingleColorContext
    {
        cv::Mat hsvImage;
        int lowerH = 0, upperH = 180;
        int lowerS = 100, upperS = 255;
        int lowerV = 100, upperV = 255;
        int morphSize = 3;
    };

    void updateSingleMask(int, void *userdata)
    {
        SingleColorContext *ctx = static_cast<SingleColorContext *>(userdata);
        cv::Mat mask;

        cv::inRange(ctx->hsvImage,
                    cv::Scalar(ctx->lowerH, ctx->lowerS, ctx->lowerV),
                    cv::Scalar(ctx->upperH, ctx->upperS, ctx->upperV),
                    mask);

        int kSize = ctx->morphSize;
        if (kSize % 2 == 0)
            kSize += 1;
        if (kSize < 1)
            kSize = 1;

        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kSize, kSize));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);

        cv::imshow("Filtered Mask Result", mask);
    }
}

// --- CLASS METHODS ---

void ColorSegmenter::tuneRedMask(const std::string &imagePath) const
{
    cv::Mat rawImage = cv::imread(imagePath);
    if (rawImage.empty())
        return;

    // Standardize the width to 600 pixels while keeping the correct aspect ratio
    int targetWidth = 600;
    int targetHeight = (int)(rawImage.rows * ((double)targetWidth / rawImage.cols));
    cv::resize(rawImage, rawImage, cv::Size(targetWidth, targetHeight));

    cv::Mat blurredImage;
    cv::GaussianBlur(rawImage, blurredImage, cv::Size(5, 5), 0);

    static TunerContext context;
    cv::cvtColor(blurredImage, context.hsvImage, cv::COLOR_BGR2HSV);

    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Red Control Panel", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Filtered Mask Result", cv::WINDOW_AUTOSIZE);

    cv::imshow("Original Image", rawImage);
    cv::createTrackbar("Lower Hue 1", "Red Control Panel", &context.lowerH1, 180, updateRedMask, &context);
    cv::createTrackbar("Upper Hue 1", "Red Control Panel", &context.upperH1, 180, updateRedMask, &context);
    cv::createTrackbar("Lower Hue 2", "Red Control Panel", &context.lowerH2, 180, updateRedMask, &context);
    cv::createTrackbar("Upper Hue 2", "Red Control Panel", &context.upperH2, 180, updateRedMask, &context);
    cv::createTrackbar("Min Sat", "Red Control Panel", &context.lowerS, 255, updateRedMask, &context);
    cv::createTrackbar("Max Sat", "Red Control Panel", &context.upperS, 255, updateRedMask, &context);
    cv::createTrackbar("Min Val", "Red Control Panel", &context.lowerV, 255, updateRedMask, &context);
    cv::createTrackbar("Max Val", "Red Control Panel", &context.upperV, 255, updateRedMask, &context);
    cv::createTrackbar("Morph", "Red Control Panel", &context.morphSize, 15, updateRedMask, &context);

    updateRedMask(0, &context);
    while (cv::waitKey(10) != 27)
    {
    }
    cv::destroyAllWindows();
}

void ColorSegmenter::tuneYellowMask(const std::string &imagePath) const
{
    cv::Mat rawImage = cv::imread(imagePath);
    if (rawImage.empty())
        return;

    // Standardize the width to 600 pixels while keeping the correct aspect ratio
    int targetWidth = 600;
    int targetHeight = (int)(rawImage.rows * ((double)targetWidth / rawImage.cols));
    cv::resize(rawImage, rawImage, cv::Size(targetWidth, targetHeight));

    cv::Mat blurredImage;
    cv::GaussianBlur(rawImage, blurredImage, cv::Size(5, 5), 0);

    static SingleColorContext context;
    // Pre-load estimated Yellow boundaries
    context.lowerH = 15;
    context.upperH = 35;
    cv::cvtColor(blurredImage, context.hsvImage, cv::COLOR_BGR2HSV);

    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Yellow Control Panel", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Filtered Mask Result", cv::WINDOW_AUTOSIZE);

    cv::imshow("Original Image", rawImage);
    cv::createTrackbar("Lower Hue", "Yellow Control Panel", &context.lowerH, 180, updateSingleMask, &context);
    cv::createTrackbar("Upper Hue", "Yellow Control Panel", &context.upperH, 180, updateSingleMask, &context);
    cv::createTrackbar("Min Sat", "Yellow Control Panel", &context.lowerS, 255, updateSingleMask, &context);
    cv::createTrackbar("Max Sat", "Yellow Control Panel", &context.upperS, 255, updateSingleMask, &context);
    cv::createTrackbar("Min Val", "Yellow Control Panel", &context.lowerV, 255, updateSingleMask, &context);
    cv::createTrackbar("Max Val", "Yellow Control Panel", &context.upperV, 255, updateSingleMask, &context);
    cv::createTrackbar("Morph", "Yellow Control Panel", &context.morphSize, 15, updateSingleMask, &context);

    updateSingleMask(0, &context);
    while (cv::waitKey(10) != 27)
    {
    }
    cv::destroyAllWindows();
}

void ColorSegmenter::tuneBlueMask(const std::string &imagePath) const
{
    cv::Mat rawImage = cv::imread(imagePath);
    if (rawImage.empty())
        return;

    // Standardize the width to 600 pixels while keeping the correct aspect ratio
    int targetWidth = 600;
    int targetHeight = (int)(rawImage.rows * ((double)targetWidth / rawImage.cols));
    cv::resize(rawImage, rawImage, cv::Size(targetWidth, targetHeight));

    cv::Mat blurredImage;
    cv::GaussianBlur(rawImage, blurredImage, cv::Size(5, 5), 0);

    static SingleColorContext context;
    // Pre-load estimated Blue boundaries
    context.lowerH = 90;
    context.upperH = 130;
    cv::cvtColor(blurredImage, context.hsvImage, cv::COLOR_BGR2HSV);

    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Blue Control Panel", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Filtered Mask Result", cv::WINDOW_AUTOSIZE);

    cv::imshow("Original Image", rawImage);
    cv::createTrackbar("Lower Hue", "Blue Control Panel", &context.lowerH, 180, updateSingleMask, &context);
    cv::createTrackbar("Upper Hue", "Blue Control Panel", &context.upperH, 180, updateSingleMask, &context);
    cv::createTrackbar("Min Sat", "Blue Control Panel", &context.lowerS, 255, updateSingleMask, &context);
    cv::createTrackbar("Max Sat", "Blue Control Panel", &context.upperS, 255, updateSingleMask, &context);
    cv::createTrackbar("Min Val", "Blue Control Panel", &context.lowerV, 255, updateSingleMask, &context);
    cv::createTrackbar("Max Val", "Blue Control Panel", &context.upperV, 255, updateSingleMask, &context);
    cv::createTrackbar("Morph", "Blue Control Panel", &context.morphSize, 15, updateSingleMask, &context);

    updateSingleMask(0, &context);
    while (cv::waitKey(10) != 27)
    {
    } // 27 is the ESC key
    cv::destroyAllWindows();
}

cv::Mat ColorSegmenter::getStaticRedMask(const cv::Mat &inputImage) const
{
    cv::Mat blurredImage, hsvImage;
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(5, 5), 0);
    cv::cvtColor(blurredImage, hsvImage, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2, combinedMask;
    cv::inRange(hsvImage, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsvImage, cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255), mask2);
    cv::bitwise_or(mask1, mask2, combinedMask);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, element);
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_CLOSE, element);

    return combinedMask;
}

cv::Mat ColorSegmenter::getStaticYellowMask(const cv::Mat &inputImage) const
{
    cv::Mat blurredImage, hsvImage;
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(5, 5), 0);
    cv::cvtColor(blurredImage, hsvImage, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    // Values extracted from pedestrian crossing tuning
    cv::inRange(hsvImage, cv::Scalar(15, 100, 100), cv::Scalar(35, 255, 255), mask);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);

    return mask;
}

cv::Mat ColorSegmenter::getStaticBlueMask(const cv::Mat &inputImage) const
{
    cv::Mat blurredImage, hsvImage;
    cv::GaussianBlur(inputImage, blurredImage, cv::Size(5, 5), 0);
    cv::cvtColor(blurredImage, hsvImage, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    // Values extracted from disabled parking tuning (Lower Hue corrected to 90)
    cv::inRange(hsvImage, cv::Scalar(90, 61, 87), cv::Scalar(130, 255, 255), mask);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);

    return mask;
}