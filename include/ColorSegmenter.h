/**
 * @file ColorSegmenter.h
 * @brief Declares the ColorSegmenter class responsible for HSV masking.
 * @author Ishaan
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class ColorSegmenter
{
public:
    // Default constructor
    ColorSegmenter() = default;

    /**
     * @brief Launches an interactive UI to tune red HSV thresholds.
     * @param imagePath Path to the test image.
     * @pre The imagePath must point to a valid image file.
     * @post Opens OpenCV windows. Blocks execution until 'ESC' or 'q' is pressed.
     */
    void tuneRedMask(const std::string &imagePath) const;
    void tuneYellowMask(const std::string &imagePath) const;
    void tuneBlueMask(const std::string &imagePath) const;

    /**
     * @brief Generates a binary mask using hardcoded red HSV values.
     * @param inputImage The raw BGR image.
     * @return cv::Mat The binary mask isolating red areas.
     */
    cv::Mat getStaticRedMask(const cv::Mat &inputImage) const;
};