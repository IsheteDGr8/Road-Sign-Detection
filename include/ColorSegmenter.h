// ColorSegmenter.h
// Pulls sign colors out of a frame using HSV (and grayscale for white panels).
// Author: Ishaan
#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class ColorSegmenter
{
public:
    ColorSegmenter() = default;

    // Sliders for tuning thresholds on a still image. Hit ESC to close.
    void tuneRedMask(const std::string &imagePath) const;
    void tuneYellowMask(const std::string &imagePath) const;
    void tuneBlueMask(const std::string &imagePath) const;

    // Fixed thresholds used by the detector at runtime.
    cv::Mat getStaticRedMask(const cv::Mat &inputImage) const;
    cv::Mat getStaticYellowMask(const cv::Mat &inputImage) const;
    cv::Mat getStaticBlueMask(const cv::Mat &inputImage) const;
    cv::Mat getStaticOrangeMask(const cv::Mat &inputImage) const;
    cv::Mat getStaticGreenMask(const cv::Mat &inputImage) const;
    cv::Mat getStaticWhiteMask(const cv::Mat &inputImage) const;
};
