/**
 * @file ShapeAnalyzer.h
 * @brief Declares the ShapeAnalyzer class responsible for geometric classification.
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class ShapeAnalyzer
{
public:
    ShapeAnalyzer() = default;

    void detectStopSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;

    // New Methods for 4-sided shapes
    void detectWarningSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
    void detectInfoSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
};