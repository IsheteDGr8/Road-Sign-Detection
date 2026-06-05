// ShapeAnalyzer.h
// Looks at color masks, finds sign-shaped blobs, and labels them.
// Author: Ishaan
#pragma once
#include <opencv2/opencv.hpp>

class ShapeAnalyzer
{
public:
    ShapeAnalyzer() = default;

    void detectRegulatorySigns(const cv::Mat &bgrImage, const cv::Mat &redMask,
                               const cv::Mat &whiteMask, cv::Mat &outputImage) const;
    void detectWarningSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
    void detectConstructionSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
    void detectGuideSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
    void detectServiceSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
};
