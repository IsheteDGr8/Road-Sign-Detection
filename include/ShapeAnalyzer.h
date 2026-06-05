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

    /**
     * @brief Classifies regulatory signs: stop, do-not-enter, speed limit, and generic.
     * @param bgrImage Original color frame used for OCR and inner-sign checks.
     * @param redMask Red segmentation mask.
     * @param whiteMask White segmentation mask.
     * @param outputImage Annotated output image.
     */
    void detectRegulatorySigns(const cv::Mat &bgrImage, const cv::Mat &redMask,
                               const cv::Mat &whiteMask, cv::Mat &outputImage) const;
    void detectWarningSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
    void detectConstructionSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
    void detectGuideSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
    void detectServiceSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
};