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
     * @brief Detects octagons (Stop) and circles (Do Not Enter) from a red binary mask.
     * @param binaryMask The isolated black-and-white image from the ColorSegmenter.
     * @param outputImage The original BGR image where bounding boxes and labels will be drawn.
     */
    void detectRedSigns(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
    void detectWarningSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
    void detectInfoSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
};