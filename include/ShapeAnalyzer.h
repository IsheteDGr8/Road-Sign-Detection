/**
 * @file ShapeAnalyzer.h
 * @brief Declares the ShapeAnalyzer class responsible for geometric classification.
 * @author Manish & Ishaan
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class ShapeAnalyzer
{
public:
    ShapeAnalyzer() = default;

    /**
     * @brief Detects an octagonal stop sign from a binary mask and annotates the original image.
     * @param binaryMask The isolated black-and-white image from the ColorSegmenter.
     * @param outputImage The original BGR image where bounding boxes and labels will be drawn.
     * @pre binaryMask must be a single-channel 8-bit image (CV_8UC1).
     * @post Modifies outputImage by drawing a green bounding box and text if a stop sign is found.
     */
    void detectStopSign(const cv::Mat &binaryMask, cv::Mat &outputImage) const;
};