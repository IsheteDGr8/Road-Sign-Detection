#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../include/ColorSegmenter.h"
#include "../include/ShapeAnalyzer.h"

namespace fs = std::filesystem;

static fs::path dataRoot()
{
    const fs::path candidates[] = {"data", "build/data"};
    for (const auto &candidate : candidates)
    {
        if (fs::exists(candidate))
            return candidate;
    }
    return "data";
}

static bool isImageFile(const fs::path &path)
{
    if (!path.has_extension())
        return false;
    std::string ext = path.extension().string();
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

static void resizeStatic(cv::Mat &frame, int staticWidth)
{
    int staticHeight = static_cast<int>(frame.rows * (static_cast<double>(staticWidth) / frame.cols));
    cv::resize(frame, frame, cv::Size(staticWidth, staticHeight));
}

static bool isTargetRegulatoryImage(const fs::path &path)
{
    std::string stem = path.stem().string();
    return stem.rfind("stop_", 0) == 0 || stem.rfind("no_entry_", 0) == 0 ||
           stem.rfind("speed_limit_", 0) == 0;
}

static void showStaticFrame(const std::string &windowTitle, const cv::Mat &frame)
{
    const int topPad = 50;
    const int displayWidth = 800;

    cv::Mat padded;
    cv::copyMakeBorder(frame, padded, topPad, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(30, 30, 30));

    int displayHeight = static_cast<int>(padded.rows * (static_cast<double>(displayWidth) / padded.cols));
    cv::Mat display;
    cv::resize(padded, display, cv::Size(displayWidth, displayHeight));

    cv::namedWindow(windowTitle, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowTitle, displayWidth, displayHeight);
    cv::imshow(windowTitle, display);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

static void runRegulatoryFolderTests(ColorSegmenter &segmenter, ShapeAnalyzer &analyzer, int staticWidth)
{
    const fs::path folder = dataRoot() / "regulatory signs";
    if (!fs::exists(folder))
    {
        std::cerr << "Skipping regulatory tests — folder not found: " << folder.string() << std::endl;
        return;
    }

    std::cout << "--- REGULATORY SIGNS: STOP / DO NOT ENTER / SPEED LIMIT (press any key) ---"
              << std::endl;
    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (!entry.is_regular_file() || !isImageFile(entry.path()) ||
            !isTargetRegulatoryImage(entry.path()))
            continue;

        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty())
        {
            std::cerr << "Failed to load: " << entry.path().string() << std::endl;
            continue;
        }

        resizeStatic(image, staticWidth);
        cv::Mat redMask = segmenter.getStaticRedMask(image);
        cv::Mat whiteMask = segmenter.getStaticWhiteMask(image);
        analyzer.detectRegulatorySigns(image, redMask, whiteMask, image);

        std::string windowTitle = "Regulatory - " + entry.path().filename().string();
        std::cout << windowTitle << std::endl;
        showStaticFrame(windowTitle, image);
    }
}

static void runConstructionFolderTests(ColorSegmenter &segmenter, ShapeAnalyzer &analyzer, int staticWidth)
{
    const fs::path folder = dataRoot() / "construction signs";
    if (!fs::exists(folder))
    {
        std::cerr << "Skipping construction tests — folder not found: " << folder.string() << std::endl;
        return;
    }

    std::cout << "--- CONSTRUCTION SIGNS (press any key for next) ---" << std::endl;
    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (!entry.is_regular_file() || !isImageFile(entry.path()))
            continue;

        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty())
        {
            std::cerr << "Failed to load: " << entry.path().string() << std::endl;
            continue;
        }

        resizeStatic(image, staticWidth);
        cv::Mat orangeMask = segmenter.getStaticOrangeMask(image);
        analyzer.detectConstructionSign(orangeMask, image);

        std::string windowTitle = "Construction - " + entry.path().filename().string();
        std::cout << windowTitle << std::endl;
        showStaticFrame(windowTitle, image);
    }
}

static void runServiceFolderTests(ColorSegmenter &segmenter, ShapeAnalyzer &analyzer, int staticWidth)
{
    const fs::path folder = dataRoot() / "service signs";
    if (!fs::exists(folder))
    {
        std::cerr << "Skipping service tests — folder not found: " << folder.string() << std::endl;
        return;
    }

    std::cout << "--- SERVICE SIGNS (press any key for next) ---" << std::endl;
    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (!entry.is_regular_file() || !isImageFile(entry.path()))
            continue;

        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty())
        {
            std::cerr << "Failed to load: " << entry.path().string() << std::endl;
            continue;
        }

        resizeStatic(image, staticWidth);
        cv::Mat blueMask = segmenter.getStaticBlueMask(image);
        analyzer.detectServiceSign(blueMask, image);

        std::string windowTitle = "Service - " + entry.path().filename().string();
        std::cout << windowTitle << std::endl;
        showStaticFrame(windowTitle, image);
    }
}

static void runWarningFolderTests(ColorSegmenter &segmenter, ShapeAnalyzer &analyzer, int staticWidth)
{
    const fs::path folder = dataRoot() / "warning signs";
    if (!fs::exists(folder))
    {
        std::cerr << "Skipping warning tests — folder not found: " << folder.string() << std::endl;
        return;
    }

    std::cout << "--- WARNING SIGNS (press any key for next) ---" << std::endl;
    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (!entry.is_regular_file() || !isImageFile(entry.path()))
            continue;

        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty())
        {
            std::cerr << "Failed to load: " << entry.path().string() << std::endl;
            continue;
        }

        resizeStatic(image, staticWidth);
        cv::Mat yellowMask = segmenter.getStaticYellowMask(image);
        analyzer.detectWarningSign(yellowMask, image);

        std::string windowTitle = "Warning - " + entry.path().filename().string();
        std::cout << windowTitle << std::endl;
        showStaticFrame(windowTitle, image);
    }
}

static void runGuideFolderTests(ColorSegmenter &segmenter, ShapeAnalyzer &analyzer, int staticWidth)
{
    const fs::path folder = dataRoot() / "guide signs";
    if (!fs::exists(folder))
    {
        std::cerr << "Skipping guide tests — folder not found: " << folder.string() << std::endl;
        return;
    }

    std::cout << "--- GUIDE SIGNS (press any key for next) ---" << std::endl;
    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (!entry.is_regular_file() || !isImageFile(entry.path()))
            continue;

        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty())
        {
            std::cerr << "Failed to load: " << entry.path().string() << std::endl;
            continue;
        }

        resizeStatic(image, staticWidth);
        cv::Mat greenMask = segmenter.getStaticGreenMask(image);
        analyzer.detectGuideSign(greenMask, image);

        std::string windowTitle = "Guide - " + entry.path().filename().string();
        std::cout << windowTitle << std::endl;
        showStaticFrame(windowTitle, image);
    }
}

int main()
{
    ColorSegmenter segmenter;
    ShapeAnalyzer analyzer;
    cv::Mat frame, redMask, yellowMask, blueMask, whiteMask;

    const int STATIC_WIDTH = 400;
    const int VIDEO_WIDTH = 600;

    runConstructionFolderTests(segmenter, analyzer, STATIC_WIDTH);
    runGuideFolderTests(segmenter, analyzer, STATIC_WIDTH);
    runServiceFolderTests(segmenter, analyzer, STATIC_WIDTH);
    runWarningFolderTests(segmenter, analyzer, STATIC_WIDTH);
    runRegulatoryFolderTests(segmenter, analyzer, STATIC_WIDTH);

    std::cout << "--- STARTING VIDEO DEMO ---" << std::endl;
    std::cout << "Press ESC on the video window to exit." << std::endl;

    const std::string dashcamPath = (dataRoot() / "dashcam.mp4").string();
    cv::VideoCapture cap(dashcamPath);
    if (!cap.isOpened())
    {
        std::cerr << "Video failed to load: " << dashcamPath << std::endl;
        std::cerr << "Place dashcam.mp4 under data/ (project root) and run from the project folder." << std::endl;
        return -1;
    }

    double fps = 0.0;
    while (true)
    {
        int64 startTick = cv::getTickCount();
        cap >> frame;

        if (frame.empty())
        {
            std::cout << "Video finished." << std::endl;
            break;
        }

        cv::resize(frame, frame, cv::Size(VIDEO_WIDTH, (int)(frame.rows * ((double)VIDEO_WIDTH / frame.cols))));

        redMask = segmenter.getStaticRedMask(frame);
        yellowMask = segmenter.getStaticYellowMask(frame);
        blueMask = segmenter.getStaticBlueMask(frame);
        whiteMask = segmenter.getStaticWhiteMask(frame);

        analyzer.detectRegulatorySigns(frame, redMask, whiteMask, frame);
        analyzer.detectWarningSign(yellowMask, frame);
        analyzer.detectServiceSign(blueMask, frame);

        int64 endTick = cv::getTickCount();
        fps = cv::getTickFrequency() / (endTick - startTick);

        cv::putText(frame, "FPS: " + std::to_string((int)fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Video Demo - Live Dashcam", frame);

        if (cv::waitKey(30) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
