/*
 * main.cpp
 *
 * Purpose: Demo entry point — still-image tests by sign type, then dashcam video.
 * Authors: Ishaan, Manish
 *
 * Assumptions:
 *   - OpenCV 4 is installed; no other libraries or external executables.
 *   - Test images/videos live under data/ or build/data/ (see dataRoot()).
 *   - Working directory is the project root when launched from Visual Studio.
 */
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../include/ColorSegmenter.h"
#include "../include/ShapeAnalyzer.h"
namespace fs = std::filesystem;

static bool hasSignData(const fs::path &root)
{
    if (!fs::exists(root))
        return false;

    const char *folders[] = {"construction signs", "guide signs", "warning signs",
                             "service signs", "regulatory signs"};
    for (const char *name : folders)
    {
        if (fs::exists(root / name))
            return true;
    }

    return fs::exists(root / "dashcam.mp4");
}

static fs::path dataRoot()
{
    const fs::path candidates[] = {"data", "build/data"};
    for (const auto &candidate : candidates)
    {
        if (hasSignData(candidate))
            return candidate;
    }

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

static bool runDashcamVideo(const fs::path &videoPath, ColorSegmenter &segmenter, ShapeAnalyzer &analyzer,
                            int videoWidth)
{
    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened())
    {
        std::cerr << "Video failed to load: " << videoPath.string() << std::endl;
        return false;
    }

    cv::Mat frame, redMask, yellowMask, blueMask, whiteMask;
    const std::string windowTitle = "Video Demo - " + videoPath.filename().string();
    double fps = 0.0;

    while (true)
    {
        int64 startTick = cv::getTickCount();
        cap >> frame;

        if (frame.empty())
        {
            std::cout << "Finished: " << videoPath.filename().string() << std::endl;
            break;
        }

        cv::resize(frame, frame,
                   cv::Size(videoWidth, static_cast<int>(frame.rows * (static_cast<double>(videoWidth) / frame.cols))));

        redMask = segmenter.getStaticRedMask(frame);
        yellowMask = segmenter.getStaticYellowMask(frame);
        blueMask = segmenter.getStaticBlueMask(frame);
        whiteMask = segmenter.getStaticWhiteMask(frame);

        analyzer.detectRegulatorySigns(frame, redMask, whiteMask, frame);
        analyzer.detectWarningSign(yellowMask, frame);
        analyzer.detectServiceSign(blueMask, frame);

        int64 endTick = cv::getTickCount();
        fps = cv::getTickFrequency() / (endTick - startTick);

        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow(windowTitle, frame);

        if (cv::waitKey(30) == 27)
        {
            cap.release();
            cv::destroyWindow(windowTitle);
            return true;
        }
    }

    cap.release();
    cv::destroyWindow(windowTitle);
    return false;
}

static void runDashcamVideoTests(ColorSegmenter &segmenter, ShapeAnalyzer &analyzer, int videoWidth)
{
    const char *videoNames[] = {"dashcam.mp4"};
    bool foundAny = false;

    std::cout << "--- STARTING VIDEO DEMO ---" << std::endl;
    std::cout << "Press ESC on a video window to skip remaining clips." << std::endl;

    for (const char *name : videoNames)
    {
        const fs::path videoPath = dataRoot() / name;
        if (!fs::exists(videoPath))
        {
            std::cerr << "Skipping missing video: " << videoPath.string() << std::endl;
            continue;
        }

        foundAny = true;
        std::cout << "Playing: " << name << std::endl;
        if (runDashcamVideo(videoPath, segmenter, analyzer, videoWidth))
            break;
    }

    if (!foundAny)
    {
        std::cerr << "No dashcam videos found under " << dataRoot().string()
                  << "/ (expected dashcam.mp4)" << std::endl;
    }

    cv::destroyAllWindows();
}

// Purpose: Run folder-based sign demos, then play dashcam.mp4.
// Pre-condition: data/ or build/data/ contains sign folders or dashcam.mp4.
// Post-condition: Shows annotated images and video windows; returns 1 if no data found.
int main()
{
    ColorSegmenter segmenter;
    ShapeAnalyzer analyzer;

    const int STATIC_WIDTH = 400;
    const int VIDEO_WIDTH = 600;

    std::cout << "Using data folder: " << fs::absolute(dataRoot()).string() << std::endl;
    if (!hasSignData(dataRoot()))
    {
        std::cerr << "No sign images or videos found. Put test data under data/ or build/data/."
                  << std::endl;
        return 1;
    }

    runConstructionFolderTests(segmenter, analyzer, STATIC_WIDTH);
    runGuideFolderTests(segmenter, analyzer, STATIC_WIDTH);
    runServiceFolderTests(segmenter, analyzer, STATIC_WIDTH);
    runWarningFolderTests(segmenter, analyzer, STATIC_WIDTH);
    runRegulatoryFolderTests(segmenter, analyzer, STATIC_WIDTH);
    runDashcamVideoTests(segmenter, analyzer, VIDEO_WIDTH);

    return 0;
}
