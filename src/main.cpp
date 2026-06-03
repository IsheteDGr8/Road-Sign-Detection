#include <iostream>
#include "../include/ColorSegmenter.h"

int main()
{
    // Point this to your new yellow image
    std::string imagePath = "data/pedestrian_crossing_sign_2.jpg";

    ColorSegmenter segmenter;
    segmenter.tuneYellowMask(imagePath);

    return 0;
}