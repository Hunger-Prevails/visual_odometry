# pragma once
# include <vector>
# include <opencv2/opencv.hpp>

class Keyframe {
public:
    size_t frame;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    std::map<int, int> feature_to_landmark;
};
