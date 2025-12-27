# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>
# include "feature_extractor.hpp"

Extractor::Extractor(int count_features) {
    sift = cv::SIFT::create(count_features, 3, 0.04, 10, 1.6, true);
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> Extractor::extract(cv::Mat& image) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    return {keypoints, descriptors};
}
