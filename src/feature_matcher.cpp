# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>
# include "feature_matcher.hpp"

Matcher::Matcher() {
    matcher = cv::FlannBasedMatcher::create();
}

void Matcher::match(cv::Mat& descriptors_a, cv::Mat& descriptors_b, std::vector<cv::DMatch>& matches) {
    matcher->match(descriptors_a, descriptors_b, matches);
}
