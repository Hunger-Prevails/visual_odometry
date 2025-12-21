# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>
# include "feature_matcher.hpp"

Matcher::Matcher() {
    matcher = cv::FlannBasedMatcher::create();
}

void Matcher::match(cv::Mat& descriptors_a, cv::Mat& descriptors_b, std::vector<cv::DMatch>& matches) {
    if (descriptors_a.type() != CV_32F) {
        descriptors_a.convertTo(descriptors_a, CV_32F);
    }
    if (descriptors_b.type() != CV_32F) {
        descriptors_b.convertTo(descriptors_b, CV_32F);
    }
    matcher->match(descriptors_a, descriptors_b, matches);
}

 void Matcher::paint_matches(
    const cv::Mat& image_a,
    const cv::Mat& image_b,
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches,
    const fs::path& write_path
) {
    cv::Mat dest;
    cv::drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        matches,
        dest,
        cv::Scalar(0, 128, 0),
        cv::Scalar(128, 0, 0),
        std::vector<char>(),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );
    cv::imwrite(write_path.string(), dest);
}
