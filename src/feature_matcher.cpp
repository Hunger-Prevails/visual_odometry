# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>
# include <vector>
# include "feature_matcher.hpp"

Matcher::Matcher() {
    matcher = cv::FlannBasedMatcher::create();
}

std::vector<cv::DMatch> Matcher::match(cv::Mat& descriptors_a, cv::Mat& descriptors_b) {
    std::vector<cv::DMatch> matches;

    if (descriptors_a.type() != CV_32F) {
        descriptors_a.convertTo(descriptors_a, CV_32F);
    }
    if (descriptors_b.type() != CV_32F) {
        descriptors_b.convertTo(descriptors_b, CV_32F);
    }
    matcher->match(descriptors_a, descriptors_b, matches);

    return matches;
}

std::vector<cv::DMatch> Matcher::match_knn(cv::Mat& descriptors_a, cv::Mat& descriptors_b, int n_neighbors, float ratio_threshold) {
    std::vector<std::vector<cv::DMatch>> matches_knn;

    if (descriptors_a.type() != CV_32F) {
        descriptors_a.convertTo(descriptors_a, CV_32F);
    }
    if (descriptors_b.type() != CV_32F) {
        descriptors_b.convertTo(descriptors_b, CV_32F);
    }
    matcher->knnMatch(descriptors_a, descriptors_b, matches_knn, n_neighbors);

    std::cout << "Found " << matches_knn.size() << " initial matches" << std::endl;

    std::vector<cv::DMatch> matches;

    for (size_t i = 0; i < matches_knn.size(); i++) {
        if (matches_knn[i].size() == n_neighbors) {
            float d1 = matches_knn[i][0].distance;
            float d2 = matches_knn[i][1].distance;

            if (d1 / d2 < ratio_threshold) {
                matches.push_back(matches_knn[i][0]);
            }
        }
    }
    std::cout << "To keep " << matches.size() << " good matches after ratio test" << std::endl;

    return matches;
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
