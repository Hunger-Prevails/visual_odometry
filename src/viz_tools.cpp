# include <vector>
# include <unordered_map>
# include <filesystem>
# include <opencv2/opencv.hpp>
# include <Eigen/Core>
# include <Eigen/Geometry>
# include "viz_tools.hpp"

namespace fs = std::filesystem;

cv::Point2f eigen2cv(const Eigen::Vector2d& point) {
    return {static_cast<float>(point.x()), static_cast<float>(point.y())};
}

void paint_matches(
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

void paint_projections(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<Eigen::Vector3d>& landmarks,
    const std::unordered_map<int, int>& feature_to_landmark,
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Quaterniond& rotation,
    const Eigen::Vector3d& translation,
    const fs::path& write_path
) {
    cv::Mat dest = image.clone();

    for (auto [feature, landmark]: feature_to_landmark) {
        auto landmark_camera = rotation * landmarks[landmark] + translation;

        auto projection = eigen2cv((intrinsics * landmark_camera).hnormalized());

        cv::circle(dest, keypoints[feature].pt, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(dest, projection, 3, cv::Scalar(255, 0, 0), -1);

        cv::line(dest, keypoints[feature].pt, projection, cv::Scalar(255, 255, 0), 1);
    }
    cv::imwrite(write_path.string(), dest);
}
