# include <vector>
# include <unordered_map>
# include <filesystem>
# include <opencv2/opencv.hpp>
# include <Eigen/Core>
# include <Eigen/Geometry>

namespace fs = std::filesystem;

cv::Point2f eigen2cv(const Eigen::Vector2d& point);

void paint_matches(
    const cv::Mat& image_a,
    const cv::Mat& image_b,
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches,
    const fs::path& write_path
);

void paint_projections(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<Eigen::Vector3d>& landmarks,
    const std::unordered_map<int, int>& keypoint_to_landmark,
    const Eigen::Matrix3d& intrinsics,
    const Eigen::Quaterniond& rotation,
    const Eigen::Vector3d& translation,
    const fs::path& write_path
);
