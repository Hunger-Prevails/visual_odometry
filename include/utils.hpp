# include <Eigen/Core>
# include <Eigen/Geometry>
# include <opencv2/opencv.hpp>
# include <opencv2/core/eigen.hpp>
# include <utility>


std::pair<Eigen::Quaterniond, Eigen::Vector3d> to_eigen(
    const cv::Mat& rotation,
    const cv::Mat& translation
);

std::pair<cv::Mat, cv::Mat> from_eigen(
    const Eigen::Quaterniond& rotation,
    const Eigen::Vector3d& translation
);

std::vector<cv::DMatch> funnel_matches(
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& mask
);

std::vector<cv::DMatch> select_matches(
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& inliers
);

std::unordered_map<int, int> create_map_query(const std::vector<cv::DMatch>& matches);
std::unordered_map<int, int> create_map_train(const std::vector<cv::DMatch>& matches, size_t offset = 0);

Eigen::Matrix3d to_essentials(
    const Eigen::Quaterniond& rotation_a,
    const Eigen::Quaterniond& rotation_b,
    const Eigen::Vector3d& translation_a,
    const Eigen::Vector3d& translation_b
);

Eigen::MatrixX3d to_homogeneous(std::vector<cv::Point2f>& points);

Eigen::VectorXd epipolar_products(
    const Eigen::Matrix3d& fundamentals,
    const Eigen::MatrixX3d& points_a,
    const Eigen::MatrixX3d& points_b
);
