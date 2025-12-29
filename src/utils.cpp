# include <Eigen/Core>
# include <Eigen/Geometry>
# include <opencv2/opencv.hpp>
# include <opencv2/core/eigen.hpp>
# include <utility>
# include "utils.hpp"


std::pair<Eigen::Quaterniond, Eigen::Vector3d> to_eigen(
    const cv::Mat& rotation,
    const cv::Mat& translation
) {
    Eigen::Matrix3d rotation_eigen;
    cv::cv2eigen(rotation, rotation_eigen);

    Eigen::Vector3d translation_eigen;
    cv::cv2eigen(translation, translation_eigen);

    return {Eigen::Quaterniond(rotation_eigen), translation_eigen};
}

std::pair<cv::Mat, cv::Mat> from_eigen(
    const Eigen::Quaterniond& rotation,
    const Eigen::Vector3d& translation
) {
    cv::Mat rotation_mat;
    cv::eigen2cv(rotation.toRotationMatrix(), rotation_mat);

    cv::Mat translation_mat;
    cv::eigen2cv(translation, translation_mat);

    return {rotation_mat, translation_mat};
}

std::vector<cv::DMatch> funnel_matches(
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& mask
) {
    if (matches.size() != static_cast<size_t>(mask.rows)) {
        throw std::runtime_error("Mask size does not match vector size.");
    }
    std::vector<cv::DMatch> dest;

    for (size_t i = 0; i < matches.size(); ++i) {
        if (!mask.at<uchar>(i)) continue;

        dest.push_back(matches[i]);
    }
    return dest;
}

std::vector<cv::DMatch> select_matches(
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& inliers
) {
    std::vector<cv::DMatch> dest;

    dest.reserve(inliers.rows);

    for (size_t i = 0; i < inliers.rows; ++i) {
        dest.push_back(matches[inliers.at<int>(i)]);
    }
    return dest;
}

std::unordered_map<int, int> create_map_query(const std::vector<cv::DMatch>& matches) {
    std::unordered_map<int, int> map_query;

    for (size_t i = 0; i < matches.size(); ++i) {
        map_query.emplace(matches[i].queryIdx, static_cast<int>(i));
    }
    return map_query;
}

std::unordered_map<int, int> create_map_train(const std::vector<cv::DMatch>& matches) {
    std::unordered_map<int, int> map_train;

    for (size_t i = 0; i < matches.size(); ++i) {
        map_train.emplace(matches[i].trainIdx, static_cast<int>(i));
    }
    return map_train;
}
