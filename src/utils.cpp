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

Eigen::MatrixX2d to_eigen(const std::vector<cv::Point2f>& landmarks) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>> map(
        reinterpret_cast<const float*>(landmarks.data()), landmarks.size(), 2
    );

    return map.cast<double>();
}

Eigen::MatrixX3d to_eigen(const std::vector<cv::Point3f>& landmarks) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> map(
        reinterpret_cast<const float*>(landmarks.data()), landmarks.size(), 3
    );

    return map.cast<double>();
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

std::vector<cv::Point2f> from_eigen(const Eigen::MatrixX2d& keypoints) {
    std::vector<cv::Point2f> points(keypoints.rows());

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>> map(
        reinterpret_cast<float*>(points.data()), keypoints.rows(), 2
    );

    map = keypoints.cast<float>();

    return points;
}

std::vector<cv::Point3f> from_eigen(const Eigen::MatrixX3d& landmarks) {
    std::vector<cv::Point3f> points(landmarks.rows());

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> map(
        reinterpret_cast<float*>(points.data()), landmarks.rows(), 3
    );

    map = landmarks.cast<float>();

    return points;
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

std::pair<std::vector<cv::DMatch>, std::vector<cv::DMatch>> select_matches(
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& inliers
) {
    std::vector<bool> mask(matches.size(), false);

    for (size_t i = 0; i < inliers.rows; ++i) {
        mask[inliers.at<int>(i)] = true;
    }

    std::vector<cv::DMatch> matches_inliers;
    std::vector<cv::DMatch> matches_outliers;

    matches_inliers.reserve(inliers.rows);
    matches_outliers.reserve(matches.size() - inliers.rows);

    for (size_t i = 0; i < matches.size(); ++i) {
        if (mask[i]) {
            matches_inliers.push_back(matches[i]);
        } else {
            matches_outliers.push_back(matches[i]);
        }
    }
    return {matches_inliers, matches_outliers};
}

std::unordered_map<int, int> create_map_query(const std::vector<cv::DMatch>& matches, const size_t offset) {
    std::unordered_map<int, int> map_query;

    for (size_t i = 0; i < matches.size(); ++i) {
        map_query.emplace(matches[i].queryIdx, static_cast<int>(i + offset));
    }
    return map_query;
}

std::unordered_map<int, int> create_map_train(const std::vector<cv::DMatch>& matches, const size_t offset) {
    std::unordered_map<int, int> map_train;

    for (size_t i = 0; i < matches.size(); ++i) {
        map_train.emplace(matches[i].trainIdx, static_cast<int>(i + offset));
    }
    return map_train;
}

Eigen::Matrix3d to_essentials(
    const Eigen::Quaterniond& rotation_a,
    const Eigen::Quaterniond& rotation_b,
    const Eigen::Vector3d& translation_a,
    const Eigen::Vector3d& translation_b
) {
    auto rotation = rotation_b * rotation_a.conjugate();

    auto translation = translation_b - rotation * translation_a;

    auto rotation_matrix = rotation.toRotationMatrix();

    Eigen::Matrix3d essentials;

    essentials.col(0) = translation.cross(rotation_matrix.col(0));
    essentials.col(1) = translation.cross(rotation_matrix.col(1));
    essentials.col(2) = translation.cross(rotation_matrix.col(2));

    return essentials;
}

Eigen::MatrixX2d hnormalize(const Eigen::MatrixX3d& points) {
    return (
        points.leftCols<2>().array().colwise() / points.col(2).array()
    ).matrix();
}

Eigen::MatrixX3d to_homogeneous(std::vector<cv::Point2f>& points) {
    Eigen::MatrixXd homogeneous = Eigen::MatrixXd::Ones(points.size(), 3);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>> points_eigen(reinterpret_cast<float*>(points.data()), points.size(), 2);

    homogeneous.leftCols<2>() = points_eigen.cast<double>();

    return homogeneous;
}

Eigen::VectorXd epipolar_products(
    const Eigen::Matrix3d& fundamentals,
    const Eigen::MatrixX3d& points_a,
    const Eigen::MatrixX3d& points_b
) {
    auto lines = (fundamentals * points_a.transpose()).transpose();

    auto norms = lines.leftCols<2>().rowwise().norm();

    return ((points_b.array() * lines.array()).rowwise().sum().abs() / norms.array()).matrix();
}

float compute_median_pixel_motion(
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b,
    const std::vector<cv::DMatch>& matches
) {
    std::vector<float> distances;

    distances.reserve(matches.size());

    for (const auto& match: matches) {
        auto& point_a = keypoints_a[match.queryIdx].pt;
        auto& point_b = keypoints_b[match.trainIdx].pt;

        distances.push_back(cv::norm(point_a - point_b));
    }
    std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());

    return distances[distances.size() / 2];
}
