# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>
# include "odometer.hpp"

Odometer::Odometer(Eigen::Matrix3f intrinsics): is_initialized(false), intrinsics(intrinsics) {}

void Odometer::initialize(const cv::Mat& frame_a, const cv::Mat& frame_b) {
    is_initialized = true;
    rotations.push_back(Eigen::Quaternionf::Identity());
    translations.push_back(Eigen::Vector3f::Zero());
    rotations.push_back(Eigen::Quaternionf::Identity());
    translations.push_back(Eigen::Vector3f::Random());
}

void Odometer::processFrame(const cv::Mat& image) {
    if (!is_initialized) {
        throw std::runtime_error("Call initialize() with two frames before processing frames.");
    }
    rotations.push_back(Eigen::Quaternionf::Identity());
    translations.push_back(Eigen::Vector3f::Zero());
}

const std::vector<Eigen::Quaternionf>& Odometer::getRotations() {
    return rotations;
}

const std::vector<Eigen::Vector3f>& Odometer::getTranslations() {
    return translations;
}
