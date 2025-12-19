# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>
# include "odometer.hpp"

Odometer::Odometer(Eigen::Matrix3f intrinsics): is_initialized(false), intrinsics(intrinsics) {}

void Odometer::initialize(const cv::Mat& frame_a, const cv::Mat& frame_b) {
    this->is_initialized = true;
}

void Odometer::processFrame(const cv::Mat& image) {
    if (!is_initialized) {
        // Initialization code here
        is_initialized = true;
    } else {
        // Frame processing code here
    }
}
