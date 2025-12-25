# include <opencv2/opencv.hpp>
# include <Eigen/Core>
# include "cost_functions.hpp"

ReprojectionError::ReprojectionError(const cv::Point2f& point, const Eigen::Matrix3d& intrinsics): point({point.x, point.y}), intrinsics(intrinsics) {}

template <typename T>
bool ReprojectionError::operator()(
    const T* const _rotation,
    const T* const _translation,
    const T* const _landmark,
    T* _residuals
) const {
    Eigen::Map<const Eigen::Quaternion<T>> rotation(_rotation);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(_translation);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> landmark(_landmark);

    auto landmark_camera = rotation * landmark + translation;

    auto intrinsics = this->intrinsics.cast<T>();

    auto projection = intrinsics * landmark_camera.hnormalized();

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(_residuals);

    auto point = this->point.cast<T>();

    residuals = projection - point;

    return true;
}
