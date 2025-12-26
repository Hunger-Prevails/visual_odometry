# include <opencv2/opencv.hpp>
# include <Eigen/Core>
# include <ceres/ceres.h>


class ProjectionError {
protected:
    Eigen::Vector2d point;
    Eigen::Matrix3d intrinsics;

public:
    ProjectionError(const cv::Point2f& point, const Eigen::Matrix3d& intrinsics): point({point.x, point.y}), intrinsics(intrinsics) {}

    template <typename T>
    bool operator()(
        const T* const _landmark,
        T* _residuals
    ) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> landmark(_landmark);

        auto intrinsics = this->intrinsics.cast<T>();

        auto projection = (intrinsics * landmark).hnormalized();

        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(_residuals);

        auto point = this->point.cast<T>();

        residuals = projection - point;

        return true;
    }
};


class ProjectionErrorFull {
protected:
    Eigen::Vector2d point;
    Eigen::Matrix3d intrinsics;

public:
    ProjectionErrorFull(const cv::Point2f& point, const Eigen::Matrix3d& intrinsics): point({point.x, point.y}), intrinsics(intrinsics) {}

    template <typename T>
    bool operator()(
        const T* const _landmark,
        const T* const _rotation,
        const T* const _translation,
        T* _residuals
    ) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> landmark(_landmark);
        Eigen::Map<const Eigen::Quaternion<T>> rotation(_rotation);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(_translation);

        auto landmark_camera = rotation * landmark + translation;

        auto intrinsics = this->intrinsics.cast<T>();

        auto projection = (intrinsics * landmark_camera).hnormalized();

        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(_residuals);

        auto point = this->point.cast<T>();

        residuals = projection - point;

        return true;
    }
};
