# include <opencv2/opencv.hpp>
# include <Eigen/Core>


class ReprojectionError {
protected:
    Eigen::Vector2d point;
    Eigen::Matrix3d intrinsics;

public:
    ReprojectionError(const cv::Point2f& point, const Eigen::Matrix3d& intrinsics);

    template <typename T>
    bool operator()(
        const T* const _rotation,
        const T* const _translation,
        const T* const _landmark,
        T* _residuals
    ) const;
};
