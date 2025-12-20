# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>


class Odometer {
protected:
    bool is_initialized;
    Eigen::Matrix3f intrinsics;

    std::vector<Eigen::Vector4f> rotations;
    std::vector<Eigen::Vector3f> translations;

public:
    Odometer(Eigen::Matrix3f intrinsics);

    void initialize(const cv::Mat& frame_a, const cv::Mat& frame_b);
    void processFrame(const cv::Mat& image);

    const std::vector<Eigen::Vector4f>& getRotations();
    const std::vector<Eigen::Vector3f>& getTranslations();
};
