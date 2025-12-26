# include <map>
# include <filesystem>
# include <vector>
# include <queue>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class Matcher;
class ImageLoader;
class Extractor;


class Keyframe {
public:
    size_t frame;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    std::map<int, int> feature_to_landmark;
};


class Odometer {
protected:
    bool is_initialized;
    int n_keyframes;
    int temporal_baseline;
    fs::path write_path;

    Eigen::Matrix3d intrinsics;
    std::shared_ptr<ImageLoader> loader;
    std::unique_ptr<Extractor> extractor;
    std::unique_ptr<Matcher> matcher;

    std::map<int, Eigen::Quaterniond> rotations;
    std::map<int, Eigen::Vector3d> translations;

    std::queue<std::shared_ptr<Keyframe>> keyframes;
    std::vector<cv::Point3f> landmarks;

public:
    Odometer(Eigen::Matrix3d intrinsics, std::shared_ptr<ImageLoader> loader, fs::path write_path, int temporal_baseline = 10, int n_keyframes = 2, int count_features = 2000);
    ~Odometer();

    void initialize();
    void processFrame(int index);
    void processFrames();

    const std::vector<Eigen::Quaterniond> getRotations() const;
    const std::vector<Eigen::Vector3d> getTranslations() const;

protected:
    std::tuple<cv::Mat, cv::Mat, cv::Mat> computePose(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches
    ) const;

    std::vector<Eigen::Vector3d> triangulate(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches,
        const cv::Mat& rotation,
        const cv::Mat& translation
    ) const;

    std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_to_points(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches
    ) const;

    void bundle_adjustment(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches,
        std::vector<Eigen::Vector3d>& landmarks,
        Eigen::Quaterniond& rotation,
        Eigen::Vector3d& translation
    ) const;

    Eigen::Quaterniond to_eigen_rotation(const cv::Mat& rotation) const;
    Eigen::Vector3d to_eigen_translation(const cv::Mat& translation) const;
};
