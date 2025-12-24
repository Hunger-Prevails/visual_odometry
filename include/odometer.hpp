# include <map>
# include <filesystem>
# include <vector>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class Matcher;
class ImageLoader;
class Extractor;

class Odometer {
protected:
    bool is_initialized;
    int temporal_baseline;
    fs::path write_path;

    Eigen::Matrix3d intrinsics;
    std::shared_ptr<ImageLoader> loader;
    std::unique_ptr<Extractor> extractor;
    std::unique_ptr<Matcher> matcher;

    std::map<int, Eigen::Quaternionf> rotations;
    std::map<int, Eigen::Vector3f> translations;

public:
    Odometer(Eigen::Matrix3d intrinsics, std::shared_ptr<ImageLoader> loader, fs::path write_path, int temporal_baseline = 10, int count_features = 2000);
    ~Odometer();

    void initialize();
    void processFrame(int index);
    void processFrames();

    std::tuple<cv::Mat, cv::Mat, cv::Mat> computePose(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches
    );

    std::vector<cv::Point3f> triangulate(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches,
        const cv::Mat& rotation,
        const cv::Mat& translation
    );

    std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_to_points(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches
    );

    const std::vector<Eigen::Quaternionf> getRotations();
    const std::vector<Eigen::Vector3f> getTranslations();
};
