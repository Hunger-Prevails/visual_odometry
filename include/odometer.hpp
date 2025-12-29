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

    std::unordered_map<int, int> feature_to_landmark;
};


class Odometer {
protected:
    bool is_initialized;

    fs::path write_path;

    int n_keyframes;
    int temporal_baseline;
    float track_ratio;
    float function_tolerance;

    Eigen::Matrix3d intrinsics;
    std::shared_ptr<ImageLoader> loader;
    std::unique_ptr<Extractor> extractor;
    std::unique_ptr<Matcher> matcher;

    std::map<int, Eigen::Quaterniond> rotations;
    std::map<int, Eigen::Vector3d> translations;

    std::queue<std::shared_ptr<Keyframe>> keyframes;
    std::vector<Eigen::Vector3d> landmarks;

    static const float essential_error;
    static const float essential_confidence;

    static const int perspective_iterations;
    static const float perspective_error;
    static const float perspective_confidence;

    static const Eigen::Quaterniond rotation_initial;
    static const Eigen::Vector3d translation_initial;

public:
    Odometer(
        Eigen::Matrix3d intrinsics,
        std::shared_ptr<ImageLoader> loader,
        fs::path write_path,
        int temporal_baseline = 10,
        int n_keyframes = 2,
        int count_features = 2000,
        float test_ratio = 0.75,
        float track_ratio = 0.6,
        float function_tolerance = 1e-3
    );
    ~Odometer();

    void initialize();
    void process_frame(int frame, bool allow_keyframe = true);
    void process_frames();

    const std::vector<Eigen::Quaterniond> getRotations() const;
    const std::vector<Eigen::Vector3d> getTranslations() const;

protected:
    std::unordered_map<int, int> create_map(const std::vector<cv::DMatch>& matches, std::shared_ptr<Keyframe>& keyframe) const;

    std::pair<cv::Mat, cv::Mat> fetch_camera_pose(int frame) const;

    std::pair<std::vector<cv::DMatch>, std::vector<cv::DMatch>> track_or_chart(
        const std::shared_ptr<Keyframe>& keyframe,
        const std::vector<cv::DMatch>& matches
    ) const;

    std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Vector3d, Eigen::Vector3d, std::vector<cv::DMatch>> compute_pose_initial(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches
    ) const;

    std::vector<cv::DMatch> compute_pose(
        const std::shared_ptr<Keyframe>& keyframe,
        const std::vector<cv::KeyPoint>& keypoints,
        const std::vector<cv::DMatch>& matches,
        cv::Mat& rotation,
        cv::Mat& translation
    ) const;

    std::vector<cv::DMatch> epipolar_check(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches,
        const Eigen::Quaterniond& rotation_a,
        const Eigen::Quaterniond& rotation_b,
        const Eigen::Vector3d& translation_a,
        const Eigen::Vector3d& translation_b
    ) const;

    std::vector<Eigen::Vector3d> triangulate(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches,
        const Eigen::Quaterniond& rotation_a,
        const Eigen::Quaterniond& rotation_b,
        const Eigen::Vector3d& translation_a,
        const Eigen::Vector3d& translation_b
    ) const;

    std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_to_keypoints(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches
    ) const;

    std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>> keypoints_to_landmarks(
        const std::shared_ptr<Keyframe>& keyframe,
        const std::vector<cv::KeyPoint>& keypoints,
        const std::vector<cv::DMatch>& matches
    ) const;

    void bundle_adjustment_initial(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches,
        std::vector<Eigen::Vector3d>& landmarks,
        Eigen::Quaterniond& rotation_a,
        Eigen::Quaterniond& rotation_b,
        Eigen::Vector3d& translation_a,
        Eigen::Vector3d& translation_b
    ) const;
};
