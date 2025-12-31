# include <map>
# include <filesystem>
# include <vector>
# include <deque>
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
    static const int perspective_iterations;
    static const float perspective_error;
    static const float perspective_confidence;

    static const Eigen::Quaterniond rotation_initial;
    static const Eigen::Vector3d translation_initial;

    bool is_initialized;

    int count_keyframes;
    int temporal_baseline;

    float essential_confidence;
    float essential_error;
    float essential_error_initial;
    float tolerance_function;
    float tolerance_gradient;
    float tolerance_parameter;
    float track_ratio;

    fs::path write_path;

    Eigen::Matrix3d intrinsics;
    std::shared_ptr<ImageLoader> loader;
    std::unique_ptr<Extractor> extractor;
    std::unique_ptr<Matcher> matcher;

    std::map<int, Eigen::Quaterniond> rotations;
    std::map<int, Eigen::Vector3d> translations;

    std::deque<std::shared_ptr<Keyframe>> keyframes;
    std::vector<Eigen::Vector3d> landmarks;

public:
    Odometer(
        Eigen::Matrix3d intrinsics,
        std::shared_ptr<ImageLoader> loader,
        fs::path write_path,
        int count_features = 2000,
        int count_keyframes = 2,
        int temporal_baseline = 10,
        float essential_confidence = 0.99,
        float essential_error = 2.0,
        float essential_error_initial = 1.0,
        float tolerance_function = 1e-6,
        float tolerance_gradient = 1e-10,
        float tolerance_parameter = 1e-8,
        float test_ratio = 0.75,
        float track_ratio = 0.6
    );
    ~Odometer();

    void initialize();
    void process_frame(int frame, bool allow_keyframe = true);
    void process_frames();

    const std::vector<Eigen::Quaterniond> getRotations() const;
    const std::vector<Eigen::Vector3d> getTranslations() const;

protected:
    std::vector<bool> landmarks_to_freeze(const std::shared_ptr<Keyframe>& keyframe) const;

    std::unordered_map<int, int> create_map(const std::vector<cv::DMatch>& matches, std::shared_ptr<Keyframe>& keyframe) const;

    std::pair<std::vector<cv::DMatch>, std::vector<cv::DMatch>> track_or_chart(
        const std::shared_ptr<Keyframe>& keyframe,
        const std::vector<cv::DMatch>& matches
    ) const;

    std::tuple<Eigen::Quaterniond, Eigen::Quaterniond, Eigen::Vector3d, Eigen::Vector3d, std::vector<cv::DMatch>> compute_pose_initial(
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches
    ) const;

    std::tuple<std::vector<cv::DMatch>, Eigen::Quaterniond, Eigen::Vector3d> compute_pose(
        const std::shared_ptr<Keyframe>& keyframe,
        const std::vector<cv::KeyPoint>& keypoints,
        const std::vector<cv::DMatch>& matches,
        const Eigen::Quaterniond& rotation,
        const Eigen::Vector3d& translation
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

    std::pair<std::vector<Eigen::Vector3d>, std::vector<cv::DMatch>> triangulate(
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

    void bundle_adjustment(std::vector<bool>& to_freeze);
};
