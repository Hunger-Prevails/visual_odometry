# include <filesystem>
# include <vector>
# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>

namespace fs = std::filesystem;

class Matcher {
    cv::Ptr<cv::DescriptorMatcher> matcher;

public:
    Matcher();

    void match(cv::Mat& descriptors_a, cv::Mat& descriptors_b, std::vector<cv::DMatch>& matches);

    void paint_matches(
        const cv::Mat& image_a,
        const cv::Mat& image_b,
        const std::vector<cv::KeyPoint>& keypoints_a,
        const std::vector<cv::KeyPoint>& keypoints_b,
        const std::vector<cv::DMatch>& matches,
        const fs::path& write_path
    );
};
