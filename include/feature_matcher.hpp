# include <filesystem>
# include <vector>
# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>

namespace fs = std::filesystem;

class Matcher {
protected:
    cv::Ptr<cv::DescriptorMatcher> matcher;

    float test_ratio;

    static const int n_neighbors;

public:
    Matcher(float test_ratio);

    std::vector<cv::DMatch> match(cv::Mat& descriptors_a, cv::Mat& descriptors_b);
    std::vector<cv::DMatch> match_knn(cv::Mat& descriptors_a, cv::Mat& descriptors_b);

    std::vector<cv::DMatch> enforce_bijection(const std::vector<cv::DMatch>& matches);
};
