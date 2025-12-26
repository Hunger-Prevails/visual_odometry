# include <filesystem>
# include <vector>
# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>

namespace fs = std::filesystem;

class Matcher {
    cv::Ptr<cv::DescriptorMatcher> matcher;

public:
    Matcher();

    std::vector<cv::DMatch> match(cv::Mat& descriptors_a, cv::Mat& descriptors_b);
    std::vector<cv::DMatch> match_knn(cv::Mat& descriptors_a, cv::Mat& descriptors_b, int n_neighbors = 2, float ratio_threshold = 0.75);

    std::vector<cv::DMatch> enforce_bijection(const std::vector<cv::DMatch>& matches);
};
