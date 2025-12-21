# include <opencv2/opencv.hpp>
# include <opencv2/features2d.hpp>

class Matcher {
    cv::Ptr<cv::DescriptorMatcher> matcher;

public:
    Matcher();
    void match(cv::Mat& descriptors_a, cv::Mat& descriptors_b, std::vector<cv::DMatch>& matches);
};
